# Task 3 — Solution

### Observed values

```
--- rescore=True (oversampling=2.0) ---
Average precision@10: 0.9990
Average ANN query time: 16.52 ms
Average exact k-NN query time: 300.04 ms

--- rescore=False (oversampling=2.0) ---
Average precision@10: 0.8320
Average ANN query time: 21.75 ms
Average exact k-NN query time: 321.83 ms
```

(Same setup as Tasks 1–2 — 100 queries, 411,120-point `arxiv_papers` collection, cosine. Scalar `int8` quantization with `quantile=0.99`, `always_ram=False`, original full-precision vectors retained on disk. Exact k-NN ground truth uses `quantization=ignore=True` so the baseline is unchanged from Task 1.)

### Reflection

**Rescore is the dominant lever — not oversampling alone.**
- With `rescore=True` precision is **0.999**, identical to the un-quantized baseline (Task 1).
- With `rescore=False` precision drops to **0.832** — roughly one out of every six retrieved IDs is wrong relative to the full-precision ground truth.

That ~17% precision gap is the cost of approximating cosine distance with `int8`-quantized vectors. With `oversampling=2.0` the engine retrieves `2 × k = 20` candidates from the quantized index; `rescore=True` then re-ranks those 20 using the **original full-precision** vectors and returns the top 10. The original vectors live on disk (`always_ram=False`), so the rescore step costs a few extra ms per query but is enough to fully recover precision because the true top-10 almost always falls inside the quantized top-20.

**Why precision survives at all without rescore.**
0.832 is still surprisingly high considering each component dropped from 32-bit float to 8-bit signed integer (a 4× compression). Two factors help:
1. `text-embedding-ada-002` produces embeddings whose component magnitudes are well-behaved (Gaussian-ish, no extreme outliers); `quantile=0.99` clipping handles the long tail.
2. Cosine distance is a sum of 1536 component-wise products — a lot of independent noise that mostly averages out across dimensions.

**Why the speed didn't improve (and even regressed slightly) vs the un-quantized baseline (~14.84 ms).**
- `rescore=True` came in at **16.52 ms** and `rescore=False` at **21.75 ms**, both *higher* than Task 1's 14.84 ms.
- This is the *small-collection* regime: 411k × 1536 × 4 bytes ≈ 2.4 GiB fits comfortably in OS page cache on this machine, so the original full-precision search is already memory-bound, not compute-bound. Quantization's main wins — fewer bytes off disk, denser SIMD packing — don't apply when the bottleneck was network/HTTP overhead and tiny in-RAM scans.
- Quantization adds the cost of decoding `int8` → distance, plus (for `rescore=True`) loading 20 originals back from disk to re-rank. On a larger or disk-bound collection (10s of millions of vectors, mmap-backed) the trade-off would flip dramatically; quantization typically yields a 2–4× speed-up there.
- The fact that `rescore=False` is *slower* than `rescore=True` is counterintuitive but small enough (~5 ms, within run-to-run variance for this 100-query sample) to be measurement noise. Both are essentially the same cost on this dataset.

**Decision matrix — when to use what**:

| Scenario | Quantization? | Rescore? |
|---|---|---|
| Fits in RAM, latency-critical, small N | No (Task 1 baseline is best) | n/a |
| Memory-bound (10s of M+ vectors), recall-sensitive | Yes, scalar `int8` | `rescore=True` |
| Memory-bound, throughput-sensitive, recall-tolerant (e.g., re-ranked downstream) | Yes, scalar `int8` | `rescore=False` (or product/binary quantization) |
| Storage-bound only | Yes | `rescore=True` (4× smaller on disk for negligible precision cost) |

**Cross-task synthesis**:
- Task 1 established the baseline (precision 0.999, ANN ~14.8 ms).
- Task 2 showed `hnsw_ef` is the precision-vs-time knob within HNSW; sweet spot ~50.
- Task 3 shows scalar quantization on this collection is "free" precision-wise *with rescore* but doesn't unlock a speed gain because RAM isn't the bottleneck. It would on a 10× larger collection.

### Reverting (optional)

The collection still has `quantization_config={'scalar': {'type':'int8', ...}}` applied. To revert and free the disk Qdrant used for quantized segments:

```python
client.update_collection(
    collection_name="arxiv_papers",
    quantization_config=models.Disabled(),
)
```
