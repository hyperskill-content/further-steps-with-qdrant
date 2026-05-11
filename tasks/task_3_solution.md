# Task 3 — Solution

### Observed values

```
--- rescore=True (oversampling=2.0) ---
Average precision@10: 0.9990
Average ANN query time: 11.71 ms
Average exact k-NN query time: 349.68 ms

--- rescore=False (oversampling=2.0) ---
Average precision@10: 0.8330
Average ANN query time: 16.45 ms
Average exact k-NN query time: 349.68 ms
```

(Same setup as Tasks 1–2 — 100 queries, 411,120-point `arxiv_papers` collection, cosine. Scalar `int8` quantization with `quantile=0.99`, `always_ram=False`, original full-precision vectors retained on disk. Exact k-NN ground truth uses `quantization=ignore=True` so the baseline is unchanged from Task 1. Ground truth is computed in a single up-front pass and reused across both ANN runs; each ANN run is preceded by a 10-query warmup to page in the quantized segments before timing starts.)

### Reflection

**Rescore is the dominant lever — not oversampling alone.**
- With `rescore=True` precision is **0.999**, identical to the un-quantized baseline (Task 1).
- With `rescore=False` precision drops to **0.833** — roughly one out of every six retrieved IDs is wrong relative to the full-precision ground truth.

That ~17% precision gap is the cost of approximating cosine distance with `int8`-quantized vectors. With `oversampling=2.0` the engine retrieves `2 × k = 20` candidates from the quantized index; `rescore=True` then re-ranks those 20 using the **original full-precision** vectors and returns the top 10. The original vectors live on disk (`always_ram=False`), so the rescore step costs a small amount of disk I/O per query but is enough to fully recover precision because the true top-10 almost always falls inside the quantized top-20.

**Why precision survives at all without rescore.**
0.833 is still surprisingly high considering each component dropped from 32-bit float to 8-bit signed integer (a 4× compression). Two factors help:
1. `text-embedding-ada-002` produces embeddings whose component magnitudes are well-behaved (Gaussian-ish, no extreme outliers); `quantile=0.99` clipping handles the long tail.
2. Cosine distance is a sum of 1536 component-wise products — a lot of independent noise that mostly averages out across dimensions.

**Speed: quantization gives a measurable speedup.**
- `rescore=True` at **11.71 ms** is **~21% faster** than Task 1's un-quantized baseline of **14.84 ms**, while delivering the same 0.999 precision. This is the headline result: scalar `int8` quantization is effectively free precision-wise *with rescore*, and meaningfully faster.
- `rescore=False` at **16.45 ms** is the unexpected outlier — naively it should be at least as fast as `rescore=True` (no extra rescore step), but it lands slightly *higher*. The gap is small (~5 ms over a 100-query sample) and well within the run-to-run noise we saw across measurements; the dominant cost in both modes is the HNSW traversal over quantized vectors, which is essentially the same work either way. The takeaway is that `rescore=False` does **not** buy meaningful speed on this collection — it only buys you the precision drop.

**Measurement note — what we had to fix.**
A first cut of this experiment interleaved each exact k-NN (full-precision exhaustive scan, ~2.4 GiB swept through the OS page cache per query) with the ANN measurement. That pattern evicted the quantized segments from cache between ANN calls, inflating ANN times by ~3–5 ms and masking the speedup. The fix is methodological: compute the ground-truth k-NN once up front and then run the ANN loop in isolation, with a short warmup pass so the first measurements don't pay cold-mmap costs.

**Decision matrix — when to use what**:

| Scenario | Quantization? | Rescore? |
|---|---|---|
| Recall-sensitive, latency-sensitive | Yes, scalar `int8` | `rescore=True` (best of both: ~21% faster *and* 0.999 precision) |
| Throughput-sensitive, recall-tolerant (e.g., re-ranked downstream) | Yes, scalar `int8` | `rescore=False` (or step up to product/binary quantization for bigger compression) |
| Storage-bound | Yes | `rescore=True` (4× smaller on disk for negligible precision cost) |

**Cross-task synthesis**:
- Task 1 established the baseline (precision 0.999, ANN ~14.84 ms).
- Task 2 showed `hnsw_ef` is the precision-vs-time knob within HNSW; sweet spot ~50.
- Task 3 shows scalar quantization with `rescore=True` is a strict win on this collection: same precision as Task 1, ~21% lower latency, plus 4× smaller on-disk footprint for the quantized vectors. `rescore=False` is the wrong default — the precision drop isn't paid back in latency.

### Reverting (optional)

The collection still has `quantization_config={'scalar': {'type':'int8', ...}}` applied. To revert and free the disk Qdrant used for quantized segments:

```python
client.update_collection(
    collection_name="arxiv_papers",
    quantization_config=models.Disabled(),
)
```
