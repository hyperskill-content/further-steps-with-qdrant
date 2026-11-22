# Task 1 — Catching Up

### Observed values

```
Average precision@10: 1.0000
Average ANN query time: 24.40 ms
Average exact k-NN query time: 40.02 ms
```

(100 queries from `dataset/queries_embeddings.json` against local `arxiv_papers`, ~411,120 points, Cosine / HNSW defaults. Script: `task_1.py`.)

### Reflection

**What the numbers say**

1. **Precision@10 = 1.0** — For every test query, the ANN (default HNSW) top-10 set was identical to the exact k-NN top-10. On this collection and this probe set, the approximate index did not drop any true nearest neighbors inside the top-10 window. Precision@k here equals recall@k because the golden set size is also *k*.

2. **ANN is faster than exact** — ~24 ms vs ~40 ms average (~1.6×). That matches the theory: HNSW walks a sparse graph of neighbors instead of comparing the query to every vector. Exact search (`SearchParams(exact=True)`) is the baseline “scan for the true top-k”; it is the right tool for *evaluation*, not for production latency SLOs at this scale.

3. **Why might precision be perfect?** Several reinforcing reasons:
   - HNSW with Qdrant’s default construction/`ef` is already high-recall for Cosine on this embedding space.
   - *k = 10* is a small window; ANN errors often show up deeper in the ranking before they hit the top-10.
   - The 100 queries are in-distribution relative to the arXiv ML corpus the collection was built from (same `text-embedding-ada-002` space), so the neighborhood geometry is well covered by the graph.

4. **How to read this cautiously**
   - This is a **sanity check**, not a full IR eval: relevance is defined as “exact neighbor,” not human labels.
   - Perfect precision@10 does **not** mean ANN equals exact for *all* ranks or for adversarial queries; try larger *k*, harder queries, or degraded HNSW settings (Task 2’s `ef` tuning) and the gap usually appears.
   - Absolute milliseconds depend on machine load, Qdrant config, and whether vectors are on disk vs warm cache — ratios matter more than single-run absolutes.

**Takeaway for the rest of the project:** Default ANN already matches exact top-10 here while staying faster. Task 2 will push on *latency vs precision* by changing search parameters (`ef`, etc.). Task 3 will trade *storage / speed* via quantization, which can also move precision.

---

# Task 2 — Balance the Search

### Observed values

```
results =
[
  {"hnsw_ef": 10,  "avg_precision": 0.968, "avg_query_time_ms": 3.976},
  {"hnsw_ef": 20,  "avg_precision": 0.991, "avg_query_time_ms": 2.965},
  {"hnsw_ef": 50,  "avg_precision": 0.996, "avg_query_time_ms": 3.635},
  {"hnsw_ef": 100, "avg_precision": 1.000, "avg_query_time_ms": 4.102},
  {"hnsw_ef": 200, "avg_precision": 1.000, "avg_query_time_ms": 5.321},
]
```

(100 queries; exact k-NN ground truth; cache warm-up before the sweep. Script: `task_2.py`.)

### Reflection

**What `hnsw_ef` is**

During ANN search, HNSW keeps a candidate priority queue of size `ef` (`hnsw_ef` in Qdrant). A larger queue means the walker explores more neighbors before stopping — more chance to find the true nearest neighbors, more work per query.

**What the numbers say**

1. **Precision rises with `hnsw_ef`** — At `ef=10`, average precision@10 is **0.968** (ANN misses some exact neighbors). By `ef=20` we are at **0.991**, `ef=50` at **0.996**, and `ef=100` / `200` reach **1.000**. That is the classic recall/precision–vs–exploration trade-off.

2. **Latency tends to grow once the cache is warm** — From `ef=20` → `200`, average query time went ~3.0 ms → ~5.3 ms. Exploring a bigger candidate list costs CPU. Absolute ms are small on this machine with a warm collection; the *direction* of the trade-off matters more than the absolute numbers.

3. **Why `ef=10` looked slower than `ef=20` in this run** — Even with warm-up queries, the first timed setting can still absorb a bit of residual cold cost or timing noise. We discarded the first query of each sweep and still see that quirk at `ef=10`. Do not over-interpret a single cell; the stable story is: **higher ef → higher precision**, and **higher ef → generally higher latency** once past the noisy first setting.

4. **Practical takeaway** — Default / mid-range `ef` (here ~50–100) already gets near-perfect top-10 agreement with exact search on this dataset. Pushing to 200 buys no more precision@10 but costs more time. For production you’d pick the smallest `ef` that meets your quality SLO.

**Link to Task 1:** Task 1’s default ANN (implicit default `ef`) already hit precision 1.0. Task 2 shows that if you *force* a small `ef` (10), quality drops — so the knob is real.

---

# Task 3 — Quantization

### Observed values

```
results =
[
  {"rescore": True,  "avg_precision": 0.998, "avg_query_time_ms": 3.219},
  {"rescore": False, "avg_precision": 0.836, "avg_query_time_ms": 3.478},
]
```

(100 queries; exact ground truth with `QuantizationSearchParams(ignore=True)`; scalar INT8, `quantile=0.99`, `always_ram=False`; ANN with `oversampling=2.0`. Collection waited until status=green before measuring. Script: `task_3.py`.)

### Reflection

**What quantization does here**

Scalar INT8 maps each float component onto 256 buckets (clipped by the 99th-percentile quantile so outliers don’t stretch the range). Searches then use those cheap integers by default. Original float vectors stay available for exact search and for rescoring.

**What the numbers say**

1. **`rescore=True` almost matches exact** — Average precision@10 **0.998**. Coarse stage pulls `k * oversampling = 20` candidates from quantized scores; then Qdrant re-ranks those 20 with full-precision distances and returns the top 10. That recovers nearly all of the accuracy quantization would otherwise cost.

2. **`rescore=False` drops quality hard** — Precision falls to **0.836**. Ranking is decided only by INT8 similarity, which is a coarser signal than float Cosine — true neighbors fall out of the top-10 often enough that ~16% of slots are wrong on average.

3. **Latency is in the same ballpark** — ~3.2 ms with rescore vs ~3.5 ms without in this warm-cache run. Rescoring 20 vectors is cheap relative to the HNSW walk, so you do **not** get a meaningful speed win by turning rescore off here — you mainly lose accuracy. (On larger oversampling factors, colder disk, or bigger *k*, rescore cost would show up more clearly.)

4. **Why `ignore=True` on the baseline** — After quantization is configured, default search uses quantized vectors. Exact ground truth must force full-precision (`ignore=True` + `exact=True`), or the “golden” set itself would be quantized and the comparison would be meaningless.

5. **Practical takeaway** — Scalar INT8 + `rescore=True` + modest oversampling is the production-friendly default: smaller/faster candidate scoring, accuracy restored on a shortlist. Skipping rescore is only attractive when you explicitly accept lower recall for absolute cheapest ranking.

**Link to Tasks 1–2:** Task 1 showed default ANN already matches exact top-10. Task 2 showed search-time `hnsw_ef` can break that if set too low. Task 3 shows a *representation* trade-off: compress vectors for efficiency, then use **rescore** (not larger `ef` alone) to buy accuracy back.
