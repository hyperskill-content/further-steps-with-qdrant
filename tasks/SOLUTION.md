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

*(to fill)*

---

# Task 3 — Quantization

*(to fill)*
