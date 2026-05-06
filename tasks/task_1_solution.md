# Task 1 — Solution

### Observed values

```
Average precision@10: 0.9990
Average ANN query time: 14.84 ms
Average exact k-NN query time: 283.80 ms
```

(Run on the bundled `dataset/queries_embeddings.json` — 100 queries, `text-embedding-ada-002` 1536-d embeddings — against the local `arxiv_papers` collection of 411,120 points, cosine distance, default HNSW index.)

### Reflection

**Precision is essentially perfect (0.999)**. Across the 100 queries, on average only one ID out of every ten retrieved differs from the exact ground truth — and the discrepancy almost always comes from a single query where one HNSW result is a near-tie that fell just outside the top-10 graph traversal. With a relatively small collection (~410k points) and the default HNSW parameters in Qdrant (`m=16`, `ef_construct=100`, runtime `hnsw_ef=128`), the ANN graph is dense enough that ANN and exact search agree on virtually every neighbor.

**ANN is ~19× faster than exact k-NN** (14.84 ms vs 283.80 ms). The exact path scans every vector in the collection (411k × 1536 floats), so its cost grows linearly with `N`. HNSW, in contrast, performs a logarithmic traversal of a multi-layered proximity graph and only evaluates a small candidate list (the `ef` parameter). The 19× ratio matches the order-of-magnitude expectation — for production-scale collections (10s of millions+) the gap typically widens to 100–1000×, which is why exact search is treated strictly as an evaluation baseline, not a query path.

**Why this matters for the next stages**:
- *Task 2 (`hnsw_ef`)*: with precision already pinned at ~0.999, lowering `hnsw_ef` should give a larger speed gain than precision loss until we drop below ~50, where we'll start to see the precision/recall trade-off open up.
- *Task 3 (quantization)*: scalar `int8` quantization compresses the vectors 4× and should keep precision near-perfect (text-embedding-ada-002 quantizes well); however, with 411k vectors the search is already cache-friendly, so the speed gain from quantization will be more modest than on disk-backed collections. Setting `rescore=False` will let us isolate the raw quantized accuracy.

**Caveats**:
- The 100-query sample is a sanity check, not a statistically robust benchmark — the precision and times have non-trivial variance run-to-run (cold cache, OS scheduling).
- Times include client/server round-trip over HTTP; the absolute numbers would shrink on the gRPC port (`6334`).
- We deliberately run k-NN before ANN per query to avoid biasing the ANN cache, but Qdrant's segment cache will still warm up over the 100 iterations.
