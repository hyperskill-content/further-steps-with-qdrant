### Observed values 

Which patterns are present? How can they be explained? Describe your findings in detail here.

### TASK1
### Observed values
```
$ uv run python task_1.py Average precision@10: 0.9990 Average ANN query time: 28.09 ms Average exact k-NN query time: 76.37 ms
```
### Reflection

The results show the expected speed/accuracy trade-off between approximate (HNSW) and exact k-NN search. With the HNSW index built over all 411,120 points in the `arxiv_papers` collection, the approximate search averaged 28.09 ms per query, compared to 76.37 ms for the exact brute-force search — roughly 2.7x faster. This matches the theory: HNSW avoids scanning the full dataset by traversing a navigable graph structure, at the cost of only approximate correctness, while exact search must compare the query against every stored vector.

Despite being much faster, the approximate search was still highly accurate, with an average precision@10 of 0.9990 across the 100 test queries. Since precision@10 coincides with recall@10 in this setup (the exact top-10 is used as the golden set), this means that out of 1000 total retrieved slots across all queries, the ANN search returned essentially the same items as the exact search would have, missing only about one item total. This level of agreement is consistent with the default HNSW parameters (`m=16`, `ef_construct=100`) providing a strong recall/speed balance out of the box for a dataset of this size and dimensionality (1536-dim, cosine distance).

One notable observation from the process itself: initially, before the collection's HNSW index was built (indexing had been disabled via `indexing_threshold=0` during the original bulk ingestion, likely as a workaround for indexing errors under heavy write load), both "ANN" and "exact" search were effectively performing the same brute-force scan, yielding precision@10 = 1.0000 and near-identical (and in fact reversed) timings. This underscored that a collection's optimizer configuration directly determines whether HNSW is actually in effect, and that precision/speed metrics are only meaningful once the intended index is confirmed to be built — a useful sanity check for evaluating real deployments as well.

### TASK2
### Observed values
```
$ uv run python task_2.py
 hnsw_ef |  avg_precision |  avg_query_time_ms
------------------------------------------------
      10 |         0.9640 |               5.76
      20 |         0.9910 |               6.00
      50 |         0.9980 |               6.80
     100 |         0.9990 |               6.89
     200 |         1.0000 |              10.17 
```

### Reflection

The results show a clear and expected trend: as `hnsw_ef` increases, `avg_precision@10` climbs monotonically toward 1.0. This matches the theory behind the parameter — `hnsw_ef` controls the size of the candidate priority queue explored during HNSW's graph traversal at query time, so a larger value means the search considers a broader set of neighbors before returning its top-k, making it less likely to prematurely discard a node that would have led to a true nearest neighbor. At `hnsw_ef=10`, precision was only 0.9640, meaning the search missed roughly 3.6% of the true top-10 results on average. By `hnsw_ef=200`, precision reached a perfect 1.0000, meaning the approximate search matched the exact k-NN search exactly across all 100 test queries.

The gains in precision are front-loaded and show diminishing returns as `hnsw_ef` grows. Moving from `ef=10` to `ef=20` bought +0.027 precision (0.9640 -> 0.9910) for essentially no extra query time (5.76 ms -> 6.00 ms). Moving from `ef=20` to `ef=50` bought a further +0.007 for +0.8 ms. By `ef=100`, precision had already reached 0.9990 — effectively saturated — and closing the remaining 0.001 gap to reach perfect precision at `ef=200` required a disproportionate jump in query time, from 6.89 ms to 10.17 ms (about a 48% increase).

Query time itself stayed roughly flat and increased gently and near-linearly across `ef=10, 20, 50, 100` (5.76 ms -> 6.89 ms), before jumping sharply at `ef=200`. This is consistent with the mechanics of the algorithm: for smaller candidate lists, the extra exploration is cheap relative to fixed per-query overhead (e.g., network round-trip, deserialization), but past a certain point the graph exploration cost begins to dominate and scale more visibly with `ef`.

Overall, the sweep illustrates a textbook diminishing-returns curve for the precision/speed trade-off controlled by `hnsw_ef`. The "knee" of the curve sits roughly around `ef=50` to `ef=100`, where the search already achieves 99.8-99.9% of exact search's accuracy at a query time cost only marginally higher than the cheapest setting tested. Pushing to `ef=200` to close the last 0.1% precision gap costs nearly 50% more query time for a difference (0.999 vs. 1.000 precision) that is unlikely to be noticeable in most real-world applications. This suggests that, for this dataset and collection configuration, a mid-range `hnsw_ef` value (e.g., 50-100) offers the best practical balance between search accuracy and latency, rather than defaulting to the highest tested value.

### TASK2
### Observed values

```
$ uv run python task_3.py
 rescore |  avg_precision |  avg_query_time_ms
------------------------------------------------
    True |         0.9990 |               7.81
   False |         0.9990 |               7.34
```

### Reflection

### Reflection

Enabling scalar (int8) quantization on the `arxiv_papers` collection produced a striking result: both `rescore=True` and `rescore=False` achieved the same average precision@10 of 0.9990 -- identical to the plain (non-quantized) HNSW search from Task 2 at its default `hnsw_ef`. In other words, compressing each float32 vector component down to an 8-bit integer cost essentially no measurable retrieval accuracy on this dataset, even without the exact-distance rescoring step that is supposed to recover most of the accuracy quantization would otherwise sacrifice.

The timing difference between the two settings was small but in the expected direction: `rescore=True` took 7.81 ms on average versus 7.34 ms for `rescore=False`, a roughly 6% overhead. This matches the theory -- with `rescore=True`, Qdrant must fetch the original full-precision vectors for the oversampled candidate pool (`limit * oversampling = 10 * 2.0 = 20` candidates) and recompute exact distances on them before returning the top 10, whereas `rescore=False` returns the top 10 directly from the coarse quantized scores, skipping that extra read-and-recompute step.

The fact that skipping rescoring cost nothing in precision here suggests that int8 scalar quantization with `quantile=0.99` preserves the *relative ordering* of vectors extremely well for this embedding model (`text-embedding-ada-002`), at least for the top-10 neighborhood of these particular queries. This is consistent with the task's own framing of scalar quantization as "perhaps the most universal [type] in terms of retaining high accuracy." It also lines up with the note that binary and product quantization are far more lossy by comparison -- scalar quantization's 8-bit resolution per dimension is evidently enough to keep the coarse ranking of nearby vectors essentially intact, so the more expensive rescoring step becomes a safety net that, in this case, wasn't needed to hit near-perfect precision.

Practically, this implies that for this dataset and embedding model, it would be reasonable to run with `rescore=False` in production to shave a small amount of latency off every query, since the measured accuracy cost was zero. However, this conclusion is specific to `oversampling=2.0` and this particular set of 100 test queries; a smaller oversampling value, a different `quantile`, or queries drawn from a different, more challenging distribution of documents could plausibly widen the gap between rescoring and not rescoring, so this result shouldn't be assumed to generalize without re-testing under those conditions.
