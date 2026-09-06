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

### TASK3
### Observed values

```
$ uv run python task_3.py                                                   
 rescore |  avg_precision |  avg_query_time_ms
------------------------------------------------
    True |         0.9990 |              10.85
   False |         0.8470 |               5.72
```

### Reflection


### Reflection

Enabling scalar (int8) quantization on the `arxiv_papers` collection and comparing `rescore=True` against `rescore=False` produced a clear and theory-consistent gap in both precision and query time.

With `rescore=True`, average precision@10 was 0.9990 -- essentially matching the exact/HNSW baselines from Tasks 1 and 2 -- at an average query time of 10.85 ms. This is exactly what rescoring is designed to do: the coarse quantized stage first retrieves `limit * oversampling = 10 * 2.0 = 20` candidates using the cheap int8-quantized vectors, and then, because `rescore=True`, Qdrant fetches the original full-precision vectors for those 20 candidates and recomputes their exact distances, returning the top 10 from that refined ranking. The extra reads against full-precision vectors and the recomputation step explain the higher latency compared to the non-rescored run.

With `rescore=False`, average precision@10 dropped to 0.8470, and average query time fell to 5.72 ms. Without the correction step, the top 10 are returned directly from the coarse int8-quantized similarity scores. Quantization maps the original float32 vector components onto a much smaller set of discrete int8 values (clipped at the 99th percentile per the `quantile=0.99` setting), so distances computed purely on the quantized representation are a noticeably less faithful approximation of the true distances in the original embedding space. The measured ~15 percentage-point drop in precision is a direct, quantifiable illustration of that information loss -- roughly 1 to 2 of every 10 returned results would differ from the true top-10 on average when skipping rescoring.

Together, the two runs show the accuracy/speed trade-off promised by the theory: `rescore=True` recovers almost all of the precision quantization would otherwise cost, at roughly 90% higher query latency (10.85 ms vs. 5.72 ms) than `rescore=False`. Whether this trade-off is worth it depends on the use case -- if the application requires near-exact recall, rescoring is close to mandatory once quantization is enabled; if a coarser but faster search is acceptable (e.g., as a first-pass filter in a larger pipeline), disabling rescoring is a legitimate way to trade away some precision for meaningfully lower latency.

**Note on an earlier, incorrect measurement:** an initial run of this experiment showed identical precision (0.9990) for both `rescore=True` and `rescore=False`, which is not possible in theory, since rescoring should only ever help or be neutral, never make results identical when quantization is genuinely lossy. The cause was that `update_collection` with a `quantization_config` triggers a background optimization job to actually build the quantized vectors, and the script was issuing queries before that job had finished -- so both settings were effectively still searching against full-precision vectors, making `rescore` a no-op. The fix was to poll `client.get_collection(...)` and wait for `status == CollectionStatus.GREEN` before running any timed queries, ensuring the quantized index was fully built. Re-running after adding that wait produced the results shown above, which are the ones consistent with the underlying theory.
