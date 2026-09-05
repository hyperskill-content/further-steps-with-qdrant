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
