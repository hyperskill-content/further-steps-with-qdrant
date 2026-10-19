### Observed values 

```
hnsw_ef=10: avg_precision=0.9550, avg_query_time_ms=2.62
hnsw_ef=20: avg_precision=0.9840, avg_query_time_ms=2.66
hnsw_ef=50: avg_precision=0.9990, avg_query_time_ms=3.34
hnsw_ef=100: avg_precision=0.9990, avg_query_time_ms=3.85
hnsw_ef=200: avg_precision=0.9990, avg_query_time_ms=5.50
```

### Reflection

`evaluate_hnsw_ef` in `tasks/task_2.py` loads the 100 embeddings in `dataset/queries_embeddings.json`. For each query text it runs `query_points` on `arxiv_papers` with `limit=10` and `SearchParams(exact=True)`, and stores those ids in `ground_truth`. It then sends 10 untimed approximate queries at `hnsw_ef=200`. The timed sweep is `hnsw_ef` 10, 20, 50, 100, and 200: each call is `query_points` with `SearchParams(hnsw_ef=hnsw_ef)`, timed with `time.time()`, and scored as `|ANN ids ∩ ground truth ids| / 10`. The collection is the existing one: 411,120 vectors, size 1536, cosine, HNSW `m=16`, `ef_construct=100`. Nothing in the index was changed.

On this run precision is 0.9550, 0.9840, then 0.9990 from `hnsw_ef=50` through 200. At `hnsw_ef=10`, 31 of the 100 queries miss the exact top-10: 19 miss one id, 10 miss two, 2 miss three. At `hnsw_ef=20`, 15 queries are short. At 50, 100, and 200 the only miss is "Zero-shot learning in computer vision applications". Rank 10 is still *Multi-Cue Zero-Shot Learning with Strong Supervision* in the approximate result and *Lensless-camera based machine learning for image classification* in the exact result. Those are the same two papers as the default search, and `hnsw_ef=200` still does not return the exact one.

Query time on this run is 2.62 ms, 2.66 ms, 3.34 ms, 3.85 ms, and 5.50 ms. `hnsw_ef=10` and `hnsw_ef=20` are 0.04 ms apart. From 20 to 200 the time goes from 2.66 ms to 5.50 ms while precision stays at the 0.9990 that `hnsw_ef=50` already reached (3.34 ms).

Overall, for these 100 queries on this collection, `hnsw_ef=50` is the value that reaches 0.9990. `hnsw_ef=10` and `20` miss more neighbors at about the same query time. `hnsw_ef=100` and `200` take longer and leave the same rank-10 pair.

A few things to keep in mind:

- 2.62 ms through 5.50 ms are `time.time()` around the whole `query_points` call, so each number includes the client and the trip to local Qdrant and back.
- Precision was the same on the run before this one (0.9550, 0.9840, 0.9990, 0.9990, 0.9990). The times were not: that run printed 3.14, 2.92, 3.44, 4.06, and 5.52 ms, and on that run `hnsw_ef=10` was slower than `hnsw_ef=20`.
- 0.9990 is one missed id across these 100 queries. It does not say what `hnsw_ef` does on a query outside `queries_embeddings.json`.
- Only the search parameter changed. `ef_construct` is still 100 on the stored HNSW graph.
