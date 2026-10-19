### Observed values 

```
rescore=True oversampling=2.0
Average precision@10: 0.9990
Average ANN query time: 7.53 ms
Average exact k-NN query time: 39.92 ms
queries_short_of_exact: 1 of 100
precision_counts: 0.9:1, 1.0:99
rescore=False oversampling=2.0
Average precision@10: 0.8500
Average ANN query time: 5.69 ms
Average exact k-NN query time: 39.46 ms
queries_short_of_exact: 86 of 100
precision_counts: 0.5:2, 0.6:4, 0.7:11, 0.8:22, 0.9:47, 1.0:14
```

### Reflection

`tasks/task_3.py` calls `update_collection` on `arxiv_papers` with scalar quantization `int8`, `quantile=0.99`, `always_ram=False`. The collection went yellow while Qdrant built the int8 vectors, then returned to green. The original 411,120 cosine vectors are still there (size 1536, HNSW `m=16`, `ef_construct=100`). `hnsw_ef` was left at the default.

`evaluate_quantized_search` then runs twice on the 100 embeddings in `dataset/queries_embeddings.json`, first with `rescore=True`, then with `rescore=False`. Both use `oversampling=2.0`, so the quantized stage retrieves `10 * 2.0 = 20` candidates. With `rescore=True`, Qdrant recomputes those 20 scores on the original vectors and returns the top 10. With `rescore=False`, the top 10 come from the int8 scores. The exact side of each comparison is `exact=True` and `QuantizationSearchParams(ignore=True)`, so the baseline is a full scan of the original vectors. Before each timed pass the script sends 10 untimed approximate queries with that rescore setting and one untimed exact query. Precision is `|ANN ids ∩ exact ids| / 10`.

With rescore on, precision@10 is 0.9990: 99 queries at 1.0 and one at 0.9. The approximate call averages 7.53 ms. The exact call averages 39.92 ms. The 0.9 query is "Deep belief networks and their applications". Its exact top-10 includes *Deep Belief Networks used on High Resolution Multichannel Electroencephalography Data for Seizure Detection* (cosine 0.864383). The rescored search includes *Auto-encoding a Knowledge Graph Using a Deep Belief Network: A Random Fields Perspective* (cosine 0.857705) in that slot. Checked separately on this same collection, "Zero-shot learning in computer vision applications" has overlap 10 and the same order with `rescore=True`.

With rescore off, precision@10 is 0.8500 and the approximate call averages 5.69 ms. The exact call averages 39.46 ms. 86 queries are short of the exact top-10: 14 at 1.0, 47 at 0.9, 22 at 0.8, 11 at 0.7, 4 at 0.6, and 2 at 0.5. The same zero-shot query has overlap 6 with `rescore=False`.

Overall, for these 100 queries, `rescore=True` with `oversampling=2.0` prints 0.9990 at 7.53 ms, next to an exact scan at 39.92 ms. `rescore=False` prints 0.8500 at 5.69 ms. The difference between the two approximate times is 1.84 ms.

A few things to keep in mind:

- 7.53 ms, 5.69 ms, and the ~40 ms exact times are `time.time()` around the whole `query_points` call, including the client and the trip to local Qdrant.
- This is the one run printed above. The exact times are 39.92 ms and 39.46 ms because both passes scan the same original vectors with `ignore=True`.
- 100 queries. 0.9990 is one missed id on this file. 0.8500 is 86 queries short on this file.
- The int8 vectors were added next to the original ones. This run does not turn the stored vectors into int8, and it does not measure a storage change.
