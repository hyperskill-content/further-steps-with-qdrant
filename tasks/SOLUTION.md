### Observed values 

```
Average precision@10: 0.9990
Average ANN query time: 6.37 ms
Average exact k-NN query time: 40.56 ms
```

### Reflection

`evaluate_precision` in `tasks/task_1.py` loads the 100 query embeddings from `dataset/queries_embeddings.json` and, for each one, calls `query_points` on `arxiv_papers` twice with `limit=10`. The first call is the default approximate search. The second passes `SearchParams(exact=True)`. Precision for that query is `|ANN ids ∩ exact ids| / 10`, the times are the wall-clock duration of each call, and `result_formatting` prints the three averages.

Those averages show one pattern: the default search returns the exact top-10, and it does it much faster. Precision@10 is 0.9990, so out of 1,000 retrieved ids only one disagrees with the exact result. The approximate call averages 6.37 ms and the exact call averages 40.56 ms, about 6.4× slower. The collection behind those times is the existing `arxiv_papers` index: 411,120 vectors, size 1536, cosine distance, HNSW `m=16`, `ef_construct=100`.

The 0.0010 shortfall is a single query, "Zero-shot learning in computer vision applications". Ranks 1–9 are the same papers in the same order. At rank 10 the approximate search returns *Multi-Cue Zero-Shot Learning with Strong Supervision* (score 0.871422) and the exact search returns *Lensless-camera based machine learning for image classification* (score 0.871458). The cosine scores differ by about 3.6e-5, so the default HNSW search stopped on a neighbor sitting immediately beside the true 10th hit. The other 99 queries match id for id and in the same order, which is why the printed average is 0.9990.

The time gap is the same comparison on this collection. `exact=True` scores all 411,120 stored vectors and lands near 40 ms. The default call uses the HNSW index already built for `arxiv_papers` and lands near 6 ms, while still matching the exact top-10 on 99 of the 100 queries.

Overall, for this collection the default approximate search gave the same results as exact search for these queries in a fraction of the time, so it looks like the right choice. The one difference is the rank-10 near-tie above; the other 99 queries matched exactly.

A few things to keep in mind:

- The times include the whole round trip from the script to Qdrant and back, not just the search itself. `time.time()` wraps each `query_points` call, so 6.37 ms and 40.56 ms include the client and the local trip. Both calls pay that cost, which is why the ~6.4× gap is still the comparison that matters.
- This was one run on this machine. An earlier run of the same script was 6.35 ms and 39.90 ms, with precision still 0.9990. The times move a little; the precision did not.
- 100 queries is a small test set. 0.9990 means the approximate search missed one neighbor across these queries, not that it never misses.
