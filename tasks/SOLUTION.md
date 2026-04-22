### Task 1 — ANN vs Exact k-NN

#### Observed values

```
Average precision@10: 0.9990
Average ANN query time: 39.50 ms
Average exact k-NN query time: 97.10 ms
```

#### Reflection

The precision@10 of 0.9990 is basically perfect — across all 100 queries, the ANN search almost always returns the exact same 10 nearest neighbors as the brute-force exact search. That's a really strong result and means the HNSW index is doing its job well for this dataset.

Speed-wise the ANN search comes in at around 39.5 ms versus 97.1 ms for the exact k-NN, so roughly 2.5x faster. That gap will only grow as the collection gets larger — exact search scales linearly with the number of vectors while HNSW is logarithmic, so the trade-off gets increasingly worthwhile at scale.

The high precision makes sense for a dataset like this. The arXiv ML papers tend to cluster naturally around topics (computer vision, NLP, reinforcement learning, etc.), and papers within a cluster are very close to each other in embedding space. That kind of structure is exactly what HNSW is designed to exploit — it builds a graph where nearby vectors are well-connected, so the approximate search rarely misses a true neighbor.

In short: for this collection, HNSW gives us essentially the same quality as an exhaustive search at a fraction of the cost. There's very little reason to use exact search in production here.

---

### Task 2 — hnsw_ef Tuning

#### Observed values

```
hnsw_ef=  10 | precision@10: 0.9280 | avg time: 3.43 ms
hnsw_ef=  20 | precision@10: 0.9750 | avg time: 3.75 ms
hnsw_ef=  50 | precision@10: 0.9930 | avg time: 4.73 ms
hnsw_ef= 100 | precision@10: 0.9990 | avg time: 5.52 ms
hnsw_ef= 200 | precision@10: 1.0000 | avg time: 6.40 ms
```

#### Reflection

There's a clear trade-off here: higher `hnsw_ef` means more candidates explored during search, which improves precision but costs a bit more time. At `hnsw_ef=10` we're already at 92.8% precision, which is decent but noticeably worse than higher values. By the time you hit `hnsw_ef=200` you're getting perfect precision (1.0000) — basically matching exact k-NN — while still running in just 6.4 ms.

The time differences are actually pretty small in absolute terms (3.4 ms vs 6.4 ms). For most applications the extra couple of milliseconds at higher `hnsw_ef` is worth the precision gain. If you were running millions of queries per second then you'd care more about the lower end, but for typical search workloads `hnsw_ef=100` or `hnsw_ef=200` seems like the sweet spot.

It's also worth noting that these times are much faster than the ANN times from task 1 (39.5 ms). That's probably because task 1 was measuring end-to-end including network overhead across more queries, while these numbers reflect Qdrant running warm with the index already loaded.

The takeaway: `hnsw_ef` is a useful knob when you want to squeeze more precision out of your HNSW index without rebuilding it. For this dataset, anything above 100 gets you to near-perfect recall.

---

### Task 3 — Scalar Quantization

#### Observed values

```
rescore=True  | precision@10: 1.0000 | avg time: 6.03 ms
rescore=False | precision@10: 1.0000 | avg time: 6.87 ms
```

#### Reflection

Both modes hit perfect precision (1.0000), which lines up with what the task description says about `text-embedding-ada-002` being a model that handles binary and scalar quantization well. Compressing the 1536-dim float32 vectors down to int8 loses very little information for this particular embedding space.

The `rescore=True` run is actually slightly faster here (6.03 ms vs 6.87 ms), which is a bit counterintuitive since rescoring does extra work — it retrieves a larger candidate pool via quantized vectors, then re-ranks using the full float32 vectors. The difference is small enough that it's likely measurement variance rather than a meaningful signal, but it also shows that the re-ranking overhead is negligible when the index is warm and local.

The more interesting story is precision: with `rescore=False`, the search runs entirely on compressed int8 vectors without any re-ranking step. For most embedding models this would cost you accuracy, but for ada-002 the int8 quantization is tight enough that you're still getting perfect recall. In a production scenario where you care about latency, `rescore=False` is tempting precisely because of that — you're trading a safety net you don't really need for a simpler, faster query path.

In summary: scalar quantization is essentially free for this model and dataset. You get 4x storage reduction with no precision loss, and both rescore modes perform well.
