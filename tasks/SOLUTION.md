### Observed values 

```
Average precision@10: 0.9990
Average ANN query time: 39.50 ms
Average exact k-NN query time: 97.10 ms
```

### Reflection

The precision@10 of 0.9990 is basically perfect — across all 100 queries, the ANN search almost always returns the exact same 10 nearest neighbors as the brute-force exact search. That's a really strong result and means the HNSW index is doing its job well for this dataset.

Speed-wise the ANN search comes in at around 39.5 ms versus 97.1 ms for the exact k-NN, so roughly 2.5x faster. That gap will only grow as the collection gets larger — exact search scales linearly with the number of vectors while HNSW is logarithmic, so the trade-off gets increasingly worthwhile at scale.

The high precision makes sense for a dataset like this. The arXiv ML papers tend to cluster naturally around topics (computer vision, NLP, reinforcement learning, etc.), and papers within a cluster are very close to each other in embedding space. That kind of structure is exactly what HNSW is designed to exploit — it builds a graph where nearby vectors are well-connected, so the approximate search rarely misses a true neighbor.

In short: for this collection, HNSW gives us essentially the same quality as an exhaustive search at a fraction of the cost. There's very little reason to use exact search in production here.
