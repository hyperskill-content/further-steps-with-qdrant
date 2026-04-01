### Observed values 

```
Average precision@10: 1.0000
Average ANN query time: 135.87 ms
Average exact k-NN query time: 361.63 ms
```

### Reflection

The ANN search achieves perfect precision@10 of 1.0, meaning every approximate result matches the exact k-NN result for all 100 test queries. Despite this, ANN is roughly 2.5x faster than exact search (136ms vs 362ms). This demonstrates that the default HNSW index configuration in Qdrant is highly effective for this dataset — there is no accuracy trade-off at all, only a speed gain. For larger datasets or different configurations, we would expect precision to drop slightly, making this kind of benchmarking valuable for tuning the balance between speed and accuracy
