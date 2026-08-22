### Observed values 

```
Average precision@10: 1.0000
Average ANN query time: 13.89 ms
Average exact k-NN query time: 98.21 ms
```

### Reflection

100% Precision@10 meaning the results returned by ANN is exactly same as KNN with substantial performance gains.
This demonstrates HNSW graph-based approximate search achieves an ideal trade-off: it delivers a ~7x reduction in query latency with zero loss in precision compared to brute-force exact search.