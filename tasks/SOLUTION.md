### Observed values 

```
hnsw_ef=10: avg_precision=0.9600, avg_query_time=12.41 ms
hnsw_ef=20: avg_precision=0.9950, avg_query_time=9.75 ms
hnsw_ef=50: avg_precision=0.9990, avg_query_time=10.67 ms
hnsw_ef=100: avg_precision=1.0000, avg_query_time=11.52 ms
hnsw_ef=200: avg_precision=1.0000, avg_query_time=13.19 ms
```

### Reflection

Precision increases steadily with hnsw_ef — from 0.96 at ef=10 to perfect 1.0 at ef=100. Beyond 100, there's no further accuracy gain, only increased query time. Interestingly, the speed differences are relatively small across all values (9-13ms), suggesting that for this dataset the bottleneck isn't candidate exploration but other overhead. The sweet spot appears to be around hnsw_ef=50-100, where you get near-perfect precision without meaningful speed cost. At ef=10 you lose 4% accuracy for negligible speed gain, making it a poor trade-off for this collection size and index configuration.
