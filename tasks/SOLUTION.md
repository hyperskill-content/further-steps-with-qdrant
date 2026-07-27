### Observed values 

```
v run evaluate_precision.py --test-hnsw-ef
hnsw_ef=10: avg_precision=0.9330, avg_query_time_ms=6.82
hnsw_ef=20: avg_precision=0.9820, avg_query_time_ms=7.35
hnsw_ef=50: avg_precision=0.9970, avg_query_time_ms=7.65
hnsw_ef=100: avg_precision=1.0000, avg_query_time_ms=9.21
hnsw_ef=200: avg_precision=1.0000, avg_query_time_ms=15.07
```

### Reflection

The lower the hnsw_ef the faster the query but the lower the precision.