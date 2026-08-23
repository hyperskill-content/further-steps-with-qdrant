

### Observed values 
Task 1
```
Average precision@10: 1.0000
Average ANN query time: 13.89 ms
Average exact k-NN query time: 98.21 ms
```

### Reflection

100% Precision@10 meaning the results returned by ANN is exactly same as KNN with substantial performance gains.
This demonstrates HNSW graph-based approximate search achieves an ideal trade-off: it delivers a ~7x reduction in query latency with zero loss in precision compared to brute-force exact search.

Task 2 

### Observed values
...
[{'hnsw_ef': 10, 'avg_precision': 0.96, 'avg_query_time_ms': 25.065207481384277}, {'hnsw_ef': 20, 'avg_precision': 0.9890000000000001, 'avg_query_time_ms': 25.52582025527954}, {'hnsw_ef': 50, 'avg_precision': 0.9990000000000001, 'avg_query_time_ms': 29.28570032119751}, {'hnsw_ef': 100, 'avg_precision': 1.0, 'avg_query_time_ms': 22.170231342315674}, {'hnsw_ef': 200, 'avg_precision': 1.0, 'avg_query_time_ms': 30.003669261932373}]
....

### Reflection
As the hnsw_ef increases from 10 to 100, average precision steadily climbs from 0.9600 to a perfect 1.0000. A smaller ef tends to search just the nearset neighbors so the precision is less compared to higher ef.
ef at 100 is optimum
Average query times remain tightly clustered in the 22 ms – 30 ms range across all test configurations.