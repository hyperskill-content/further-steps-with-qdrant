### Observed values 

```
Creating ground_truth dictionary...
Warming cache...
Running test dataset at hnsw_ef=10...
Running test dataset at hnsw_ef=20...
Running test dataset at hnsw_ef=50...
Running test dataset at hnsw_ef=100...
Running test dataset at hnsw_ef=200...
hnsw_ef evaluation results:
hnsw_ef: 10, avg_precision: 0.9430, avg_query_time_ms: 93.47
hnsw_ef: 20, avg_precision: 0.9780, avg_query_time_ms: 118.22
hnsw_ef: 50, avg_precision: 0.9980, avg_query_time_ms: 259.73
hnsw_ef: 100, avg_precision: 1.0000, avg_query_time_ms: 405.47
hnsw_ef: 200, avg_precision: 1.0000, avg_query_time_ms: 818.71
```

### Reflection

Which patterns are present? How can they be explained? Describe your findings in detail here.

Building on the last exercise, where I observed there seemed to be a point where precision was
"saturated", meaning the approximate search returned the same results as the exact search
(precision=1.0). However, this exercise makes it clear that above a certain "exploration factor"
(hnsw_ef value) with a given dataset all approximate queries will be identical to an exact search
while benefiting from speed advantages of the approximate search. In the case of the 'arxiv_papers'
dataset, it appears that hnsw_ef value is between 50-100, with further runs likely able to get
closer to the actual number.

I think there is likely an importance to finding this optimal/efficient hsnw_ef point for a given
dataset in order to optimize queries against it. I'm not sure yet what additional variables would
go into finding this value for a given dataset (indexing, query size, k value?)
