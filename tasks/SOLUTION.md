### Observed values 

```
Creating ground_truth dictionary...
Warming cache...
Running test dataset at hnsw_ef=10...
Running test dataset at hnsw_ef=20...
Running test dataset at hnsw_ef=50...
Running test dataset at hnsw_ef=100...
Running test dataset at hnsw_ef=200...
Running test dataset at hnsw_ef=10...
Running test dataset at hnsw_ef=20...
Running test dataset at hnsw_ef=50...
Running test dataset at hnsw_ef=100...
Running test dataset at hnsw_ef=200...

rescore=True
hnsw_ef evaluation results:
hnsw_ef: 10, avg_precision: 0.9870, avg_query_time_ms: 20.77
hnsw_ef: 20, avg_precision: 0.9870, avg_query_time_ms: 9.31
hnsw_ef: 50, avg_precision: 0.9970, avg_query_time_ms: 8.62
hnsw_ef: 100, avg_precision: 1.0000, avg_query_time_ms: 8.04
hnsw_ef: 200, avg_precision: 1.0000, avg_query_time_ms: 10.28

rescore=False
hnsw_ef evaluation results:
hnsw_ef: 10, avg_precision: 0.8270, avg_query_time_ms: 7.52
hnsw_ef: 20, avg_precision: 0.8270, avg_query_time_ms: 9.84
hnsw_ef: 50, avg_precision: 0.8270, avg_query_time_ms: 10.33
hnsw_ef: 100, avg_precision: 0.8270, avg_query_time_ms: 8.40
hnsw_ef: 200, avg_precision: 0.8270, avg_query_time_ms: 9.75
```

### Reflection

Interestingly, the average precision was identical when rescore=False. The rescore
result set avg_precision values are very close (if not indentical to) to precision
values seen in task 2. However, the avg_query_time is much less (query times ranged
from 93-818 ms in task 2).

In this dataset, rescoring does not appear to levy as significant query time penalty,
as the search area from the actual vector set is much reduced.

At least for this dataset, I'm not sure why quantization (with rescoring) would not
be utilized, unless exact precision is absolutely necessary.
