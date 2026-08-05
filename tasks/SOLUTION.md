### Observed values 

```
TASK 3 - Quantization
TO DO

TASK 2 - Balance the search: Fine-tuning search parameters (e.g. hsnw-ef)
hnsw_ef= 10 | avg_precision=0.9220 | avg_query_time=11.93 ms
hnsw_ef= 20 | avg_precision=0.9770 | avg_query_time=11.91 ms
hnsw_ef= 50 | avg_precision=0.9980 | avg_query_time=13.71 ms
hnsw_ef=100 | avg_precision=0.9990 | avg_query_time=17.28 ms
hnsw_ef=200 | avg_precision=0.9990 | avg_query_time=16.96 ms

hnsw_ef= 10 | avg_precision=0.9220 | avg_query_time=14.36 ms
hnsw_ef= 20 | avg_precision=0.9770 | avg_query_time=13.42 ms
hnsw_ef= 50 | avg_precision=0.9980 | avg_query_time=14.91 ms
hnsw_ef=100 | avg_precision=0.9990 | avg_query_time=14.63 ms
hnsw_ef=200 | avg_precision=0.9990 | avg_query_time=16.29 ms

TASK 1 - k-NN vs approximate search
Average precision@10: 0.9990
Average ANN query time: 16.39 ms
Average exact k-NN query time: 109.97 ms

Average precision@10: 0.9990
Average ANN query time: 15.61 ms
Average exact k-NN query time: 97.63 ms

Average precision@10: 0.9990
Average ANN query time: 15.47 ms
Average exact k-NN query time: 98.38 ms
```


### Reflection

TASK 3 - Quantization


TASK 2 - Balance the search: Fine-tuning search parameters (e.g. hsnw-ef)
1. We see the average precision fundamentally improves to >0.99 from hnsw_ef = 50
2. While the average execution speed remains more or less the same for each value of hnsw_ef
3. So we propose to use a value for hnsw_ef of 50. 

TASK 1 - k-NN vs approximate search
After (forced) creating an index on the Qdrant client we get this better results.
1. The precision@10 is 0.9990 or just below 1.0 for both runs, indicating that the ANN index is able to return nearly correct results for all queries.
2. The query times for both ANN are 6 times les then for exact k-NN. 
3. This means it is beneficial to use the  ANN index with 6x faster and near correct accuracy for the test set.