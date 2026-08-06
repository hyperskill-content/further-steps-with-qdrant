### Observed values 

```
TASK 3 - Quantization 
* rescore = True (for calculation ANN search and warm-up)
hnsw_ef= 10 | avg_precision=0.9710 | avg_query_time=15.73 ms
hnsw_ef= 20 | avg_precision=0.9710 | avg_query_time=15.08 ms
hnsw_ef= 50 | avg_precision=0.9930 | avg_query_time=17.34 ms
hnsw_ef=100 | avg_precision=0.9970 | avg_query_time=17.22 ms
hnsw_ef=200 | avg_precision=0.9980 | avg_query_time=17.89 ms

hnsw_ef= 10 | avg_precision=0.9710 | avg_query_time=17.96 ms
hnsw_ef= 20 | avg_precision=0.9710 | avg_query_time=15.94 ms
hnsw_ef= 50 | avg_precision=0.9930 | avg_query_time=13.60 ms
hnsw_ef=100 | avg_precision=0.9970 | avg_query_time=13.40 ms
hnsw_ef=200 | avg_precision=0.9980 | avg_query_time=19.40 ms

* rescore = False (for calculation ANN search and warm-up)
hnsw_ef= 10 | avg_precision=0.8460 | avg_query_time=19.11 ms
hnsw_ef= 20 | avg_precision=0.8460 | avg_query_time=17.10 ms
hnsw_ef= 50 | avg_precision=0.8600 | avg_query_time=17.93 ms
hnsw_ef=100 | avg_precision=0.8610 | avg_query_time=15.12 ms
hnsw_ef=200 | avg_precision=0.8610 | avg_query_time=16.21 ms

hnsw_ef= 10 | avg_precision=0.8460 | avg_query_time=16.46 ms
hnsw_ef= 20 | avg_precision=0.8460 | avg_query_time=14.15 ms
hnsw_ef= 50 | avg_precision=0.8600 | avg_query_time=15.04 ms
hnsw_ef=100 | avg_precision=0.8610 | avg_query_time=15.03 ms
hnsw_ef=200 | avg_precision=0.8610 | avg_query_time=15.74 ms

*** Wrong - Only creating different hnsw-values (rescore=True) - d.d. 5-08
hnsw_ef= 10 | avg_precision=0.9220 | avg_query_time=11.98 ms
hnsw_ef= 20 | avg_precision=0.9770 | avg_query_time=11.09 ms
hnsw_ef= 50 | avg_precision=0.9980 | avg_query_time=10.87 ms
hnsw_ef=100 | avg_precision=0.9990 | avg_query_time=11.99 ms
hnsw_ef=200 | avg_precision=0.9990 | avg_query_time=14.61 ms

hnsw_ef= 10 | avg_precision=0.9220 | avg_query_time=12.92 ms
hnsw_ef= 20 | avg_precision=0.9770 | avg_query_time=16.60 ms
hnsw_ef= 50 | avg_precision=0.9980 | avg_query_time=9.80 ms
hnsw_ef=100 | avg_precision=0.9990 | avg_query_time=10.95 ms
hnsw_ef=200 | avg_precision=0.9990 | avg_query_time=14.35 ms

hnsw_ef= 10 | avg_precision=0.9220 | avg_query_time=14.01 ms
hnsw_ef= 20 | avg_precision=0.9770 | avg_query_time=15.69 ms
hnsw_ef= 50 | avg_precision=0.9980 | avg_query_time=12.15 ms
hnsw_ef=100 | avg_precision=0.9990 | avg_query_time=10.56 ms
hnsw_ef=200 | avg_precision=0.9990 | avg_query_time=13.94 ms

******
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

******
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
We varied now the rescore parameter (True and False) for ANN search and warmup, while we keep the different hnsw-ef values.
1. With rescore=false :
   - the speed remains similar or decreases only with 10%;
   - but the average precision decreases with a little bit less then 20% to 0.84 - 0.86, depending on hnsw-ef value.
2. Given the fundamental depreciation of the precision it is not an option to NOT use the rescore values (rescore=False) for the ANN search.

TASK 2 - Balance the search: Fine-tuning search parameters (e.g. hsnw-ef)
1. We see the average precision fundamentally improves to >0.99 from hnsw_ef = 50
2. While the average execution speed remains more or less the same for each value of hnsw_ef
3. So we propose to use a value for hnsw_ef of 50. 

TASK 1 - k-NN vs approximate search 
After (forced) creating an index on the Qdrant client we get this better results.
1. The precision@10 is 0.9990 or just below 1.0 for both runs, indicating that the ANN index is able to return nearly correct results for all queries.
2. The query times for both ANN are 6 times les then for exact k-NN. 
3. This means it is beneficial to use the  ANN index with 6x faster and near correct accuracy for the test set.