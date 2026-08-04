### Observed values 

```
TASK 3 - k-NN vs approximate search
TO DO

TASK 2 - Balance the search: Fine-tuning search parameters (e.g. hsnw-ef)
TO DO

TASK 1 - Quantization
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
After (forced) creating an index on the Qdrant client we get this better results.
1. The precision@10 is 0.9990 or just below 1.0 for both runs, indicating that the ANN index is able to return nearly correct results for all queries.
2. The query times for both ANN are 6 times les then for exact k-NN. 
3. This means it is beneficial to use the  ANN index with 6x faster and near correct accuracy for the test set.