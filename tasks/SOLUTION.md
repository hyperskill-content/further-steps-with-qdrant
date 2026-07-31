### Observed values 

```
stdout of the solution script for the task
Average precision@10: 1.0000
Average ANN query time: 62.86 ms
Average exact k-NN query time: 62.89 ms

Average precision@10: 1.0000
Average ANN query time: 66.96 ms
Average exact k-NN query time: 65.99 ms
```


### Reflection

1. The precision@10 is 1.0000 for both runs, indicating that the ANN index is able to return the correct results for all queries.
2. The query times for both ANN and exact k-NN are very close. This means the ANN index is able to return results that are very similar to the exact k-NN results. 
   * The ANN index has a similar accuracy as the k-NN index for a similar speed, for the given dataset.
3. Possible we have these good results for the ANN index because the test data set with only 100 embeddings is small.
   * This means we can use the k-nn index for small datasets, without losing speed 
   * Or we can use the ANN index for small datasets, without losing accuracy.