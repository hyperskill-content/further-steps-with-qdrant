### Observed values 

```
uv run evaluate_precision.py -k 10 -ef 128
Average precision@10: 1.0000
Average ANN query time: 16.43 ms
Average exact k-NN query time: 153.02 ms

uv run evaluate_precision.py -k 10 -ef 32
Average precision@10: 0.9920
Average ANN query time: 12.71 ms
Average exact k-NN query time: 152.58 ms

uv run evaluate_precision.py -k 100 -ef 128
Average precision@100: 0.9919
Average ANN query time: 25.21 ms
Average exact k-NN query time: 167.22 ms

uv run evaluate_precision.py -k 100 -ef 32
Average precision@100: 0.9771
Average ANN query time: 25.44 ms
Average exact k-NN query time: 175.22 ms
```

### Reflection

The higher the k the slower the ANN query time. The lower the ef the faster the query but the lower the precision.