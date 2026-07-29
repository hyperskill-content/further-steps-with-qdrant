### Observed values 

```
----------------------------------------------------------------------------------------------------
Rescore: True
Average precision@10: 0.9990
Average ANN query time: 5.17 ms
Average exact k-NN query time: 17.13 ms
----------------------------------------------------------------------------------------------------
Rescore: False
Average precision@10: 0.8320
Average ANN query time: 5.34 ms
Average exact k-NN query time: 17.33 ms
```

2nd run: running Rescore False first
```
----------------------------------------------------------------------------------------------------
Rescore: False
Average precision@10: 0.8320
Average ANN query time: 5.14 ms
Average exact k-NN query time: 17.29 ms
----------------------------------------------------------------------------------------------------
Rescore: True
Average precision@10: 0.9990
Average ANN query time: 5.22 ms
Average exact k-NN query time: 17.22 ms

```

### Reflection

With our dataset we get a precision gain with rescoring enabled (rescore=True)
- rescore=True: Average precision@10 is 0.9990 (99.9%).
- rescore=False: Average precision@10 drops to 0.8320 (83.2%).
And a negligible query time increases with rescoring enabled (rescore=True)
- rescore=True: Average ANN query time is 5.17 ms – 5.22 ms.
- rescore=False: Average ANN query time is 5.14 ms - 5.34 ms.
ANN Search vs. Exact k-NN Search Speedup
- ANN Query Time (Quantized): 5.14 ms - 5.34 ms
- Exact k-NN Query Time (Unquantized, ignore=True): 17.13 ms - 17.33 ms
Using INT8 scalar quantization speeds up search 3x compared to full-precision exact k-NN searches.