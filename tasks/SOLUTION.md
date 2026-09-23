### Observed values 

```
 rescore | avg precision | avg time (ms)
------------------------------------------
    True |        0.9970 |          3.01
   False |        0.8490 |          2.88
```

### Reflection

The quantization significantly reduces accuracy but rescoring in this example brings it almost on par with the exact
k-NN search. The speed difference is negligible so rescoring is a very good trade-off.
Nevertheless, unless the dataset is very large I would not quantize the dataset unless hardware limitations are a concern.