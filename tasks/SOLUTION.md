### Observed values 

```
 hnsw_ef | avg precision | avg time (ms)
------------------------------------------
      10 |        0.9690 |          2.79
      20 |        0.9930 |          2.82
      50 |        0.9980 |          3.14
     100 |        0.9990 |          3.94
     200 |        1.0000 |          5.10
```

### Reflection

Even though the precision drops quite a lot with lower hnsw_ef values, the query time doesn't decrease significantly.
Whether it's overhead of networking or not is not determined in this task
Whether the produced search results are noticeable worse is another topic that would be interesting to explore.
But based on these numbers, I would probably just go with the default value of 100.

Another interesting finding, in task 1 the timing was around ~5ms. Thus the cache warmup lowered the numbers.