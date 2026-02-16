Getting the following results:

```bash
Average precision@10: 0.9990
Average ANN query time: 5.91 ms
Average exact k-NN query time: 17.82 ms
```

This means that 99.9% of the time ANN returns the same result as k-NN, with k-NN being 3 times slower.

So in this case, ANN is more efficient while returning nearly the same results, so it should be preferred for most
cases.