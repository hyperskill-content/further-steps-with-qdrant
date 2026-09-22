### Observed values 

```
Average precision@10: 0.9990
Average ANN query time: 5.59 ms
Average exact k-NN query time: 19.63 ms
```

### Reflection
Speed: exact is ~3.5x slower (19.63 ms vs 5.59 ms). Why: Full scan vs graph traversal, O(N) vs ~O(log N),
gap grows with N.

But even though ANN is about 4 times slower in this example it matches 99.99% with the exact query.
This suggests that ANN is a good approximation for the exact k-NN query and worth it in this example. 
The one difference doesn't even mean it's not relevant to the search, it could very well be
that it matched another document that is almost as relevant to the search or even equally relevant.