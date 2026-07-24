### Observed values 

```
## ANN - hnsw_ef: Default (run 1)
Average precision@10: 1.0000
Average ANN query time: 1172.21 ms
Average exact k-NN query time: 4272.17 ms

## ANN - hnsw_ef=1 (run 2)
Average precision@10: 0.9430
Average ANN query time: 268.09 ms
Average exact k-NN query time: 4537.22 ms

```

### Reflection

Which patterns are present? How can they be explained? Describe your findings in detail here.

I ended up running this precision and timing test twice as the initial run resulted in a precision
of 1.0. I don't believe there are any problems with methodology or the code. After some research,
it appears that the default hnsw_ef factor may be high enough that this small query set was able
to achieve 1.0 precision against the indexed arxiv_papers set.

To validate this, I ran the code again with hnsw_ef=1, and did notice a difference in precision.

As expected, the ANN timings returned more quickly than the KNN queries (in approximately 1/4 of
of the time for the default ef run, and several times faster for the ef=1 run). It is interesting
to note that with the default ef achieving a precision of 1.0, you still return results much more
quickly.

This demonstrated that there is a tradeoff between speed and precision, but there are definitely
available efficiencies depending on ef values, dataset size, and dataset indexing.