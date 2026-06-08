### Observed values

```
stdout of the solution script for the task
```

Average precision@10: 1.0000
Average ANN query time: 132.99 ms
Average exact k-NN query time: 130.13 ms

### Reflection

1. Precision is 1, which is surprising for me. (Not sure if my result is correct)
   For an increased processing time of 2,86 ms (for 10 results) (or 2,2 %) a similar accuracy seems very reasonable.
   Given the limited decrease in accuracy and limited increase in processing time, I would definitely go here for an ANN query.
2. Optionally can be extended the test sample to validate further the findings.
3. Alternative investigations:
   * Reduce vector dimension to validate if you have similar accuracy or precision factor for a possible improved processing itme.



Which patterns are present?

1. Reduction of search space: through algorithm used
2. Limited test sample

*Note*> Unclear if this is correct understanding of 'patterns'

How can they be explained?
