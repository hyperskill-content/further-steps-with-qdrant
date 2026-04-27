### Observed values

```
[
  {
    "rescore": true,
    "oversampling": 2.0,
    "avg_precision": 0.9980000000000001,
    "avg_query_time_ms": 15.52267599850893,
    "total_absolute_error": 2,
    "max_absolute_error": 1
  },
  {
    "rescore": false,
    "oversampling": 2.0,
    "avg_precision": 0.8449999999999999,
    "avg_query_time_ms": 15.433264008024707,
    "total_absolute_error": 155,
    "max_absolute_error": 5
  }
]
```

### Reflection

#### A technical summary of the results
With rescoring the results had the same accuracy as the default approximate 
nearest neighbor (ANN) search I ran in [task one](../task_1/solution.md). 
Both were 99.8% accurate. The quantized query with rescoring also ran 
about 3.87 ms or about 20% faster.  This is a significant improvement in 
speed with no measurable loss of accuracy.

The query without rescoring was significantly less accurate at only 
about 84.45%.  In the worst case it failed to find half the correct results.
It was also only about 0.08 ms faster than the quantized search with 
rescoring.  This is a significant loss of accuracy with no comparable 
increase in speed.

#### An accessible explanation of the results

Rescoring works by taking the approximate results found by the 
quantized search and refining them by computing the distance using the full 
vectors, the way it would for a default ANN search.  By itself, this only 
makes approximate results more precise.  To change what is in the final set 
of results, it must work together with oversampling.  Oversampling in this 
context means returning more candidate points than the final results 
require.  Oversampling by a factor of 2 means to consider twice as many 
points as will be in the final results.  This way, when rescoring reorders 
the candidates, it can change what is included in the final results.

I don't have as much time to go into additional detail as I did for the 
first two tasks.   If I did, I would elaborate on the similarities between 
oversampling in this context and oversampling in computer graphics.  I 
might also go into more detail on how quantization works and the importance 
of the quantile parameter excluding outliers. 

#### Further explorations
Since this task had less guidance on the format of the output, I left the 
additional details inline.  I continued running warmup passes.  I used time.
perf_counter() as I mentioned in task 1, because there wasn't an example of 
using time.time() in the code provided for this task, so I saw less reason 
not to do so.