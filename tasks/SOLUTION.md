### Observed values 

```
uv run python evaluate_precision.py -k 10 --enable-quantization --test-quantization
quantized_search rescore=True: avg_precision=1.0000, avg_query_time_ms=4.89, avg_exact_query_time_ms=68.36
quantized_search rescore=False: avg_precision=0.8330, avg_query_time_ms=4.31, avg_exact_query_time_ms=70.86
```

### Reflection

rescore=False (Precision is lower):
When rescoring is turned off, Qdrant ranks vectors using INT8 approximations. As a result, vectors get rounded into slightly coarse integer scores, causing re-ranking errors among the top results and dropping precision.
rescore=True (Precision: 100.0%):
With rescoring enabled, Qdrant executes a two-stage search:Candidate Gathering: Using fast INT8 vectors, Qdrant fetches a candidate pool of 2 times the k. Then it retrieves the original, full-precision 32-bit float vectors only for those 20 candidates and re-calculates exact similarity scores to determine the final top 10.Because all 10 true nearest neighbors were successfully captured in the top 20 candidates during Stage 1, the Stage 2 rescore restored precision back to a perfect 100%.
