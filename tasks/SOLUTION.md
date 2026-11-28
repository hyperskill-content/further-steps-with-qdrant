### Observed values 

```
stdout of the solution script for the task
```

### Reflection

Which patterns are present? How can they be explained? Describe your findings in detail here.

A good reflection:

* **Describes the setup.** Collection size, status and indexed vector count, whether quantization was configured, how warm-up and repeated passes were done, and anything you changed from the defaults.
* **Explains each pattern, not just restates it.** For example: why precision saturates, or why a speed-up is smaller than the algorithmic difference would suggest.
* **Gives counts alongside ratios.** A precision@10 of 0.998 over 100 queries means 2 missed neighbours out of 1,000.
* **Stays within what was measured.** "`hnsw_ef=50` was the lowest *tested* value that reached 1.0"; results hold for `k=10`, this collection and these 100 queries.
* **Treats small or surprising differences with care.** A gap of a fraction of a millisecond, or a result in the opposite direction to what you'd expect, is more likely noise or run order than a real effect. Say so, or rerun to check.
* **Compares like with like.** Only compare timings across tasks if they were measured the same way.
* **Ends with a takeaway.** Which setting you would choose, and why.
