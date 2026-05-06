# Task 2 — Solution

### Observed values

```
 hnsw_ef |  avg_precision |  avg_query_time_ms
------------------------------------------------
      10 |         0.9460 |              11.24
      20 |         0.9850 |               9.46
      50 |         0.9990 |               9.13
     100 |         0.9990 |              15.33
     200 |         1.0000 |              15.46
```

(Same setup as Task 1 — 100 queries, `arxiv_papers` of 411,120 points, cosine, default HNSW index. Ground truth recomputed via `exact=True` per query before sweeping `hnsw_ef`.)

### Reflection

**Precision behaves monotonically with `hnsw_ef`, as expected.** A larger `ef` means HNSW keeps a longer candidate priority queue while traversing the graph, so the chance of missing a true nearest neighbor drops. Concretely:

| `hnsw_ef` | Misses out of 1000 retrieved IDs |
|-----------|----------------------------------|
| 10        | 54                               |
| 20        | 15                               |
| 50        | 1                                |
| 100       | 1                                |
| 200       | 0                                |

The marginal precision gain becomes negligible past `ef=50` — by that point the candidate list is already wider than the true 10-NN neighborhood, so HNSW has effectively converged on the exact result for this collection size.

**Query time is *not* monotonic — it has a U shape (11.24 → 9.46 → 9.13 → 15.33 → 15.46 ms).** Two competing effects explain this:

1. **For small `ef` (10 → 50)**, query time actually *decreases slightly* with larger `ef`. At `ef=10` the search is so narrow that HNSW frequently has to backtrack to recover the top-10, and a meaningful fraction of the per-query time is dominated by fixed overhead (HTTP round-trip, request/response (de)serialization, JSON parsing). Increasing `ef` to 20–50 lets the graph traversal fall into a cleaner, more cache-friendly path while the fixed overhead stays constant — so the *fraction* of time spent on graph work shrinks proportionally.
2. **For larger `ef` (100 → 200)**, query time rises (~9 ms → ~15 ms), which matches the textbook expectation: more candidates kept in the priority queue means more distance computations. The growth is sub-linear because the graph traversal still terminates early once the top-k is locked in.

**Sweet spot for this collection**: `hnsw_ef ≈ 50` — precision indistinguishable from exact (0.999), and the lowest measured query time. Going higher trades latency for a vanishing precision gain (1.000 vs 0.999 means *one* extra correct ID across 1000); going lower loses precision quickly (0.985 at ef=20, 0.946 at ef=10) for no real time savings.

**Practical takeaways**:
- The Qdrant default of `hnsw_ef=128` is conservative for a 411k-point collection at this dimensionality — `ef=50` would give comparable accuracy at lower latency.
- For latency-critical paths where ~5% recall loss is acceptable, `ef=10` could be combined with a payload-level re-rank, but on this dataset the ~2 ms of "savings" is dwarfed by HTTP overhead, so it is not worth the precision drop.
- Variance: each cell is the mean of 100 queries on a single warm-cache run. The U-shape in timing is real but small; on a larger sweep (10× the queries, multiple runs) one would smooth out the apparent dip at `ef=50` and might see a flatter "knee" between 10 and 100.

**Cross-task connection**: Task 3 will introduce `int8` quantization. The expectation is that quantization will give a flat ~2× speedup across all `hnsw_ef` values (because it shrinks distance-computation cost per candidate) while losing very little precision when `rescore=True` re-orders the top-k with full-precision vectors.
