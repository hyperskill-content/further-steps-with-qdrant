Getting the following results:

```bash
ef=10: average precision=0.9640, query time=2.98 ms
ef=20: average precision=0.9900, query time=3.45 ms
ef=50: average precision=0.9990, query time=3.97 ms
ef=100: average precision=0.9990, query time=4.68 ms
ef=200: average precision=1.0000, query time=5.70 ms
```

Increasing hnsw_ef improves precision up to 100%, but increases the query time.
Setting ef to 200 already achieves 99.9% precision, while being 1.65 times faster.
So in this case, ef of 20 is the most efficient setting for reaching nearly exact match result.