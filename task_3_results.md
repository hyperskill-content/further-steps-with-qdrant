Getting the following results:

```sh
rescore=True: average precision=0.9980, query time=4.11 ms
rescore=False: average precision=0.8500, query time=4.01 ms
```

Using quantization has a precision of 99.8%.
Without quantization the precision drops to 85%, while being only 0.10 ms faster.
So in this case using quantization is the most efficient option.