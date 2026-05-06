def result_formatting(
    k: int,
    avg_precision: float,
    avg_ann_time: float,
    avg_knn_time: float,
) -> None:
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")
