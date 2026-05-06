def result_formatting(
    k: int,
    avg_precision: float,
    avg_ann_time: float,
    avg_knn_time: float,
) -> None:
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


def format_hnsw_ef_results(results: list[dict]) -> None:
    print(f"{'hnsw_ef':>8} | {'avg_precision':>14} | {'avg_query_time_ms':>18}")
    print("-" * 48)
    for row in results:
        print(
            f"{row['hnsw_ef']:>8} | "
            f"{row['avg_precision']:>14.4f} | "
            f"{row['avg_query_time_ms']:>18.2f}"
        )


def format_quantization_block(label: str, k: int, result: dict) -> None:
    print(f"--- {label} ---")
    print(f"Average precision@{k}: {result['avg_precision']:.4f}")
    print(f"Average ANN query time: {result['avg_ann_time'] * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {result['avg_knn_time'] * 1000:.2f} ms")
