import json
import os
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
TARGET_FILE = os.path.join(os.path.dirname(__file__), "dataset/queries_embeddings.json")
K = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def enable_scalar_quantization():
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=models.OptimizersConfigDiff(),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=False,
            ),
        ),
    )


def precision_at_k(ann_points, knn_points, k):
    ann_ids = [point.id for point in ann_points[:k]]
    knn_ids = [point.id for point in knn_points[:k]]

    if not knn_ids:
        return 0.0

    overlap = len(set(ann_ids) & set(knn_ids))
    return overlap / k


def run_benchmark(rescore: bool, oversampling: float = 2.0):
    precisions = []
    ann_times = []
    knn_times = []

    with open(TARGET_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    for query_text, embedding in test_dataset.items():
        # ANN search on quantized vectors
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=rescore,
                    oversampling=oversampling,
                )
            ),
        ).points
        ann_time = time.time() - start_time_ann

        # Exact k-NN on original vectors (ignore quantization)
        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(
                exact=True, quantization=models.QuantizationSearchParams(ignore=True)
            ),
        ).points
        knn_time = time.time() - start_time_knn

        precision = precision_at_k(ann_result, knn_result, K)

        precisions.append(precision)
        ann_times.append(ann_time)
        knn_times.append(knn_time)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_ann_time = sum(ann_times) / len(ann_times) if ann_times else 0.0
    avg_knn_time = sum(knn_times) / len(knn_times) if knn_times else 0.0

    return avg_precision, avg_ann_time, avg_knn_time


def print_results(title, k, avg_precision, avg_ann_time, avg_knn_time):
    print(title)
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")
    print()


if __name__ == "__main__":
    # Step 1: enable scalar quantization
    enable_scalar_quantization()

    # Step 2: benchmark with rescoring
    avg_precision_true, avg_ann_true, avg_knn_true = run_benchmark(
        rescore=True, oversampling=2.0
    )
    print_results(
        "Results with scalar quantization, rescore=True",
        K,
        avg_precision_true,
        avg_ann_true,
        avg_knn_true,
    )

    # Step 3: benchmark without rescoring
    avg_precision_false, avg_ann_false, avg_knn_false = run_benchmark(
        rescore=False, oversampling=2.0
    )
    print_results(
        "Results with scalar quantization, rescore=False",
        K,
        avg_precision_false,
        avg_ann_false,
        avg_knn_false,
    )
