"""Task 3 solution script."""

import json
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
K = 10
OVERSAMPLING = 2.0
# Use of __file__ based paths is not recommended for scripts that are not
# one-off.
QUERIES_FILE = Path(__file__).resolve().parents[2] / "dataset" / "queries_embeddings.json"


def load_test_dataset(queries_file):
    with open(queries_file, "r", encoding="utf-8") as file:
        return json.load(file)


def enable_scalar_quantization(client):
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


def query_knn_without_quantization(client, embedding, k):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=models.SearchParams(
            exact=True,
            quantization=models.QuantizationSearchParams(ignore=True),
        ),
    ).points


def query_quantized_ann(client, embedding, k, rescore):
    search_params = models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            oversampling=OVERSAMPLING,
        )
    )
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=search_params,
    ).points


def calculate_precision(approximate_ids, ground_truth_ids, k):
    return len(set(approximate_ids).intersection(ground_truth_ids)) / k


def calculate_absolute_error(approximate_ids, ground_truth_ids):
    return len(set(ground_truth_ids) - set(approximate_ids))


def build_ground_truth(client, test_dataset, k):
    ground_truth = {}
    for query_text, embedding in test_dataset.items():
        knn_result = query_knn_without_quantization(client, embedding, k)
        ground_truth[query_text] = {point.id for point in knn_result}
    return ground_truth


def evaluate_quantized_search(client, test_dataset, ground_truth, k, rescore):
    precisions = []
    elapsed_times_ms = []
    absolute_errors = []

    for query_text, embedding in test_dataset.items():
        start_time = time.perf_counter()
        ann_result = query_quantized_ann(client, embedding, k, rescore)
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000
        elapsed_times_ms.append(elapsed_time_ms)

        ann_ids = [point.id for point in ann_result]
        precisions.append(calculate_precision(ann_ids, ground_truth[query_text], k))
        absolute_errors.append(
            calculate_absolute_error(ann_ids, ground_truth[query_text])
        )

    return {
        "rescore": rescore,
        "oversampling": OVERSAMPLING,
        "avg_precision": sum(precisions) / len(precisions),
        "avg_query_time_ms": sum(elapsed_times_ms) / len(elapsed_times_ms),
        "total_absolute_error": sum(absolute_errors),
        "max_absolute_error": max(absolute_errors),
    }


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    test_dataset = load_test_dataset(QUERIES_FILE)

    enable_scalar_quantization(client)
    ground_truth = build_ground_truth(client, test_dataset, K)

    # Warmup run
    evaluate_quantized_search(client, test_dataset, ground_truth, K, rescore=True)
    rescore_true_results = evaluate_quantized_search(
        client, test_dataset, ground_truth, K, rescore=True
    )
    # warmup run
    evaluate_quantized_search(client, test_dataset, ground_truth, K, rescore=False)
    rescore_false_results = evaluate_quantized_search(
        client, test_dataset, ground_truth, K, rescore=False
    )

    results = [rescore_true_results, rescore_false_results]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
