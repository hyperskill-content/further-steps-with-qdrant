"""Task 2 solution script."""

import json
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
K = 10
HNSW_EF_VALUES = [10, 20, 50, 100, 200]
# Switched to global client to match provided function signatures exactly.
CLIENT = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
# Use of __file__ is for one-off scripts only.
QUERIES_FILE = Path(__file__).resolve().parents[2] / "dataset" / "queries_embeddings.json"


def load_test_dataset(queries_file):
    with open(queries_file, "r", encoding="utf-8") as file:
        return json.load(file)


def query_knn(client, embedding, k):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=models.SearchParams(exact=True),
    ).points


def query_ann(client, embedding, k, hnsw_ef):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=models.SearchParams(hnsw_ef=hnsw_ef),
    ).points


def calculate_precision(approximate_ids, ground_truth_ids, k):
    return len(set(approximate_ids).intersection(ground_truth_ids)) / k


def calculate_absolute_error(approximate_ids, ground_truth_ids):
    return len(set(ground_truth_ids) - set(approximate_ids))


def get_default_hnsw_ef(client):
    collection_info = client.get_collection(COLLECTION_NAME)
    return collection_info.config.hnsw_config.ef_construct


def print_additional_results(results):
    print()
    print("### Absolute Error Summary")
    for result in results:
        print(f"hnsw_ef={result['hnsw_ef']}")
        print(f"Total absolute error: {result['total_absolute_error']}")
        print(f"Max absolute error: {result['max_absolute_error']}")
        print()

    print("### Collection Defaults")
    print(f"Default hnsw_ef: {get_default_hnsw_ef(CLIENT)}")


def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):
    ground_truth = {}

    for query_text, embedding in test_dataset.items():
        knn_result = query_knn(CLIENT, embedding, k)
        ground_truth[query_text] = {point.id for point in knn_result}

    results = []
    for hnsw_ef in hnsw_ef_values:
        precisions = []
        elapsed_times_ms = []
        absolute_errors = []

        for query_text, embedding in test_dataset.items():
            start_time = time.time()
            ann_result = query_ann(CLIENT, embedding, k, hnsw_ef)
            elapsed_time_ms = (time.time() - start_time) * 1000
            elapsed_times_ms.append(elapsed_time_ms)

            ann_ids = [point.id for point in ann_result]
            precision = calculate_precision(ann_ids, ground_truth[query_text], k)
            precisions.append(precision)
            absolute_errors.append(
                calculate_absolute_error(ann_ids, ground_truth[query_text])
            )

        results.append(
            {
                "hnsw_ef": hnsw_ef,
                "avg_precision": sum(precisions) / len(precisions),
                "avg_query_time_ms": sum(elapsed_times_ms) / len(elapsed_times_ms),
                "total_absolute_error": sum(absolute_errors),
                "max_absolute_error": max(absolute_errors),
            }
        )

    return results


def main():
    test_dataset = load_test_dataset(QUERIES_FILE)
    # warmup run.
    evaluate_hnsw_ef(K, HNSW_EF_VALUES, test_dataset)
    results = evaluate_hnsw_ef(K, HNSW_EF_VALUES, test_dataset)
    print(json.dumps(results, indent=2))
    print_additional_results(results)


if __name__ == "__main__":
    main()
