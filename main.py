"""Task 2"""

import json
import time
from typing import Any, Dict, List

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"

QUERIES_FILE = "dataset/queries_embeddings.json"

K = 10
HNSW_EF_VALUES = [10, 20, 50, 100, 200]


def evaluate_hnsw_ef(
    client: QdrantClient, k: int, hnsw_ef_values: List[int], dataset: Dict[str, Any]
):
    """
    Evaluate the HNSW algorithm with different ef values on a given dataset.
    """
    ground_truth = {}
    for text, embedding in dataset.items():
        print(f"Ground truth / Processing query: {text}")

        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points

        point_ids = [point.id for point in knn_result]
        ground_truth[text] = point_ids

    results = []

    for hnsw_ef_value in hnsw_ef_values:
        ann_times = []
        ann_precisions = []

        for text, embedding in dataset.items():
            print(f"HNSW with ef={hnsw_ef_value} / Processing query: {text}")

            start_time_ann = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef_value),
            ).points
            ann_time = time.time() - start_time_ann
            ann_times.append(ann_time)

            ann_ids = {point.id for point in ann_result}
            ground_truth_ids = ground_truth[text]
            precision = len(ann_ids.intersection(ground_truth_ids)) / k
            ann_precisions.append(precision)

        avg_precision = sum(ann_precisions) / len(dataset)
        avg_ann_time = sum(ann_times) / len(dataset)
        result = {
            "hnsw_ef": hnsw_ef_value,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_ann_time * 1000,
        }
        results.append(result)

    return results


def result_formatting(results: List[Dict[str, Any]]):
    """Format and print the evaluation results.

    Args:
        results (List[Dict[str, Any]]): List of evaluation results.
    """
    print("\nResults:")
    for result in results:
        hnsw_ef = result["hnsw_ef"]
        avg_precision = result["avg_precision"]
        avg_ann_time = result["avg_query_time_ms"]
        print(f"HNSW ef={hnsw_ef} {avg_precision:.4f} {avg_ann_time:.2f} ms")


def main():
    """Main function to run the evaluation."""

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    evaluation_results = evaluate_hnsw_ef(client, K, HNSW_EF_VALUES, test_dataset)

    result_formatting(evaluation_results)


if __name__ == "__main__":
    main()
