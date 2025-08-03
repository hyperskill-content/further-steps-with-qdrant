"""Task 3"""

import json
import time
from typing import Any, Dict, Tuple

from qdrant_client import QdrantClient, models


QUERIES_FILE = "dataset/queries_embeddings.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"

k = 10


def evaluate(client: QdrantClient, dataset: Dict[str, Any]) -> Dict[str, float]:
    ann_times = []
    knn_times = []
    ann_precision = []

    for text, embedding in dataset.items():
        print(f"Processing query: {text}")
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME, query=embedding, limit=k
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)
        print(f"ANN query time: {ann_time * 1000:.2f} ms")

        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_time = time.time() - start_time_knn
        knn_times.append(knn_time)
        print(f"Exact k-NN query time: {knn_time * 1000:.2f} ms")

        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / k
        ann_precision.append(precision)
        print(f"Precision@{k}: {precision:.4f}\n")

    avg_precision = sum(ann_precision) / len(dataset)
    avg_ann_time = sum(ann_times) / len(dataset)
    avg_knn_time = sum(knn_times) / len(dataset)
    result = {
        "avg_precision": avg_precision,
        "avg_ann_time": avg_ann_time,
        "avg_knn_time": avg_knn_time,
    }
    return result


def display_results(results: Dict[Tuple[int, int], Dict[str, float]]):
    """Display the results of the evaluation."""
    print("\nResults:")
    print("-" * 80)
    print(
        f"{'m':<5} {'ef_construct':<15} {'Precision@' + str(k):<15} {'ANN Time (ms)':<15} {'k-NN Time (ms)':<15}"
    )
    print("-" * 80)

    for (m, ef_construct), metrics in results.items():
        print(
            f"{m:<5} {ef_construct:<15} "
            f"{metrics['avg_precision']:.4f}{'':<9} "
            f"{metrics['avg_ann_time'] * 1000:.2f}{'':<10} "
            f"{metrics['avg_knn_time'] * 1000:.2f}"
        )

    print("-" * 80)


def main():
    """Main function to run the evaluation."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    test_cases = [
        (8, 50),
        (8, 100),
        (16, 32),
        (16, 50),
    ]

    results = {}
    for m, ef_construct in test_cases:
        print(f"Running evaluation with m={m}, ef_construct={ef_construct}")
        print("Updating collection parameters...")
        start_time = time.time()
        client.update_collection(
            collection_name=COLLECTION_NAME,
            hnsw_config=models.HnswConfigDiff(
                m=m,
                ef_construct=ef_construct,
            ),
        )
        end_time = time.time()
        print(f"Collection parameters updated in {end_time - start_time:.2f} seconds")
        result = evaluate(client, dataset=test_dataset)
        results[(m, ef_construct)] = result

    display_results(results)


if __name__ == "__main__":
    main()
