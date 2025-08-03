"""Task 3"""

import json
import time
from typing import Any, Dict

from qdrant_client import QdrantClient, models


QUERIES_FILE = "dataset/queries_embeddings.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"

k = 10


def evaluate(
    client: QdrantClient, dataset: Dict[str, Any], rescore: bool
) -> Dict[str, float]:
    """Evaluate the performance of ANN and k-NN search with rescore option."""
    ann_times = []
    knn_times = []
    ann_precision = []

    for text, embedding in dataset.items():
        print(f"Processing query: {text}")
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=rescore,
                    oversampling=2.0,
                )
            ),
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)
        print(f"ANN query time: {ann_time * 1000:.2f} ms")

        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                exact=True,
                quantization=models.QuantizationSearchParams(ignore=True),
            ),
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


def display_results(results: Dict[bool, Dict[str, float]]):
    """Display the results of the evaluation."""
    print("\nResults:")
    print("-" * 54)
    print(
        f"{'rescore':<7} "
        f"{'Precision@' + str(k):<15} "
        f"{'ANN Time (ms)':<15} "
        f"{'k-NN Time (ms)':<15}"
    )
    print("-" * 54)

    for rescore, metrics in results.items():
        print(
            f"{rescore:<7} "
            f"{metrics['avg_precision']:.4f}{'':<9} "
            f"{metrics['avg_ann_time'] * 1000:.2f}{'':<10} "
            f"{metrics['avg_knn_time'] * 1000:.2f}"
        )

    print("-" * 54)


def main():
    """Main function to run the evaluation."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    #  reset the HNSW config parameters to their default values
    client.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
    )

    # while True:
    #     collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    #     if collection_info.status == models.CollectionStatus.GREEN:
    #         break

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

    test_cases = [
        True,
        False,
    ]

    results = {}
    for rescore in test_cases:
        print(f"Running evaluation with rescore={rescore}")
        result = evaluate(client, dataset=test_dataset, rescore=rescore)
        results[rescore] = result

    display_results(results)


if __name__ == "__main__":
    main()
