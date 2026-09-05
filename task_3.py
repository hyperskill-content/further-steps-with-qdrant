"""
Task 3: Quantization
Applies scalar (int8) quantization to the arxiv_papers collection and compares
ANN search precision/speed with rescore=True vs. rescore=False, against an
exact (non-quantized) k-NN baseline.

Usage:
    uv run python task_3.py
"""

import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6335  # arxiv_papers collection recovered from Stage 1 storage
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "dataset/queries_embeddings.json"
K = 10
OVERSAMPLING = 2.0
WARMUP_QUERIES = 5  # number of throwaway approximate queries to warm the cache


def result_formatting(results):
    print(f"{'rescore':>8} | {'avg_precision':>14} | {'avg_query_time_ms':>18}")
    print("-" * 48)
    for r in results:
        print(
            f"{str(r['rescore']):>8} | {r['avg_precision']:>14.4f} "
            f"| {r['avg_query_time_ms']:>18.2f}"
        )


def apply_scalar_quantization(client):
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


def build_ground_truth(client, k, test_dataset):
    """Exact k-NN search, ignoring quantized vectors, used as the golden set."""
    ground_truth = {}
    for query, embedding in test_dataset.items():
        exact_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                exact=True,
                quantization=models.QuantizationSearchParams(ignore=True),
            ),
        ).points
        ground_truth[query] = set(item.id for item in exact_result)
    return ground_truth


def evaluate_quantized_search(client, k, test_dataset, ground_truth, rescore, oversampling):
    precisions = []
    query_times = []

    for query, embedding in test_dataset.items():
        start_time = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=rescore,
                    oversampling=oversampling,
                )
            ),
        ).points
        query_time = time.time() - start_time
        query_times.append(query_time)

        ann_ids = set(item.id for item in ann_result)
        gt_ids = ground_truth[query]
        precision = len(ann_ids.intersection(gt_ids)) / k
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions)
    avg_query_time_ms = (sum(query_times) / len(query_times)) * 1000

    return avg_precision, avg_query_time_ms


def warm_cache(client, k, test_dataset, oversampling):
    warmup_items = list(test_dataset.items())[:WARMUP_QUERIES]
    for _, embedding in warmup_items:
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=oversampling,
                )
            ),
        )


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    # Step 1: enable scalar (int8) quantization on the collection
    apply_scalar_quantization(client)

    # Step 2: build the ground truth using exact search on the original vectors
    ground_truth = build_ground_truth(client, K, test_dataset)

    # Step 3: warm the cache before timing
    warm_cache(client, K, test_dataset, OVERSAMPLING)

    # Step 4: run the quantized ANN search with rescore=True, then rescore=False
    results = []
    for rescore in (True, False):
        avg_precision, avg_query_time_ms = evaluate_quantized_search(
            client, K, test_dataset, ground_truth, rescore, OVERSAMPLING
        )
        results.append({
            "rescore": rescore,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_query_time_ms,
        })

    result_formatting(results)


if __name__ == "__main__":
    main()
