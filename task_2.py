"""
Task 2: Balance the search
Sweeps the hnsw_ef search parameter over several values and measures how it
affects the precision@10 and query speed of approximate (HNSW) search on the
arxiv_papers Qdrant collection, using the same 100 pre-embedded test queries
as Task 1. Exact k-NN search is used as the ground truth for precision.

Usage:
    uv run python task_2.py
"""

import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6335  # arxiv_papers collection recovered from Stage 1 storage
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "dataset/queries_embeddings.json"
K = 10
HNSW_EF_VALUES = [10, 20, 50, 100, 200]
WARMUP_QUERIES = 5  # number of throwaway approximate queries to warm the cache


def result_formatting(results):
    print(f"{'hnsw_ef':>8} | {'avg_precision':>14} | {'avg_query_time_ms':>18}")
    print("-" * 48)
    for r in results:
        print(
            f"{r['hnsw_ef']:>8} | {r['avg_precision']:>14.4f} "
            f"| {r['avg_query_time_ms']:>18.2f}"
        )


def evaluate_hnsw_ef(client, k, hnsw_ef_values, test_dataset):
    # --- Ground truth: exact k-NN search for every query ---
    ground_truth = {}
    for query, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        ground_truth[query] = set(item.id for item in knn_result)

    # --- Warm up the cache before timing the hnsw_ef sweep ---
    warmup_items = list(test_dataset.items())[:WARMUP_QUERIES]
    for _, embedding in warmup_items:
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(hnsw_ef=hnsw_ef_values[0]),
        )

    # --- Sweep over hnsw_ef values ---
    results = []
    for hnsw_ef in hnsw_ef_values:
        precisions = []
        query_times = []

        for query, embedding in test_dataset.items():
            start_time = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            ).points
            query_time = time.time() - start_time
            query_times.append(query_time)

            ann_ids = set(item.id for item in ann_result)
            gt_ids = ground_truth[query]
            precision = len(ann_ids.intersection(gt_ids)) / k
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)
        avg_query_time_ms = (sum(query_times) / len(query_times)) * 1000

        results.append({
            "hnsw_ef": hnsw_ef,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_query_time_ms,
        })

    return results


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    results = evaluate_hnsw_ef(
        client=client,
        k=K,
        hnsw_ef_values=HNSW_EF_VALUES,
        test_dataset=test_dataset,
    )

    result_formatting(results)


if __name__ == "__main__":
    main()
