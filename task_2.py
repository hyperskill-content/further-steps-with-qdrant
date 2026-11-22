"""
Task 2 / Stage 2 — Balance the Search

Sweep HNSW search-time parameter hnsw_ef and measure:
  - average precision@10 vs exact k-NN ground truth
  - average ANN query latency (ms)

hnsw_ef = size of the candidate priority queue during search ("exploration factor").
Larger ef → more exploration → usually higher recall/precision, slower queries.

Run (Qdrant up + arxiv_papers loaded):
  poetry -C ../vector-db-qdrant run python \\
    /Users/upinder.dhami/workday/src/ai-course/further-steps-with-qdrant/task_2.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
K = 10
HNSW_EF_VALUES = [10, 20, 50, 100, 200]
QUERIES_FILE = Path(__file__).resolve().parent / "dataset" / "queries_embeddings.json"
WARMUP_QUERIES = 5


def build_ground_truth(
    client: QdrantClient,
    test_dataset: dict,
    k: int = K,
) -> dict[str, set]:
    """Exact k-NN top-k point ids per natural-language query key."""
    ground_truth: dict[str, set] = {}
    for i, (query, embedding) in enumerate(test_dataset.items(), start=1):
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        ground_truth[query] = {item.id for item in knn_result}
        if i % 20 == 0 or i == 1:
            print(f"  ground truth {i}/{len(test_dataset)} ...")
    return ground_truth


def warm_cache(client: QdrantClient, test_dataset: dict) -> None:
    """Throwaway ANN queries so disk/page-cache cost is not billed to ef=10."""
    print(f"Warming cache with {WARMUP_QUERIES} throwaway queries ...")
    for embedding in list(test_dataset.values())[:WARMUP_QUERIES]:
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(hnsw_ef=50),
        )


def evaluate_hnsw_ef(
    k: int,
    hnsw_ef_values: list[int],
    test_dataset: dict,
    client: QdrantClient,
) -> list[dict]:
    """
    For each hnsw_ef: average precision@k vs exact ground truth + avg query time ms.
    """
    print("Building exact k-NN ground truth ...")
    ground_truth = build_ground_truth(client, test_dataset, k=k)
    warm_cache(client, test_dataset)

    results: list[dict] = []
    for hnsw_ef in hnsw_ef_values:
        print(f"Evaluating hnsw_ef={hnsw_ef} ...")
        precisions: list[float] = []
        times: list[float] = []

        for query, embedding in test_dataset.items():
            start = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            ).points
            times.append(time.time() - start)

            ann_ids = {item.id for item in ann_result}
            precision = len(ann_ids.intersection(ground_truth[query])) / k
            precisions.append(precision)

        # Discard first query timing (still can be slightly cold per-ef); average rest if possible.
        timed = times[1:] if len(times) > 1 else times
        avg_precision = sum(precisions) / len(precisions)
        avg_query_time_ms = (sum(timed) / len(timed)) * 1000

        row = {
            "hnsw_ef": hnsw_ef,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_query_time_ms,
        }
        results.append(row)
        print(
            f"  → precision={avg_precision:.4f}, "
            f"avg_query_time_ms={avg_query_time_ms:.3f}"
        )

    return results


def main() -> None:
    print("Task 2 — Balance the Search (hnsw_ef sweep)")
    with open(QUERIES_FILE, encoding="utf-8") as file:
        test_dataset = json.load(file)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    info = client.get_collection(COLLECTION_NAME)
    print(f"  points: {info.points_count}")

    results = evaluate_hnsw_ef(K, HNSW_EF_VALUES, test_dataset, client)
    print()
    print("results =")
    print(results)


if __name__ == "__main__":
    main()
