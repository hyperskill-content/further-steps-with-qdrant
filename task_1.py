"""
Task 1 / Stage 1 — Catching Up

Compare Approximate Nearest Neighbor (ANN / HNSW default) vs exact k-NN
on the existing `arxiv_papers` collection.

For each of 100 precomputed query embeddings:
  - run ANN search (limit=10)
  - run exact k-NN search (search_params.exact=True, limit=10)
  - precision@10 = |ANN ids ∩ Exact ids| / 10
  - time both queries

Print average precision@10 and average latencies.

Run (Qdrant up with arxiv_papers loaded from the prior project):
  # Workday laptop: reuse the vector-db-qdrant Poetry env if `uv sync` hits PyPI 403
  poetry -C ../vector-db-qdrant run python task_1.py
  # or, after a successful `uv sync`:
  # uv run python task_1.py
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
QUERIES_FILE = Path(__file__).resolve().parent / "dataset" / "queries_embeddings.json"


def result_formatting(k: int, avg_precision: float, avg_ann_time: float, avg_knn_time: float) -> None:
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


def precision_at_k(ann_result, knn_result, k: int) -> float:
    ann_ids = {item.id for item in ann_result}
    knn_ids = {item.id for item in knn_result}
    return len(ann_ids.intersection(knn_ids)) / k


def evaluate(client: QdrantClient, test_dataset: dict) -> tuple[float, float, float]:
    precisions: list[float] = []
    ann_times: list[float] = []
    knn_times: list[float] = []

    for i, (_query, embedding) in enumerate(test_dataset.items(), start=1):
        start_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
        ).points
        ann_times.append(time.time() - start_ann)

        start_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_times.append(time.time() - start_knn)

        precisions.append(precision_at_k(ann_result, knn_result, K))

        if i % 10 == 0 or i == 1:
            print(f"  processed {i}/{len(test_dataset)} queries ...")

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)
    return avg_precision, avg_ann_time, avg_knn_time


def main() -> None:
    print("Task 1 — Catching Up (ANN vs exact k-NN)")
    print(f"  collection: {COLLECTION_NAME}")
    print(f"  queries:    {QUERIES_FILE}")

    with open(QUERIES_FILE, encoding="utf-8") as file:
        test_dataset = json.load(file)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    # Fail fast if collection is missing
    info = client.get_collection(COLLECTION_NAME)
    print(f"  points:     {info.points_count}")

    avg_precision, avg_ann_time, avg_knn_time = evaluate(client, test_dataset)
    print()
    result_formatting(K, avg_precision, avg_ann_time, avg_knn_time)


if __name__ == "__main__":
    main()
