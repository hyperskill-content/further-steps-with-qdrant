"""
Task 1: Catching up
Compares ANN (default HNSW) search vs. exact k-NN search on the arxiv_papers
Qdrant collection, using a set of 100 pre-embedded test queries.

Computes:
  - average precision@10 (ANN results vs. exact results as the golden set)
  - average query time for ANN search
  - average query time for exact k-NN search

Usage:
    uv run python task_1.py
"""

import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6335  # arxiv_papers collection recovered from Stage 1 storage
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "dataset/queries_embeddings.json"
K = 10


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    precisions = []
    ann_times = []
    knn_times = []

    for query, embedding in test_dataset.items():
        # Approximate (default HNSW) search
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)

        # Exact k-NN search (brute force baseline)
        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_time = time.time() - start_time_knn
        knn_times.append(knn_time)

        # Precision@k: overlap between ANN result set and exact result set
        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / K
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)

    result_formatting(K, avg_precision, avg_ann_time, avg_knn_time)


if __name__ == "__main__":
    main()
