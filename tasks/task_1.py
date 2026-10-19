import json
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
K = 10
QUERIES_FILE = Path(__file__).resolve().parent.parent / "dataset" / "queries_embeddings.json"


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


def precision_at_k(ann_result, knn_result, k):
    """|ANN ids ∩ exact ids| / k for one query."""
    ann_ids = {item.id for item in ann_result}
    knn_ids = {item.id for item in knn_result}
    return len(ann_ids.intersection(knn_ids)) / k


def evaluate_precision(client, test_dataset, k=K):
    """Time default and exact query_points over the test embeddings and print the averages."""
    precisions = []
    ann_times = []
    knn_times = []

    for embedding in test_dataset.values():
        start_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
        ).points
        ann_times.append(time.time() - start_ann)

        start_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_times.append(time.time() - start_knn)

        precisions.append(precision_at_k(ann_result, knn_result, k))

    n = len(precisions)
    avg_precision = sum(precisions) / n
    avg_ann_time = sum(ann_times) / n
    avg_knn_time = sum(knn_times) / n
    result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)


def main():
    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    evaluate_precision(client, test_dataset, k=K)


if __name__ == "__main__":
    main()
