import json
import time

from qdrant_client import QdrantClient, models


def result_formatting(
    k: int, avg_precision: float, avg_ann_time: float, avg_knn_time: float
):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


QUERIES_FILE = "dataset/queries_embeddings.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"

k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

with open(QUERIES_FILE, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)

ann_times = []
knn_times = []
ann_precision = []

for text, embedding in test_dataset.items():
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

avg_precision = sum(ann_precision) / len(test_dataset)
avg_ann_time = sum(ann_times) / len(test_dataset)
avg_knn_time = sum(knn_times) / len(test_dataset)

result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)
