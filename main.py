## Tast imports
## Service imports
import json
import os
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
TARGET_FILE = os.path.join(os.path.dirname(__file__), "dataset/queries_embeddings.json")
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')


with open(TARGET_FILE, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)

precisions = []
ann_times = []
knn_times = []

for query_text, embedding in test_dataset.items():
    # Approximate search
    start_time_ann = time.time()
    ann_result = client.query_points(
        collection_name=COLLECTION_NAME, query=embedding, limit=k
    ).points
    ann_time = time.time() - start_time_ann
    ann_times.append(ann_time)

    # Exact search
    start_time_knn = time.time()
    knn_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=models.SearchParams(exact=True),
    ).points
    knn_time = time.time() - start_time_knn
    knn_times.append(knn_time)

    # Precision@k
    ann_ids = set(item.id for item in ann_result)
    knn_ids = set(item.id for item in knn_result)

    precision = len(ann_ids.intersection(knn_ids)) / k
    precisions.append(precision)

avg_precision = sum(precisions) / len(precisions)
avg_ann_time = sum(ann_times) / len(ann_times)
avg_knn_time = sum(knn_times) / len(knn_times)

result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)

# Task 2: reflection
# Higher hnsw_ef improves precision
# hnsw_ef at 100 and more helps to reach perfect accuracy (1.0)
# This suggests that higher hnsw_ef produces results closer to exact k-NN
# However, with higher hnsw_ef time efficiency falls and time to execute search grows from 16.11 ms to 22.4 ms
# hnsw_ef value between 50 and 100 looks optimal for time-performance albeit difference between 16 ms and 22 ms is not huge and depends on the hardware (I believe)
