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

#  Task-1:
#  ANN search yields relatively high precision@10 of 0.832
#  This suggests that most retreuved results overlap with exact k-NN results
#  ANN query time is 28.58 ms. This is significantly lower than the exact k-NN time, which is 73.39 ms.
#  This suggests that ANN shows strong time-performance advantage
#  Given relatively high precision and materially lower time ANN offers a good balance between accuracy and time-efficiency

