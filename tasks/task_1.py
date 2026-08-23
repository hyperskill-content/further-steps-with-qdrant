from qdrant_client import QdrantClient, models
import json
import time
from typing import List

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
QUERIES_FILE = "C:\\Users\\gauth\\PycharmMisPcroject\\further-steps-with-qdrant\\dataset\\queries_embeddings.json"
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def result_formatting(K, avg_precision, avg_ann_time, avg_knn_time):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


def search_performance() -> None:
    with open(QUERIES_FILE, 'r') as file:
        test_dataset = json.load(file)

        precisions: List[float] = []
        ann_times: List[float] = []
        knn_times: List[float] = []

        for query_text, embedding in test_dataset.items():
            start_time_ann = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit= k
            ).points
            ann_time = time.time() - start_time_ann
            ann_times.append(ann_time)

            start_time_knn = time.time()
            knn_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit= k,
                search_params=models.SearchParams(exact=True),
            ).points
            knn_time = time.time() - start_time_knn
            knn_times.append(knn_time)

    ann_ids = set(item.id for item in ann_result)
    knn_ids = set(item.id for item in knn_result)

    precision = len(ann_ids.intersection(knn_ids)) / k
    precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)

    result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)

if __name__ == "__main__":
    search_performance()