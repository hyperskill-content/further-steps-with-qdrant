import asyncio
import time
import json
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams, ScoredPoint
from typing import List

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "./dataset/queries_embeddings.json"

k = 10
precisions = []
ann_times = []
knn_times = []

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_result(embedding: List[float], exact: bool) -> List[ScoredPoint]:
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        search_params=SearchParams(exact=exact),
        limit=k
    ).points


def get_ann_result(embedding: List[float]) -> List[ScoredPoint]:
    start_time_ann = time.time()
    ann_result = get_result(embedding, False)
    ann_time = time.time() - start_time_ann
    ann_times.append(ann_time)
    return ann_result


def get_knn_result(embedding: List[float]) -> List[ScoredPoint]:
    start_time_knn = time.time()
    knn_result = get_result(embedding, True)
    knn_time = time.time() - start_time_knn
    knn_times.append(knn_time)
    return knn_result


def result_formatting(k: int, avg_precision, avg_ann_time: float, avg_knn_time: float):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average k-NN query time: {avg_knn_time * 1000:.2f} ms")


async def main():
    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    for query_embedding in test_dataset.values():
        knn_result = get_knn_result(query_embedding)
        knn_ids = set(item.id for item in knn_result)

        ann_result = get_ann_result(query_embedding)
        ann_ids = set(item.id for item in ann_result)

        precision = len(ann_ids.intersection(knn_ids)) / k
        precisions.append(precision)

    avg_knn_time = sum(knn_times) / len(knn_times) if knn_times else 0.0
    avg_ann_time = sum(ann_times) / len(ann_times) if ann_times else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

    result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)


if __name__ == "__main__":
    asyncio.run(main())