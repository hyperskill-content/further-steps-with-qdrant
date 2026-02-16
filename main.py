import json
import logging
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
K = 10
DATASET = Path("./dataset/queries_embeddings.json")
logger.info("Connecting to QDrant server")


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    logger.info("Loading test dataset")
    with open(DATASET, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)

    precisions = []
    ann_times = []
    knn_times = []

    total_queries = len(test_dataset)
    logger.info(f"Total queries: {total_queries}")

    for i, embedding in enumerate(test_dataset.values(), 1):
        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_time = time.time() - start_time_knn
        knn_times.append(knn_time)
        logger.info(f"Query points: {len(knn_result)}")

        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)
        logger.info(f"Query points: {len(ann_result)}")

        knn_ids = set(item.id for item in knn_result)
        ann_ids = set(item.id for item in ann_result)

        precision = len(ann_ids.intersection(knn_ids)) / K
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)

    result_formatting(K, avg_precision, avg_ann_time, avg_knn_time)


if __name__ == "__main__":
    main()
