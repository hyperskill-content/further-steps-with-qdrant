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
HNSW_EF_VALUES = [10, 20, 50, 100, 200]
logger.info("Connecting to QDrant server")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):
    ground_truth = {}
    for query_text, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        ground_truth[query_text] = set(item.id for item in knn_result)

    results = []

    for hsw_ef in hnsw_ef_values:
        logger.info(f"Testing hnsw_ef = {hsw_ef}")
        ann_times = []
        precisions = []

        for query_text, embedding in test_dataset.items():
            start_time_ann = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hsw_ef)
            ).points
            ann_time = time.time() - start_time_ann
            ann_times.append(ann_time)

            ann_ids = set(item.id for item in ann_result)
            truth_ids = ground_truth[query_text]

            precision = len(ann_ids.intersection(truth_ids)) / k
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)
        avg_time_ms = (sum(ann_times) / len(ann_times)) * 1000

        results.append({
            "hnsw_ef": hsw_ef,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_time_ms
        })

    return results


def main():
    logger.info("Loading test dataset")
    with open(DATASET, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)

    results = evaluate_hnsw_ef(K, HNSW_EF_VALUES, test_dataset)

    for res in results:
        print(f"ef={res['hnsw_ef']}: average precision={res['avg_precision']:.4f}, query time={res['avg_query_time_ms']:.2f} ms")


if __name__ == "__main__":
    main()
