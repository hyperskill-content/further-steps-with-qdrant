import json
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
K = 10
HNSW_EF_VALUES = [10, 20, 50, 100, 200]
QUERIES_FILE = Path(__file__).resolve().parent.parent / "dataset" / "queries_embeddings.json"
WARMUP_QUERIES = 10


def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):
    """Score each hnsw_ef against one exact top-10 pass on arxiv_papers.

    ground_truth is keyed by the query text. Ten untimed query_points calls
    at hnsw_ef=200 run before the timed sweep.
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    ground_truth = {}
    for query, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        ground_truth[query] = {item.id for item in knn_result}

    warmup_embeddings = list(test_dataset.values())[:WARMUP_QUERIES]
    for embedding in warmup_embeddings:
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(hnsw_ef=max(hnsw_ef_values)),
        )

    results = []
    for hnsw_ef in hnsw_ef_values:
        precisions = []
        query_times = []
        for query, embedding in test_dataset.items():
            start = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            ).points
            query_times.append(time.time() - start)

            ann_ids = {item.id for item in ann_result}
            precisions.append(len(ann_ids.intersection(ground_truth[query])) / k)

        results.append(
            {
                "hnsw_ef": hnsw_ef,
                "avg_precision": sum(precisions) / len(precisions),
                "avg_query_time_ms": (sum(query_times) / len(query_times)) * 1000,
            }
        )

    return results


def main():
    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    results = evaluate_hnsw_ef(K, HNSW_EF_VALUES, test_dataset)
    for row in results:
        print(
            f"hnsw_ef={row['hnsw_ef']}: "
            f"avg_precision={row['avg_precision']:.4f}, "
            f"avg_query_time_ms={row['avg_query_time_ms']:.2f}"
        )


if __name__ == "__main__":
    main()
