import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "../dataset/queries_embeddings.json"
k = 10
hnsw_ef_values = [10, 20, 50, 100, 200]

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):
    # run exact search once for each query to use as ground truth
    ground_truth = {}
    for query, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        ground_truth[query] = set(item.id for item in knn_result)

    results = []
    precisions = []
    times = []

    for hnsw_ef in hnsw_ef_values:
        for query, embedding in test_dataset.items():
            start = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            ).points
            times.append(time.time() - start)

            ann_ids = set(item.id for item in ann_result)
            precision = len(ann_ids.intersection(ground_truth[query])) / k
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)
        avg_time_ms = (sum(times) / len(times)) * 1000

        results.append({
            "hnsw_ef": hnsw_ef,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_time_ms,
        })

    for r in results:
        print(f"hnsw_ef={r['hnsw_ef']:>4} | precision@{k}: {r['avg_precision']:.4f} | avg time: {r['avg_query_time_ms']:.2f} ms")


if __name__ == "__main__":
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)

    evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset)
