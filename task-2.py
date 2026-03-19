import json
import time
from qdrant_client import QdrantClient, models
import os
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
TARGET_FILE = os.path.join(
    os.path.dirname(__file__), "dataset/queries_embeddings.json"
)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


with open(TARGET_FILE, "r", encoding="utf-8") as f:
    test_dataset = json.load(f)


def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):
    ground_truth = {}
    results = []

    # Step 1: build ground truth using exact search
    for query_text, embedding in test_dataset.items():
        exact_result = client.query_points(
            collection_name=COLLECTION_NAME, query=embedding, limit=k
        ).points

        exact_ids = [point.id for point in exact_result]
        ground_truth[query_text] = exact_ids

    # Step 2: test each hnsw_ef value
    for hnsw_ef in hnsw_ef_values:
        precisions = []
        query_times_ms = []

        for query_text, embedding in test_dataset.items():
            start_time = time.time()

            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            ).points

            end_time = time.time()

            elapsed_ms = (end_time - start_time) * 1000
            query_times_ms.append(elapsed_ms)

            ann_ids = [point.id for point in ann_result]
            exact_ids = ground_truth[query_text]

            matched_ids = set(ann_ids).intersection(set(exact_ids))
            precision = len(matched_ids) / k
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)
        avg_query_time_ms = sum(query_times_ms) / len(query_times_ms)

        results.append(
            {
                "hnsw_ef": hnsw_ef,
                "avg_precision": avg_precision,
                "avg_query_time_ms": avg_query_time_ms,
            }
        )

    return results


k = 10
hnsw_ef_values = [10, 20, 50, 100, 200]

results = evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset)

for item in results:
    print(item)
