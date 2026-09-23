from qdrant_client import QdrantClient, models
import time
import json

def evaluate_quantization(k, test_dataset, ground_truth, rescore):
    times, precisions = [], []
    for query, embedding in test_dataset.items():
        start = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=rescore,
                    oversampling=2.0,
                )
            ),
        ).points
        times.append(time.time() - start)
        ann_ids = {p.id for p in ann_result}
        precisions.append(len(ann_ids & ground_truth[query]) / k)
    return {
        "rescore": rescore,
        "avg_precision": sum(precisions) / len(precisions),
        "avg_query_time_ms": (sum(times) / len(times)) * 1000,
    }


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
QUERIES_FILE = "dataset/queries_embeddings.json"
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

with open(QUERIES_FILE, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)


ground_truth = {}
for query, embedding in test_dataset.items():
    knn_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=models.SearchParams(
            exact=True,
            quantization=models.QuantizationSearchParams(ignore=True)
        ),
    ).points
    ground_truth[query] = {p.id for p in knn_result}

# warm up cache with some throwaway queries
warm_embeddings = list(test_dataset.values())[:10]
for emb in warm_embeddings:
    client.query_points(collection_name=COLLECTION_NAME, query=emb, limit=k)

results = [evaluate_quantization(k, test_dataset, ground_truth, r) for r in (True, False)]

print(f"{'rescore':>8} | {'avg precision':>13} | {'avg time (ms)':>13}")
print("-" * 42)
for r in results:
    print(f"{str(r['rescore']):>8} | {r['avg_precision']:>13.4f} | {r['avg_query_time_ms']:>13.2f}")
