from qdrant_client import QdrantClient, models
import time
import json

def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')

def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):
    ground_truth = {}
    for query, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        ground_truth[query] = {p.id for p in knn_result}

    # warm up cache with some throwaway queries
    warm_embeddings = list(test_dataset.values())[:10]
    for emb in warm_embeddings:
        client.query_points(collection_name=COLLECTION_NAME, query=emb, limit=k)

    results = []
    for hnsw_ef in hnsw_ef_values:
        times = []
        precisions = []
        for query, embedding in test_dataset.items():
            start = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            ).points
            times.append(time.time() - start)

            ann_ids = {p.id for p in ann_result}
            precision = len(ann_ids & ground_truth[query]) / k
            precisions.append(precision)

        results.append({
            "hnsw_ef": hnsw_ef,
            "avg_precision": sum(precisions) / len(precisions),
            "avg_query_time_ms": (sum(times) / len(times)) * 1000,
        })

    return results



QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
QUERIES_FILE = "dataset/queries_embeddings.json"
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

with open(QUERIES_FILE, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)

results = evaluate_hnsw_ef(
    k=10,
    hnsw_ef_values=[10, 20, 50, 100, 200],
    test_dataset=test_dataset,
)

print(f"{'hnsw_ef':>8} | {'avg precision':>13} | {'avg time (ms)':>13}")
print("-" * 42)
for r in results:
    print(f"{r['hnsw_ef']:>8} | {r['avg_precision']:>13.4f} | {r['avg_query_time_ms']:>13.2f}")

