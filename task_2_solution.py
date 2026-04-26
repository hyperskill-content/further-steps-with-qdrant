import json
import time
from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
QUERIES_FILE = 'dataset/queries_embeddings.json'
k = 10

def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
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
    for hnsw_ef in hnsw_ef_values:
        precisions = []
        query_times = []
        
        for query_text, embedding in test_dataset.items():
            start_time = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef)
            ).points
            query_times.append(time.time() - start_time)
            
            ann_ids = set(item.id for item in ann_result)
            precision = len(ann_ids.intersection(ground_truth[query_text])) / k
            precisions.append(precision)
        
        avg_precision = sum(precisions) / len(precisions)
        avg_query_time_ms = (sum(query_times) / len(query_times)) * 1000
        
        results.append({
            "hnsw_ef": hnsw_ef,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_query_time_ms
        })
    
    return results

with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)

hnsw_ef_values = [10, 20, 50, 100, 200]
results = evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset)

for result in results:
    print(f"hnsw_ef={result['hnsw_ef']}: avg_precision={result['avg_precision']:.4f}, avg_query_time={result['avg_query_time_ms']:.2f} ms")
