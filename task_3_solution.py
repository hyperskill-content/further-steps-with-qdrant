import json
import time
from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
QUERIES_FILE = 'dataset/queries_embeddings.json'
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

print("Applying scalar quantization...")
client.update_collection(
    collection_name=COLLECTION_NAME,
    optimizer_config=models.OptimizersConfigDiff(),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=False,
        ),
    ),
)
print("Quantization applied. Waiting for optimization...")
time.sleep(5)

with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)

def evaluate_quantization(rescore):
    precisions = []
    ann_times = []
    knn_times = []
    
    for query_text, embedding in test_dataset.items():
        start_time = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                exact=True
            ),
        ).points
        knn_times.append(time.time() - start_time)
        
        start_time = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=rescore,
                    oversampling=2.0,
                )
            )
        ).points
        ann_times.append(time.time() - start_time)
        
        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / k
        precisions.append(precision)
    
    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times) * 1000
    avg_knn_time = sum(knn_times) / len(knn_times) * 1000
    
    return avg_precision, avg_ann_time, avg_knn_time

print("\n=== With rescore=True ===")
precision_rescore, ann_time_rescore, knn_time_rescore = evaluate_quantization(rescore=True)
print(f"Average precision@{k}: {precision_rescore:.4f}")
print(f"Average ANN query time: {ann_time_rescore:.2f} ms")
print(f"Average k-NN query time: {knn_time_rescore:.2f} ms")

print("\n=== With rescore=False ===")
precision_no_rescore, ann_time_no_rescore, knn_time_no_rescore = evaluate_quantization(rescore=False)
print(f"Average precision@{k}: {precision_no_rescore:.4f}")
print(f"Average ANN query time: {ann_time_no_rescore:.2f} ms")
print(f"Average k-NN query time: {knn_time_no_rescore:.2f} ms")
