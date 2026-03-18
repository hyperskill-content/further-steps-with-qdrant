import json
import time
from qdrant_client import QdrantClient, models

# Constants
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
QUERIES_FILE = '../dataset/queries_embeddings.json'
K = 10

def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')

def main():
    # Initialize Qdrant client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Load the test dataset
    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
            test_dataset = json.load(file)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {QUERIES_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {QUERIES_FILE}")
        return

    ann_times = []
    knn_times = []
    precisions = []

    # Iterate through each query and its embedding
    for query_text, embedding in test_dataset.items():
        # 1. Approximate (ANN) Search
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)

        # 2. Exact (k-NN) Search
        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_time = time.time() - start_time_knn
        knn_times.append(knn_time)

        # 3. Calculate Precision@K
        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        
        if K > 0:
            precision = len(ann_ids.intersection(knn_ids)) / K
        else:
            precision = 0
        precisions.append(precision)

    # Calculate averages
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_ann_time = sum(ann_times) / len(ann_times) if ann_times else 0
    avg_knn_time = sum(knn_times) / len(knn_times) if knn_times else 0

    # Display results
    result_formatting(K, avg_precision, avg_ann_time, avg_knn_time)

if __name__ == "__main__":
    main()
