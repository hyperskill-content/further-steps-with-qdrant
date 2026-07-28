from qdrant_client import QdrantClient, models
import time
import json

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
k = 10
QUERIES_FILE = 'dataset/queries_embeddings.json'

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Print results.
def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    """
    Formats and prints the results of the precision@k calculation and query times.

    Parameters:
    - k: integer, number of top results to consider
    - avg_precision: float, average precision@k score
    - avg_ann_time: float, average time for approximate (default) search query
    - avg_knn_time: float, average time for exact k-NN search query
    """
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')


def precision_at_k(test_dataset, k=10):

    # Iterate over test_dataset to get embeddings and
    # run the exact and the approximate searches, calculate precision, and log the time for each query.
    precisions = []
    ann_times = []
    knn_times = []

    for query, embedding in test_dataset.items():
        # ANN (Approximate Nearest Neighbor) search
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)

        # Exact k-NN search
        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_time = time.time() - start_time_knn
        knn_times.append(knn_time)

        # Calculate precision@k
        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        # print(f"Query: {query}")
        # print(f"ANN IDs: {ann_ids}")
        # print(f"k-NN IDs: {knn_ids}")
        precision = len(ann_ids.intersection(knn_ids)) / k
        print(f"Query: {query}, Precision: {precision}")
        # print(f"-"*100)
        precisions.append(precision)

    # Calculate averages
    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)

    # Display results
    print(f"-" * 100)
    result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)

# Load test dataset: 100 queries and their corresponding text-embedding-ada-002 embeddings.
with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)

precision_at_k(test_dataset, k=10)

