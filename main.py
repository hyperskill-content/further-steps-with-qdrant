from qdrant_client import QdrantClient, models
import json
import time
from pathlib import Path
from dotenv import load_dotenv


QUERIES_FILE = Path(__file__).parent / "dataset" / "queries_embeddings.json"

with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)

load_dotenv()

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# For each embedding in test_dataset.values() (or each query, embedding pair via test_dataset.items()),
# Run the exact and the approximate searches, calculate the precision, and log the time for a single query.

ann_times = []
knn_times = []
precisions = []
for query, embedding in test_dataset.items():
    # Run exact and approximate searches / Log the average time for a single query
    # Calculate precision / Append the logged times and the precision to their corresponding lists.

    start_time_ann = time.time()
    ann_result = client.query_points (
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k
    ).points
    ann_time = time.time() - start_time_ann
    ann_times.append(ann_time)
    # print(f"ANN query time: {ann_time * 1000:.2f} ms")

    start_time_knn = time.time()
    knn_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=models.SearchParams(exact=True),
    ).points
    knn_time = time.time() - start_time_knn
    knn_times.append(knn_time)
    # print(f"KNN query time: {knn_time * 1000:.2f} ms")

    ann_ids = set(item.id for item in ann_result)
    knn_ids = set(item.id for item in knn_result)
    precision = len(ann_ids.intersection(knn_ids)) / k
    precisions.append(precision)
    # print(f"Precision@{k}: {precision:.4f}")

averages = {
    "avg_ann_time": sum(ann_times) / len(ann_times),
    "avg_knn_time": sum(knn_times) / len(knn_times),
    "avg_precision": sum(precisions) / len(precisions),
}

# Display the averages and write up a reflection on the obtained results
def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')


if __name__ == "__main__":
    result_formatting (10, averages["avg_precision"], averages["avg_ann_time"], averages["avg_knn_time"])
