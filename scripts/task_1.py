from qdrant_client import QdrantClient, models
from pathlib import Path
import time, json


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
SCRIPT_DIR = Path(__file__).resolve().parent
QUERIES_FILE = SCRIPT_DIR / ".." / "dataset" / "queries_embeddings.json"
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)


def do_query(embedding: list, k: int, exact: bool=False):
    if exact:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
    else:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(hnsw_ef=1),
        ).points
    return result


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


ann_times, knn_times, precision_log = [], [], []

for query, embedding in test_dataset.items():
    # Perform the default query
    start_time_ann = time.time()
    ann_result = do_query(embedding, k)
    ann_time = time.time() - start_time_ann
    ann_times.append(ann_time)

    # Perform the exact query
    start_time_knn = time.time()
    knn_result = do_query(embedding, k, exact=True)
    knn_time = time.time() - start_time_knn
    knn_times.append(knn_time)

    # Precision calculation
    ann_ids = set(item.id for item in ann_result)
    knn_ids = set(item.id for item in knn_result)
    precision = len(ann_ids.intersection(knn_ids)) / k
    precision_log.append(precision)

avg_ann_time = sum(ann_times) / len(ann_times)
avg_knn_time = sum(knn_times) / len(knn_times)
avg_precision = sum(precision_log) / len(precision_log)

result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)
