import time

from qdrant_client import QdrantClient, models
import json

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QUERIES_FILE = "/Users/bdhakshn/Downloads/queries_embeddings.json"
COLLECTION_NAME = 'arxiv_papers'
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')


def main():
    with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)

    print(f"Loaded {len(test_dataset)} queries.\n")

    ann_times = []
    knn_times = []
    precisions = []

    for query_text, embedding in test_dataset.items():
        start_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k
        ).points
        ann_times.append(time.time() - start_ann)

        start_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_times.append(time.time() - start_knn)

        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / k
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)

    formatting(k, avg_precision, avg_ann_time, avg_knn_time)


if __name__ == "__main__":
    main()
