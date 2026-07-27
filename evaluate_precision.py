import argparse
import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "dataset/queries_embeddings.json"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


def evaluate_precision(test_dataset, k, ef):
    ann_times = []
    knn_times = []
    precisions = []

    for embedding in test_dataset.values():
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(hnsw_ef=ef),
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)

        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_time = time.time() - start_time_knn
        knn_times.append(knn_time)

        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / k
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)

    result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qdrant ANN precision.")
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve (default: 10)"
    )

    parser.add_argument(
        "-ef",
        type=int,
        default=128,
        help="hnsw_ef (default: 128)"
    )

    args = parser.parse_args()

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    evaluate_precision(test_dataset, k=args.k, ef=args.ef)