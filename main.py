import time

from qdrant_client import QdrantClient, models
import json

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QUERIES_FILE = "/Users/bdhakshn/Downloads/queries_embeddings.json"
COLLECTION_NAME = 'arxiv_papers'
k = 10
HNSW_EF_VALUES = [10, 20, 50, 100, 200]

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def display_results(items_to_print):
    print("\n" + "=" * 55)
    print(f"{'hnsw_ef':<12} {'avg_precision':<20} {'avg_query_time_ms'}")
    print("=" * 55)
    for items in items_to_print:
        print(f"{items['hnsw_ef']:<12} {items['avg_precision']:<20.4f} {items['avg_query_time_ms']:.4f}")
    print("=" * 55)


def evaluate_hnsw_ef(results_count, hnsw_ef_values, dataset):
    ground_truth = {}
    print(f"Loaded {len(dataset)} queries.\n")
    for query_text, embedding in dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=results_count,
            search_params=models.SearchParams(exact=True),
        ).points
        ground_truth[query_text] = set(item.id for item in knn_result)

    resultsArray = []

    for hnsw_ef in hnsw_ef_values:
        query_times = []
        precisions = []
        for query_text, embedding in dataset.items():
            start_time = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=results_count,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef)  # ← key change
            ).points

            elapsed_ms = (time.time() - start_time) * 1000  # convert to ms
            query_times.append(elapsed_ms)
            ann_ids = set(item.id for item in ann_result)
            exact_ids = ground_truth[query_text]  # fetch ground truth
            precision = len(ann_ids.intersection(exact_ids)) / results_count
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)
        avg_query_time = sum(query_times) / len(query_times)

        resultsArray.append({
            "hnsw_ef": hnsw_ef,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_query_time
        })

    return resultsArray


if __name__ == "__main__":
    with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)
        results = evaluate_hnsw_ef(k, HNSW_EF_VALUES, test_dataset)
        display_results(results)
