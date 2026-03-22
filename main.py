import time
from qdrant_client import QdrantClient, models
import json

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QUERIES_FILE = "/Users/bdhakshn/Downloads/queries_embeddings.json"
COLLECTION_NAME = 'arxiv_papers'
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

client.update_collection(
    collection_name=COLLECTION_NAME,
    optimizer_config=models.OptimizersConfigDiff(),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=False,
        )
    )
)

def display(avg_precision, avg_query_time_ms, rescore):
    label = f"rescore={rescore}"
    print(f"\n{'=' * 45}")
    print(f"  Results with {label}")
    print(f"{'=' * 45}")
    print(f"  Avg Precision:       {avg_precision:.4f}")
    print(f"  Avg Query Time (ms): {avg_query_time_ms:.4f}")
    print(f"{'=' * 45}\n")

def evaluate(k, dataset, rescore: bool):
    ground_truth = {}
    print(f"Loaded {len(dataset)} queries.")

    for query_text, embedding in dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                exact=True,
                quantization=models.QuantizationSearchParams(ignore=True)
            )
        ).points
        ground_truth[query_text] = set(item.id for item in knn_result)

    query_times = []
    precisions  = []

    for query_text, embedding in dataset.items():
        start_time = time.time()

        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=rescore,
                    oversampling=2.0
                )
            )
        ).points

        elapsed_ms = (time.time() - start_time) * 1000
        query_times.append(elapsed_ms)

        ann_ids   = set(item.id for item in ann_result)
        exact_ids = ground_truth[query_text]
        precision = len(ann_ids.intersection(exact_ids)) / k
        precisions.append(precision)

    avg_precision    = sum(precisions)  / len(precisions)
    avg_query_time   = sum(query_times) / len(query_times)

    display(avg_precision, avg_query_time, rescore)

if __name__ == "__main__":
    with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)
    evaluate(k, test_dataset, rescore=True)
    evaluate(k, test_dataset, rescore=False)
