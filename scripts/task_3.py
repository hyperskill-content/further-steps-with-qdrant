from qdrant_client import QdrantClient, models
from pathlib import Path
import time, json


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
SCRIPT_DIR = Path(__file__).resolve().parent
QUERIES_FILE = SCRIPT_DIR / ".." / "dataset" / "queries_embeddings.json"
k = 10
hnsw_ef_values = [10, 20, 50, 100, 200]

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)

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


def exact_search(k: int, embedding: list):
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        with_payload=True,
        timeout=30,
        search_params=models.SearchParams(
            exact=True,
            quantization=models.QuantizationSearchParams(ignore=True)) # ignore is False by default),
    ).points
    return result

    
def approx_search(k: int, embedding: list, ef: int, rescore=True):
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        with_payload=True,
        timeout=30,
        search_params=models.SearchParams(
            hnsw_ef=ef,
            quantization=models.QuantizationSearchParams(
                rescore=rescore,
                oversampling=2.0,
            )
        )
    ).points
    return result


def evaluate_hnsw_ef(k: int, 
                     hnsw_ef_values: list[int], 
                     ground_truth: dict[str, list], 
                     dataset: dict[str, list],
                     warming=False,
                     rescore=True):
    results = []
    for ef in hnsw_ef_values:
        if not warming:
            print(f"Running test dataset at hnsw_ef={ef}...")
        ann_timings = []
        precision_log = []

        for query, embedding in dataset.items():
            ann_ids = []
            start_time = time.time()
            result = approx_search(k, embedding, ef, rescore)
            if not warming:
                elapsed_time = time.time() - start_time
                ann_timings.append(elapsed_time)
                ann_ids = set([point.id for point in result])
                knn_ids = set(ground_truth[query])
                precision = len(ann_ids.intersection(knn_ids)) / k
                precision_log.append(precision)

        if not warming:
            avg_precision = sum(precision_log) / len(precision_log)
            avg_query_time_ms = sum(ann_timings) / len(ann_timings)
            hnsw_stats = {"hnsw_ef": ef, "avg_precision": avg_precision, "avg_query_time_ms": avg_query_time_ms}
            results.append(hnsw_stats)
    return results    


def result_formatting(hnsw_stats):
    print("hnsw_ef evaluation results:")
    for stat in hnsw_stats:
        print(f'hnsw_ef: {stat["hnsw_ef"]}, '
              f'avg_precision: {stat["avg_precision"]:.4f}, '
              f'avg_query_time_ms: {stat["avg_query_time_ms"] * 1000:.2f}')


# Create ground_truth dictionary
ground_truth = {}
print("Creating ground_truth dictionary...")
for query, embedding in test_dataset.items():
    result = exact_search(k=10, embedding=embedding)
    ids = [point.id for point in result]
    ground_truth[query] = ids

# Cache warming
# Perform evaluate_hnsw_ef with a smaller dataset, and discard the results
print("Warming cache...")
evaluate_hnsw_ef(5, [100], ground_truth, test_dataset, warming=True)

# hnsw evaluation - first with rescore=True, then with rescore=False
eval_rescore_true = evaluate_hnsw_ef(k, hnsw_ef_values, ground_truth, test_dataset, False, True)
eval_rescore_false = evaluate_hnsw_ef(k, hnsw_ef_values, ground_truth, test_dataset, False, False)

print("\nrescore=True")
result_formatting(eval_rescore_true)
print("\nrescore=False")
result_formatting(eval_rescore_false)
