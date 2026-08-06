from qdrant_client import QdrantClient, models
import json
import time
from pathlib import Path
from dotenv import load_dotenv

QUERIES_FILE = Path(__file__).parent / "dataset" / "queries_embeddings.json"

with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset_queries = json.load(file)

load_dotenv()

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

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


def evaluate_hnsw_ef(k_nearest, hnsw_ef_values, test_dataset):
    ground_truths = {}
    # For each embedding in test_dataset.values() (or each query, embedding pair via test_dataset.items()),
    # run the exact and the approximate searches.

    for query, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k_nearest,
            search_params=models.SearchParams (
                exact=True,
                quantization=models.QuantizationSearchParams (ignore=True)  # ignore is False by default
            ),
        ).points
        ground_truths[query] = set(item.id for item in knn_result)

    results = []
    warmup_embedding = next(iter(test_dataset.values()))

    for hnsw_ef in hnsw_ef_values:
        ann_times = {hnsw_ef: []}
        precisions = {hnsw_ef: []}

        # Warm up cache, with rescore value
        for _ in range (5):
            client.query_points (
                collection_name=COLLECTION_NAME,
                query=warmup_embedding,
                limit=k_nearest,
                search_params=models.SearchParams (
                    hnsw_ef=hnsw_ef,
                    quantization=models.QuantizationSearchParams (
                        rescore=False,
                        oversampling=2.0,
                    ),
                ),
            )
        for query, embedding in test_dataset.items():
            # Run approximate searches, with rescore value / Log the average time for a single query
            # Calculate precision / Append the logged times and the precision to their corresponding lists.

            start_time_ann = time.time()
            ann_result = client.query_points (
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k_nearest,
                search_params = models.SearchParams (
                    hnsw_ef=hnsw_ef,
                    quantization=models.QuantizationSearchParams (
                        rescore=False,
                        oversampling=2.0,
                    ),
                ),
            ).points
            ann_time = time.time() - start_time_ann
            ann_times[hnsw_ef].append(ann_time)

            ann_ids = set(item.id for item in ann_result)
            knn_ids = ground_truths[query]
            precision = len(ann_ids.intersection(knn_ids)) / k_nearest
            precisions[hnsw_ef].append(precision)

        avg_precision = sum (precisions[hnsw_ef]) / len (precisions[hnsw_ef])
        avg_query_time_ms = (sum (ann_times[hnsw_ef]) / len (ann_times[hnsw_ef])) * 1000  # sec -> ms

        results.append ({
            "hnsw_ef": hnsw_ef,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_query_time_ms,
        })

    for row in results:
        print (
            f"hnsw_ef={row['hnsw_ef']:>3} | "
            f"avg_precision={row['avg_precision']:.4f} | "
            f"avg_query_time={row['avg_query_time_ms']:.2f} ms"
        )


if __name__ == "__main__":
    evaluate_hnsw_ef(10, [10, 20, 50, 100, 200], test_dataset_queries)