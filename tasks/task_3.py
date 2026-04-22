import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "../dataset/queries_embeddings.json"
k = 10

# ---
# same timeout bump as task 2 — exact search on 411k vectors is slow
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)


def apply_scalar_quantization():
    # quantize into int8 — clips top 1% outliers to prevent skew,
    # lets the quantized vectors live on disk rather than always in RAM
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
    print("Quantization applied.")


def evaluate_quantization(k, test_dataset):
    # step 1: get exact k-NN ground truth with quantization ignored
    ground_truth = {}
    for query, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(ignore=True)
            ),
        ).points
        ground_truth[query] = set(item.id for item in knn_result)

    # step 2: run quantized search with rescore=True and rescore=False
    for rescore in [True, False]:
        precisions = []
        times = []

        for query, embedding in test_dataset.items():
            start = time.time()
            result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        rescore=rescore,
                        oversampling=2.0,
                    )
                ),
            ).points
            times.append(time.time() - start)

            result_ids = set(item.id for item in result)
            precision = len(result_ids.intersection(ground_truth[query])) / k
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)
        avg_time_ms = (sum(times) / len(times)) * 1000

        print(f"rescore={rescore} | precision@{k}: {avg_precision:.4f} | avg time: {avg_time_ms:.2f} ms")


if __name__ == "__main__":
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)

    apply_scalar_quantization()
    evaluate_quantization(k, test_dataset)
