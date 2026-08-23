import json
import time
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "C:\\Users\\gauth\\PycharmMisPcroject\\further-steps-with-qdrant\\dataset\\queries_embeddings.json"
K = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def apply_scalar_quantization():
     # enable Scalar Quantization (INT8)."""
    client.update_collection(
        collection_name=COLLECTION_NAME,
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=False,
            ),
        ),
    )


def run_quantization_experiment(test_dataset: Dict[str, List[float]]):
    # original vectors Ignore quantization
    ground_truth: Dict[str, set] = {}
    for query_text, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(
                exact=True,
                quantization=models.QuantizationSearchParams(ignore=True)
            ),
        ).points
        ground_truth[query_text] = set(item.id for item in knn_result)

    # 2. Warm up cache with throwaway queries
    warmup_embeddings = list(test_dataset.values())[:5]
    for emb in warmup_embeddings:
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=emb,
            limit=K
        )

    # two scenarios: with rescore=True and rescore=False
    experiments = [
        {"rescore": True, "oversampling": 2.0},
        {"rescore": False, "oversampling": 1.0}
    ]

    for exp in experiments:
        rescore_flag = exp["rescore"]
        oversampling_val = exp["oversampling"]

        precisions: List[float] = []
        elapsed_times: List[float] = []

        for query_text, embedding in test_dataset.items():
            start_time = time.time()

            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=K,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=rescore_flag,
                        oversampling=oversampling_val
                    )
                ),
            ).points

            elapsed_time_ms = (time.time() - start_time) * 1000
            elapsed_times.append(elapsed_time_ms)

            ann_ids = set(item.id for item in ann_result)
            gt_ids = ground_truth[query_text]

            precision = len(ann_ids.intersection(gt_ids)) / K
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)
        avg_query_time_ms = sum(elapsed_times) / len(elapsed_times)

        print(f"Rescore={rescore_flag} | Avg Precision@{K}: {avg_precision:.4f} | Avg Query Time: {avg_query_time_ms:.2f} ms")


if __name__ == "__main__":
    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    apply_scalar_quantization()
    run_quantization_experiment(test_dataset)