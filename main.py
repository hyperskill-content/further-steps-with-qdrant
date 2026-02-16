import json
import logging
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
K = 10
DATASET = Path("./dataset/queries_embeddings.json")
RESCORE_SETTINGS = [True, False]

logger.info("Connecting to QDrant server")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def evaluate_quantizaion(k, rescore_settings, test_dataset):
    ground_truth = {}

    for query_text, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                exact=True,
                quantization=models.QuantizationSearchParams(ignore=True)
            ),
        ).points
        ground_truth[query_text] = set(item.id for item in knn_result)

    results = []

    for rescore in rescore_settings:
        logger.info(f"Testing rescore = {rescore}")
        ann_times = []
        precisions = []

        for query_text, embedding in test_dataset.items():
            start_time_ann = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        rescore=rescore,
                        oversampling=2.0,
                    )
                )
            ).points
            ann_time = time.time() - start_time_ann
            ann_times.append(ann_time)

            ann_ids = set(item.id for item in ann_result)
            truth_ids = ground_truth[query_text]

            precision = len(ann_ids.intersection(truth_ids)) / k
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)
        avg_time_ms = (sum(ann_times) / len(ann_times)) * 1000

        results.append({
            "rescore": rescore,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_time_ms
        })

    return results


def apply_quantization():
    logger.info("Applying Scalar Quantization (INT8)...")

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


def main():
    logger.info("Loading test dataset")
    with open(DATASET, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)

    apply_quantization()

    results = evaluate_quantizaion(K, RESCORE_SETTINGS, test_dataset)

    for res in results:
        print(f"rescore={res['rescore']}: average precision={res['avg_precision']:.4f}, query time={res['avg_query_time_ms']:.2f} ms")


if __name__ == "__main__":
    main()
