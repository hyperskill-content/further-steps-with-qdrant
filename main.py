import asyncio
import time
import json
from qdrant_client import QdrantClient, models
from typing import List, Dict, Any

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "./dataset/queries_embeddings.json"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
client.update_collection(
    collection_name=COLLECTION_NAME,
    optimizer_config=models.OptimizersConfigDiff(),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=False
        )
    )
)

def get_result(k: int, embedding: List[float], exact: bool, hnsw_ef: int | None, disable_quantized_search: bool,
               rescore: bool | None = None, oversampling: float | None = None) -> List[models.ScoredPoint]:
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        search_params=models.SearchParams(
            exact=exact,
            hnsw_ef=hnsw_ef,
            quantization=models.QuantizationSearchParams(
                ignore=disable_quantized_search,
                rescore=rescore,
                oversampling=oversampling
            )
        ),
        limit=k
    ).points


def get_ann_result(k: int, embedding: List[float], hnsw_ef: int, rescore: bool, oversampling: float) -> List[models.ScoredPoint]:
    return get_result(
        k=k,
        embedding=embedding,
        exact=False,
        hnsw_ef=hnsw_ef,
        disable_quantized_search=False,
        rescore=rescore,
        oversampling=oversampling
    )


def get_knn_result(k: int, embedding: List[float]) -> List[models.ScoredPoint]:
    return get_result(k=k, embedding=embedding, exact=True, hnsw_ef=None, disable_quantized_search=True)


def display_results(k: int, results: List[Dict[str, Any]]):
    for result in results:
        print(f"HNSW EF: {result['hnsw_ef']:3}\tAverage precision@{k}: {result['avg_precision']:.4f}\t"
              f"Average query time: {result['avg_query_time'] * 1000:.2f} ms")


def evaluate_hnsw_ef(k: int, hnsw_ef_values: List[int], test_dataset: Dict[str, Any]):
    rescore = False
    oversampling = 2.0

    results = []
    precisions = {}
    ann_times = {}

    for query, embedding in test_dataset.items():
        knn_result = get_knn_result(k, embedding)
        ground_truth = {query: (set(item.id for item in knn_result))}

        for hnsw_ef in hnsw_ef_values:
            start_time_ann = time.time()
            ann_result = get_ann_result(k, embedding, hnsw_ef, rescore, oversampling)
            ann_time = time.time() - start_time_ann
            ann_times.setdefault(hnsw_ef, []).append(ann_time)

            ann_ids = set(item.id for item in ann_result)
            precision = len(ann_ids.intersection(ground_truth[query])) / k
            precisions.setdefault(hnsw_ef, []).append(precision)

    for hnsw_ef in hnsw_ef_values:
        avg_ann_time = sum(ann_times[hnsw_ef]) / len(ann_times[hnsw_ef]) if ann_times[hnsw_ef] else 0.0
        avg_precision = sum(precisions[hnsw_ef]) / len(precisions[hnsw_ef]) if precisions[hnsw_ef] else 0.0
        results.append({
            "hnsw_ef": hnsw_ef,
            "avg_precision": avg_precision,
            "avg_query_time": avg_ann_time
        })

    display_results(k, results)


async def main():
    k = 10
    hnsw_ef_values = [10, 20, 50, 100, 200]

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset)


if __name__ == "__main__":
    asyncio.run(main())
