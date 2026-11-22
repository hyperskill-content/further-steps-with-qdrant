"""
Task 3 / Stage 3 — Quantization

Enable scalar INT8 quantization on arxiv_papers, then compare ANN search with:
  - rescore=True  (quantize → oversample → rescore with full vectors)
  - rescore=False (quantize scores only)

Exact k-NN ground truth uses QuantizationSearchParams(ignore=True) so the
baseline stays on full-precision vectors.

Run (Qdrant up + arxiv_papers loaded):
  poetry -C ../vector-db-qdrant run python \\
    /Users/upinder.dhami/workday/src/ai-course/further-steps-with-qdrant/task_3.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
K = 10
QUERIES_FILE = Path(__file__).resolve().parent / "dataset" / "queries_embeddings.json"
WARMUP_QUERIES = 5


def _status_is_green(status) -> bool:
    return str(status).lower().endswith("green")


def enable_scalar_quantization(client: QdrantClient) -> None:
    """Configure INT8 scalar quantization and wait until the collection is green."""
    print("Updating collection with ScalarQuantization (INT8, quantile=0.99) ...")
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
    # Quantization is applied asynchronously; yellow means still building.
    for _ in range(120):
        info = client.get_collection(COLLECTION_NAME)
        print(f"  collection status={info.status}, points={info.points_count}")
        if _status_is_green(info.status):
            return
        time.sleep(5)
    raise RuntimeError("Timed out waiting for collection to finish quantizing")


def build_ground_truth(
    client: QdrantClient,
    test_dataset: dict,
    k: int = K,
) -> dict[str, set]:
    """Exact top-k using full-precision vectors (ignore quantized)."""
    ground_truth: dict[str, set] = {}
    for i, (query, embedding) in enumerate(test_dataset.items(), start=1):
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                exact=True,
                quantization=models.QuantizationSearchParams(ignore=True),
            ),
        ).points
        ground_truth[query] = {item.id for item in knn_result}
        if i % 20 == 0 or i == 1:
            print(f"  ground truth {i}/{len(test_dataset)} ...")
    return ground_truth


def warm_cache(client: QdrantClient, test_dataset: dict) -> None:
    print(f"Warming cache with {WARMUP_QUERIES} throwaway queries ...")
    for embedding in list(test_dataset.values())[:WARMUP_QUERIES]:
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=2.0,
                )
            ),
        )


def evaluate_quantized_search(
    client: QdrantClient,
    test_dataset: dict,
    ground_truth: dict[str, set],
    *,
    rescore: bool,
    k: int = K,
) -> dict:
    """Average precision@k and query time for quantized ANN with/without rescore."""
    precisions: list[float] = []
    times: list[float] = []

    search_params = models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            oversampling=2.0,
        )
    )

    for query, embedding in test_dataset.items():
        start = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=search_params,
        ).points
        times.append(time.time() - start)

        ann_ids = {item.id for item in ann_result}
        precisions.append(len(ann_ids.intersection(ground_truth[query])) / k)

    timed = times[1:] if len(times) > 1 else times
    return {
        "rescore": rescore,
        "avg_precision": sum(precisions) / len(precisions),
        "avg_query_time_ms": (sum(timed) / len(timed)) * 1000,
    }


def main() -> None:
    print("Task 3 — Quantization (scalar INT8, rescore on/off)")
    with open(QUERIES_FILE, encoding="utf-8") as file:
        test_dataset = json.load(file)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    enable_scalar_quantization(client)

    print("Building exact ground truth (ignore quantization) ...")
    ground_truth = build_ground_truth(client, test_dataset, k=K)
    warm_cache(client, test_dataset)

    print("Evaluating rescore=True ...")
    with_rescore = evaluate_quantized_search(
        client, test_dataset, ground_truth, rescore=True
    )
    print(
        f"  → precision={with_rescore['avg_precision']:.4f}, "
        f"avg_query_time_ms={with_rescore['avg_query_time_ms']:.3f}"
    )

    print("Evaluating rescore=False ...")
    without_rescore = evaluate_quantized_search(
        client, test_dataset, ground_truth, rescore=False
    )
    print(
        f"  → precision={without_rescore['avg_precision']:.4f}, "
        f"avg_query_time_ms={without_rescore['avg_query_time_ms']:.3f}"
    )

    results = [with_rescore, without_rescore]
    print()
    print("results =")
    print(results)


if __name__ == "__main__":
    main()
