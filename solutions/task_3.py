import sys
import time
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qdrant_client import QdrantClient, models

from qdrant_eval.client import build_client
from qdrant_eval.config import COLLECTION_NAME, K
from qdrant_eval.dataset import load_test_dataset
from qdrant_eval.formatting import format_quantization_block
from qdrant_eval.metrics import precision_at_k
from qdrant_eval.search import ann_search, knn_search, timed

WARMUP_QUERIES = 10


def enable_scalar_quantization(client: QdrantClient) -> None:
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


def wait_until_green(client: QdrantClient, poll_s: float = 1.0, max_s: float = 600.0) -> None:
    deadline = time.time() + max_s
    while time.time() < deadline:
        info = client.get_collection(COLLECTION_NAME)
        if info.status == models.CollectionStatus.GREEN:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Collection did not reach GREEN within {max_s}s")


def compute_ground_truth(
    client: QdrantClient,
    embeddings: list[list[float]],
    k: int = K,
) -> tuple[list[list[models.ScoredPoint]], float]:
    # Bypass quantization so the baseline is full-precision exact k-NN.
    knn_params = models.SearchParams(
        quantization=models.QuantizationSearchParams(ignore=True)
    )
    truths: list[list[models.ScoredPoint]] = []
    times: list[float] = []
    for embedding in embeddings:
        points, t = timed(knn_search, client, embedding, k, knn_params)
        truths.append(points)
        times.append(t)
    return truths, mean(times)


def measure_ann(
    client: QdrantClient,
    embeddings: list[list[float]],
    ground_truth: list[list[models.ScoredPoint]],
    rescore: bool,
    oversampling: float = 2.0,
    k: int = K,
) -> dict[str, float]:
    ann_params = models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            oversampling=oversampling,
        )
    )

    # Warmup: page in the quantized segments and (for rescore=True) the
    # original vectors, so the first measurements aren't dominated by cold
    # mmap page-faults.
    for embedding in embeddings[:WARMUP_QUERIES]:
        ann_search(client, embedding, k, ann_params)

    precisions: list[float] = []
    ann_times: list[float] = []
    for embedding, truth in zip(embeddings, ground_truth):
        points, t = timed(ann_search, client, embedding, k, ann_params)
        precisions.append(precision_at_k(points, truth, k))
        ann_times.append(t)

    return {
        "avg_precision": mean(precisions),
        "avg_ann_time": mean(ann_times),
    }


if __name__ == "__main__":
    client = build_client()

    print(f"Enabling scalar int8 quantization on '{COLLECTION_NAME}' ...")
    enable_scalar_quantization(client)
    print("Waiting for collection to re-index (status=green) ...")
    wait_until_green(client)
    print("Collection ready.\n")

    test_dataset = load_test_dataset()
    embeddings = list(test_dataset.values())

    print("Computing exact k-NN ground truth (one pass, full-precision) ...")
    ground_truth, avg_knn_time = compute_ground_truth(client, embeddings)
    print(f"Ground truth ready. Avg exact k-NN time: {avg_knn_time * 1000:.2f} ms\n")

    print("Measuring ANN with rescore=True ...")
    res_rescore = measure_ann(client, embeddings, ground_truth, rescore=True)
    res_rescore["avg_knn_time"] = avg_knn_time

    print("Measuring ANN with rescore=False ...")
    res_no_rescore = measure_ann(client, embeddings, ground_truth, rescore=False)
    res_no_rescore["avg_knn_time"] = avg_knn_time

    print()
    format_quantization_block("rescore=True (oversampling=2.0)", K, res_rescore)
    print()
    format_quantization_block("rescore=False (oversampling=2.0)", K, res_no_rescore)
