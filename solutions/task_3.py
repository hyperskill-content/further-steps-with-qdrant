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


def evaluate_quantization(
    test_dataset: dict[str, list[float]],
    rescore: bool,
    oversampling: float = 2.0,
    k: int = K,
) -> dict[str, float]:
    client = build_client()

    # Ground truth: bypass quantization so the baseline is full-precision exact k-NN.
    knn_params = models.SearchParams(
        quantization=models.QuantizationSearchParams(ignore=True)
    )
    # ANN: search over quantized vectors with the requested rescore behavior.
    ann_params = models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            oversampling=oversampling,
        )
    )

    precisions: list[float] = []
    ann_times: list[float] = []
    knn_times: list[float] = []

    for embedding in test_dataset.values():
        knn_points, knn_t = timed(knn_search, client, embedding, k, knn_params)
        ann_points, ann_t = timed(ann_search, client, embedding, k, ann_params)
        precisions.append(precision_at_k(ann_points, knn_points, k))
        ann_times.append(ann_t)
        knn_times.append(knn_t)

    return {
        "avg_precision": mean(precisions),
        "avg_ann_time": mean(ann_times),
        "avg_knn_time": mean(knn_times),
    }


if __name__ == "__main__":
    client = build_client()

    print(f"Enabling scalar int8 quantization on '{COLLECTION_NAME}' ...")
    enable_scalar_quantization(client)
    print("Waiting for collection to re-index (status=green) ...")
    wait_until_green(client)
    print("Collection ready. Running evaluation.\n")

    test_dataset = load_test_dataset()
    res_rescore = evaluate_quantization(test_dataset, rescore=True)
    res_no_rescore = evaluate_quantization(test_dataset, rescore=False)

    format_quantization_block("rescore=True (oversampling=2.0)", K, res_rescore)
    print()
    format_quantization_block("rescore=False (oversampling=2.0)", K, res_no_rescore)
