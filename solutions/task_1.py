import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qdrant_eval.client import build_client
from qdrant_eval.config import K
from qdrant_eval.dataset import load_test_dataset
from qdrant_eval.formatting import result_formatting
from qdrant_eval.metrics import precision_at_k
from qdrant_eval.search import ann_search, knn_search, timed


def evaluate(test_dataset: dict[str, list[float]], k: int = K) -> dict[str, float]:
    client = build_client()
    precisions: list[float] = []
    ann_times: list[float] = []
    knn_times: list[float] = []

    for embedding in test_dataset.values():
        knn_points, knn_t = timed(knn_search, client, embedding, k)
        ann_points, ann_t = timed(ann_search, client, embedding, k)
        precisions.append(precision_at_k(ann_points, knn_points, k))
        ann_times.append(ann_t)
        knn_times.append(knn_t)

    return {
        "avg_precision": mean(precisions),
        "avg_ann_time": mean(ann_times),
        "avg_knn_time": mean(knn_times),
    }


if __name__ == "__main__":
    test_dataset = load_test_dataset()
    res = evaluate(test_dataset)
    result_formatting(
        K, res["avg_precision"], res["avg_ann_time"], res["avg_knn_time"]
    )
