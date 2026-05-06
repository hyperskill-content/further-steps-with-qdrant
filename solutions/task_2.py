import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qdrant_client import models

from qdrant_eval.client import build_client
from qdrant_eval.config import K
from qdrant_eval.dataset import load_test_dataset
from qdrant_eval.formatting import format_hnsw_ef_results
from qdrant_eval.metrics import precision_from_ids
from qdrant_eval.search import ann_search, knn_search, timed

HNSW_EF_VALUES = [10, 20, 50, 100, 200]


def evaluate_hnsw_ef(
    k: int,
    hnsw_ef_values: list[int],
    test_dataset: dict[str, list[float]],
) -> list[dict]:
    client = build_client()

    ground_truth: dict[str, set] = {}
    for query, embedding in test_dataset.items():
        knn_points = knn_search(client, embedding, k)
        ground_truth[query] = {p.id for p in knn_points}

    results: list[dict] = []
    for ef in hnsw_ef_values:
        params = models.SearchParams(hnsw_ef=ef)
        precisions: list[float] = []
        times: list[float] = []
        for query, embedding in test_dataset.items():
            ann_points, elapsed = timed(ann_search, client, embedding, k, params)
            ann_ids = {p.id for p in ann_points}
            precisions.append(precision_from_ids(ann_ids, ground_truth[query], k))
            times.append(elapsed)
        results.append(
            {
                "hnsw_ef": ef,
                "avg_precision": mean(precisions),
                "avg_query_time_ms": mean(times) * 1000,
            }
        )
    return results


if __name__ == "__main__":
    test_dataset = load_test_dataset()
    results = evaluate_hnsw_ef(K, HNSW_EF_VALUES, test_dataset)
    format_hnsw_ef_results(results)
