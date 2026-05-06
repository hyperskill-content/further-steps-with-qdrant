from qdrant_client import models

from .config import K


def precision_at_k(
    ann_points: list[models.ScoredPoint],
    knn_points: list[models.ScoredPoint],
    k: int = K,
) -> float:
    ann_ids = {p.id for p in ann_points}
    knn_ids = {p.id for p in knn_points}
    return len(ann_ids & knn_ids) / k
