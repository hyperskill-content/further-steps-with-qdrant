from typing import Hashable

from qdrant_client import models

from .config import K


def precision_from_ids(
    ann_ids: set[Hashable], knn_ids: set[Hashable], k: int = K
) -> float:
    return len(ann_ids & knn_ids) / k


def precision_at_k(
    ann_points: list[models.ScoredPoint],
    knn_points: list[models.ScoredPoint],
    k: int = K,
) -> float:
    return precision_from_ids({p.id for p in ann_points}, {p.id for p in knn_points}, k)
