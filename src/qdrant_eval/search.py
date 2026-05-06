import time
from typing import Any, Callable

from qdrant_client import QdrantClient, models

from .config import COLLECTION_NAME, K


def _with_exact(
    search_params: models.SearchParams | None, exact: bool = True
) -> models.SearchParams:
    if search_params is None:
        return models.SearchParams(exact=exact)
    update = {"exact": exact}
    if hasattr(search_params, "model_copy"):
        return search_params.model_copy(update=update)
    return search_params.copy(update=update)


def ann_search(
    client: QdrantClient,
    embedding: list[float],
    k: int = K,
    search_params: models.SearchParams | None = None,
) -> list[models.ScoredPoint]:
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=search_params,
    ).points


def knn_search(
    client: QdrantClient,
    embedding: list[float],
    k: int = K,
    search_params: models.SearchParams | None = None,
) -> list[models.ScoredPoint]:
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=_with_exact(search_params, exact=True),
    ).points


def timed(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start
