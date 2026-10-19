import json
import time
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
K = 10
OVERSAMPLING = 2.0
QUANTILE = 0.99
QUERIES_FILE = Path(__file__).resolve().parent.parent / "dataset" / "queries_embeddings.json"
WARMUP_QUERIES = 10


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


def enable_scalar_quantization(client):
    """Apply int8 scalar quantization if this collection does not already have it.

    quantile=0.99 and always_ram=False are the values from the task. The
    original vectors stay; Qdrant builds the int8 copy in the background.
    """
    info = client.get_collection(COLLECTION_NAME)
    scalar = None if info.config.quantization_config is None else info.config.quantization_config.scalar
    already_applied = (
        scalar is not None
        and scalar.type == models.ScalarType.INT8
        and scalar.quantile == QUANTILE
        and scalar.always_ram is False
    )
    if already_applied and info.status == models.CollectionStatus.GREEN:
        print(f"status={info.status} optimizer={info.optimizer_status}", flush=True)
        return

    if not already_applied:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config=models.OptimizersConfigDiff(),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=QUANTILE,
                    always_ram=False,
                ),
            ),
        )

    # The config shows up immediately. The int8 vectors are built while the
    # collection is yellow, so wait until it has gone busy and come back green.
    seen_busy = False
    deadline = time.time() + 900
    while time.time() < deadline:
        info = client.get_collection(COLLECTION_NAME)
        print(f"status={info.status} optimizer={info.optimizer_status}", flush=True)
        quant = info.config.quantization_config
        config_ready = (
            quant is not None
            and quant.scalar is not None
            and quant.scalar.type == models.ScalarType.INT8
        )
        if info.status != models.CollectionStatus.GREEN:
            seen_busy = True
        if seen_busy and info.status == models.CollectionStatus.GREEN and config_ready:
            return
        time.sleep(3)
    raise TimeoutError("scalar quantization did not finish within 15 minutes")


def _ann_params(rescore):
    return models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            oversampling=OVERSAMPLING,
        )
    )


def _exact_params():
    return models.SearchParams(
        exact=True,
        quantization=models.QuantizationSearchParams(ignore=True),
    )


def evaluate_quantized_search(client, test_dataset, k, rescore):
    """Compare quantized ANN at this rescore setting with exact search on the original vectors."""
    warmup_embeddings = list(test_dataset.values())[:WARMUP_QUERIES]
    for embedding in warmup_embeddings:
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=_ann_params(rescore),
        )
    client.query_points(
        collection_name=COLLECTION_NAME,
        query=warmup_embeddings[0],
        limit=k,
        search_params=_exact_params(),
    )

    precisions = []
    ann_times = []
    knn_times = []
    for embedding in test_dataset.values():
        start_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=_ann_params(rescore),
        ).points
        ann_times.append(time.time() - start_ann)

        start_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=_exact_params(),
        ).points
        knn_times.append(time.time() - start_knn)

        ann_ids = {item.id for item in ann_result}
        knn_ids = {item.id for item in knn_result}
        precisions.append(len(ann_ids.intersection(knn_ids)) / k)

    n = len(precisions)
    result_formatting(
        k,
        sum(precisions) / n,
        sum(ann_times) / n,
        sum(knn_times) / n,
    )
    short = sum(1 for precision in precisions if precision < 1)
    print(f"queries_short_of_exact: {short} of {n}")
    print(
        "precision_counts: "
        + ", ".join(
            f"{precision:.1f}:{count}"
            for precision, count in sorted(Counter(precisions).items())
        )
    )


def main():
    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    enable_scalar_quantization(client)

    for rescore in (True, False):
        print(f"rescore={rescore} oversampling={OVERSAMPLING}")
        evaluate_quantized_search(client, test_dataset, k=K, rescore=rescore)


if __name__ == "__main__":
    main()
