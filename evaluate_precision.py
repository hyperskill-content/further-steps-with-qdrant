import argparse
import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "dataset/queries_embeddings.json"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


def evaluate_precision(test_dataset, k, ef):
    ann_times = []
    knn_times = []
    precisions = []

    for embedding in test_dataset.values():
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(hnsw_ef=ef),
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)

        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_time = time.time() - start_time_knn
        knn_times.append(knn_time)

        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / k
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)

    result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)


def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):
    ground_truth = {}

    for query, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points

        ground_truth[query] = set(item.id for item in knn_result)

    warmup_embeddings = list(test_dataset.values())[:5]
    for embedding in warmup_embeddings:
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(hnsw_ef=hnsw_ef_values[0]),
        ).points

    results = []

    for hnsw_ef in hnsw_ef_values:
        query_times = []
        precisions = []

        for query, embedding in test_dataset.items():
            start_time = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            ).points
            query_time = time.time() - start_time
            query_times.append(query_time)

            ann_ids = set(item.id for item in ann_result)
            exact_ids = ground_truth[query]
            precision = len(ann_ids.intersection(exact_ids)) / k
            precisions.append(precision)

        avg_precision = sum(precisions) / len(precisions)
        avg_query_time_ms = (sum(query_times) / len(query_times)) * 1000

        results.append(
            {
                "hnsw_ef": hnsw_ef,
                "avg_precision": avg_precision,
                "avg_query_time_ms": avg_query_time_ms,
            }
        )

    for result in results:
        print(
            f"hnsw_ef={result['hnsw_ef']}: "
            f"avg_precision={result['avg_precision']:.4f}, "
            f"avg_query_time_ms={result['avg_query_time_ms']:.2f}"
        )

    return results


def enable_scalar_quantization():
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


def evaluate_quantization(k, test_dataset, rescore):
    ground_truth = {}
    exact_times = []

    quantization_search_params = (
        models.QuantizationSearchParams(
            rescore=True,
            oversampling=2.0,
        )
        if rescore
        else models.QuantizationSearchParams(
            rescore=False,
        )
    )

    for query, embedding in test_dataset.items():
        start_time = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                exact=True,
                quantization=models.QuantizationSearchParams(ignore=True),
            ),
        ).points
        exact_time = time.time() - start_time
        exact_times.append(exact_time)

        ground_truth[query] = set(item.id for item in knn_result)

    warmup_embeddings = list(test_dataset.values())[:5]
    for embedding in warmup_embeddings:
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                quantization=quantization_search_params,
            ),
        ).points

    ann_times = []
    precisions = []

    for query, embedding in test_dataset.items():
        start_time = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(
                quantization=quantization_search_params,
            ),
        ).points
        ann_time = time.time() - start_time
        ann_times.append(ann_time)

        ann_ids = set(item.id for item in ann_result)
        exact_ids = ground_truth[query]
        precision = len(ann_ids.intersection(exact_ids)) / k
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time_ms = (sum(ann_times) / len(ann_times)) * 1000
    avg_exact_time_ms = (sum(exact_times) / len(exact_times)) * 1000

    print(
        f"quantized_search rescore={rescore}: "
        f"avg_precision={avg_precision:.4f}, "
        f"avg_query_time_ms={avg_ann_time_ms:.2f}, "
        f"avg_exact_query_time_ms={avg_exact_time_ms:.2f}"
    )

    return {
        "rescore": rescore,
        "avg_precision": avg_precision,
        "avg_query_time_ms": avg_ann_time_ms,
        "avg_exact_query_time_ms": avg_exact_time_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qdrant ANN precision.")
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve (default: 10)"
    )

    parser.add_argument(
        "-ef",
        type=int,
        default=128,
        help="hnsw_ef (default: 128)"
    )

    parser.add_argument(
        "--test-hnsw-ef",
        action="store_true",
        help="Run hnsw_ef sweep for Task 2"
    )

    parser.add_argument(
        "--enable-quantization",
        action="store_true",
        help="Update the collection with scalar int8 quantization"
    )

    parser.add_argument(
        "--test-quantization",
        action="store_true",
        help="Run quantized search evaluation for Task 3"
    )

    args = parser.parse_args()

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    if args.enable_quantization:
        enable_scalar_quantization()

    if args.test_hnsw_ef:
        evaluate_hnsw_ef(
            k=args.k,
            hnsw_ef_values=[10, 20, 50, 100, 200],
            test_dataset=test_dataset,
        )
    elif args.test_quantization:
        evaluate_quantization(k=args.k, test_dataset=test_dataset, rescore=True)
        evaluate_quantization(k=args.k, test_dataset=test_dataset, rescore=False)
    else:
        evaluate_precision(test_dataset, k=args.k, ef=args.ef)