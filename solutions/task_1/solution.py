"""Task 1 solution script."""

import json
import statistics
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
K = 10

"""
Usually __file__ based paths should be avoided.  The limited purpose of
this script means that it's a reasonable choice here, but it should be 
explained.

Python packages are often distributed as archives, often compressed (
zipped).  Inside an archive, __file__ based paths will usually fail.  The 
alternatives is usually something like importlib.resources()

Even if the project is not destined to be packaged, if a script is intended 
for reuse then passing the path in as a command line argument or using a 
configuration option such as configparser would be preferred.

However, this script has a very narrow purpose so the use of __file__ is 
unlikely to cause any problems in the future. The biggest risk is setting a 
poor example, hence this comment. 
"""
QUERIES_FILE = Path(__file__).resolve().parents[2] / "dataset" / "queries_embeddings.json"


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f"Average precision@{k}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")


def load_test_dataset(queries_file):
    with open(queries_file, "r", encoding="utf-8") as file:
        return json.load(file)


def query_ann(client, embedding, k):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
    ).points


def query_knn(client, embedding, k):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=models.SearchParams(exact=True),
    ).points


def calculate_precision(ann_result, knn_result, k):
    ann_ids = set(item.id for item in ann_result)
    knn_ids = set(item.id for item in knn_result)
    return len(ann_ids.intersection(knn_ids)) / k


def calculate_mismatches(ann_result, knn_result):
    ann_ids = set(item.id for item in ann_result)
    knn_ids = set(item.id for item in knn_result)
    return len(knn_ids - ann_ids)


def find_ann_limit_for_exact_coverage(client, embedding, k):
    knn_result = query_knn(client, embedding, k)
    knn_ids = set(item.id for item in knn_result)

    for ann_limit in range(k, (2 * k) + 1):
        ann_result = query_ann(client, embedding, ann_limit)
        ann_ids = set(item.id for item in ann_result)
        if knn_ids.issubset(ann_ids):
            return ann_limit

    return f"{2 * k}+"


def summarize_ann_coverage(client, test_dataset, k):
    coverage_limits = []
    failures = 0

    for embedding in test_dataset.values():
        coverage_limit = find_ann_limit_for_exact_coverage(client, embedding, k)
        if isinstance(coverage_limit, int):
            coverage_limits.append(coverage_limit)
        else:
            failures += 1

    return {
        "best_case": min(coverage_limits) if coverage_limits else None,
        "median_case": statistics.median(coverage_limits) if coverage_limits else None,
        "worst_case": max(coverage_limits) if coverage_limits else None,
        "failures": failures,
        "failure_label": f"{2 * k}+",
    }


def evaluate_search_quality(client, test_dataset, k):
    precisions = []
    ann_times = []
    knn_times = []
    ann_perf_times = []
    knn_perf_times = []
    total_mismatches = 0
    """
    time.time() works just fine in this context, but I would like to 
    highlight that Python provides a class specifically for performance 
    benchmarking tasks such as this:  time.perf_counter().  I have 
    demonstrated its use below.  Although the results are unlikely to be 
    significantly different in this context time.perf_counter() does have 
    the advantage of expressing intent. 
    """
    for embedding in test_dataset.values():
        start_time_ann = time.time()
        start_perf_ann = time.perf_counter()
        ann_result = query_ann(client, embedding, k)
        ann_time = time.time() - start_time_ann
        ann_perf_time = time.perf_counter() - start_perf_ann
        ann_times.append(ann_time)
        ann_perf_times.append(ann_perf_time)

        start_time_knn = time.time()
        start_perf_knn = time.perf_counter()
        knn_result = query_knn(client, embedding, k)
        knn_time = time.time() - start_time_knn
        knn_perf_time = time.perf_counter() - start_perf_knn
        knn_times.append(knn_time)
        knn_perf_times.append(knn_perf_time)

        precision = calculate_precision(ann_result, knn_result, k)
        precisions.append(precision)
        total_mismatches += calculate_mismatches(ann_result, knn_result)

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)
    avg_ann_perf_time = sum(ann_perf_times) / len(ann_perf_times)
    avg_knn_perf_time = sum(knn_perf_times) / len(knn_perf_times)
    total_requested_neighbors = len(test_dataset) * k
    return {
        "avg_precision": avg_precision,
        "avg_ann_time": avg_ann_time,
        "avg_knn_time": avg_knn_time,
        "avg_ann_perf_time": avg_ann_perf_time,
        "avg_knn_perf_time": avg_knn_perf_time,
        "total_requested_neighbors": total_requested_neighbors,
        "total_mismatches": total_mismatches,
    }


def print_timer_comparison(label, time_value, perf_value):
    absolute_difference = abs(perf_value - time_value)
    percent_difference = (
        (absolute_difference / time_value) * 100 if time_value else 0.0
    )
    print(f"{label} with time.time(): {time_value * 1000:.2f} ms")
    print(f"{label} with time.perf_counter(): {perf_value * 1000:.2f} ms")
    print(f"{label} absolute difference: {absolute_difference * 1000:.4f} ms")
    print(f"{label} timer percent difference: {percent_difference:.4f}%")


def print_search_comparison(ann_value, knn_value):
    absolute_difference = abs(knn_value - ann_value)
    percent_difference = (
        (absolute_difference / ann_value) * 100 if ann_value else 0.0
    )
    time_ratio = (knn_value / ann_value) if ann_value else 0.0
    print(f"ANN average: {ann_value * 1000:.2f} ms")
    print(f"Exact k-NN average: {knn_value * 1000:.2f} ms")
    print(f"Average difference: {absolute_difference * 1000:.2f} ms")
    print(f"Time ratio (exact k-NN / ANN): {time_ratio:.2f}x")
    print(f"Percent increase relative to ANN: {percent_difference:.2f}%")


def print_coverage_summary(coverage_summary):
    print("### ANN Coverage of Exact Top-K")
    print(
        "Smallest ANN limit needed to contain all exact top-K neighbors, "
        f"searched from K through {coverage_summary['failure_label'][:-1]}"
    )
    print(f"Best case: {coverage_summary['best_case']}")
    print(f"Median case: {coverage_summary['median_case']}")
    print(f"Worst case: {coverage_summary['worst_case']}")
    print(
        f"Queries not fully covered by ANN limit {coverage_summary['failure_label']}: "
        f"{coverage_summary['failures']}"
    )


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    test_dataset = load_test_dataset(QUERIES_FILE)
    results = evaluate_search_quality(client, test_dataset, K)
    coverage_summary = summarize_ann_coverage(client, test_dataset, K)
    result_formatting(
        K,
        results["avg_precision"],
        results["avg_ann_time"],
        results["avg_knn_time"],
    )
    print()
    print("### Error Summary")
    print(f"Total requested neighbors: {results['total_requested_neighbors']}")
    print(f"Total mismatched neighbors: {results['total_mismatches']}")
    print()
    print("### ANN vs Exact k-NN")
    print_search_comparison(results["avg_ann_time"], results["avg_knn_time"])
    print()
    print_coverage_summary(coverage_summary)
    print()
    print("### Timer Comparison")
    print_timer_comparison(
        "Average ANN query time",
        results["avg_ann_time"],
        results["avg_ann_perf_time"],
    )
    print()
    print_timer_comparison(
        "Average exact k-NN query time",
        results["avg_knn_time"],
        results["avg_knn_perf_time"],
    )


if __name__ == "__main__":
    main()
