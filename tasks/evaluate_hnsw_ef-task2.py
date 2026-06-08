from qdrant_client import QdrantClient, models
import time
from typing import Any, List
import os
from dotenv import load_dotenv
import json

load_dotenv()

COLLECTION_NAME='arxiv_papers'
VECTOR_SIZE = 1536
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
k = 10

client = QdrantClient (host=QDRANT_HOST, port=QDRANT_PORT)


def load_query_embeddings(file_path: str | None) -> dict[str, Any]:
    """Load query embeddings from a regular JSON file.
    Expected format:
    {
        "query text": [embedding values],
        ...
    }
    """
    if not file_path:
        raise ValueError("FILE_PATH not set in .env or provided as argument.")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected queries file to contain a JSON object where keys are queries "
            "and values are embeddings."
        )

    return data

def ann_search(embedding) -> List[models.ScoredPoint]:
    """Perform approximate nearest neighbor search using Qdrant."""
    ann_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k
    ).points
    return ann_results

def knn_search(embedding) -> List[models.ScoredPoint]:
    """Perform exact k-nearest neighbor search using Qdrant."""
    # The exact k-NN search is done by adding exact=True to the search_params of the .query_points() method

    knn_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=models.SearchParams(exact=True),
    ).points
    return knn_results

def result_formatting(k_steps, avg_precision, avg_ann_time, avg_knn_time):
    print(f"Average precision@{k_steps}: {avg_precision:.4f}")
    print(f"Average ANN query time: {avg_ann_time * 1000:.2f} ms")
    print(f"Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms")

def precision_at_10(file_path: str | None):
    """Calculate precision at 10 for retrieval performance evaluation."""
    test_dataset = load_query_embeddings(file_path)

    precisions = []
    ann_times = []
    knn_times = []

    for query, embeddings in list(test_dataset.items())[:10]:
        if not isinstance(embeddings, list):
            raise ValueError(f"Embedding for query {query!r} is not a list.")
        start_time_ann = time.time()
        ann_result = ann_search(embeddings)
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)

        start_time_knn = time.time()
        knn_result = knn_search(embeddings)
        knn_time = time.time() - start_time_knn
        knn_times.append(knn_time)

        ann_ids = {item.id for item in ann_result}
        knn_ids = {item.id for item in knn_result}

        precision = len(ann_ids.intersection(knn_ids)) / k
        precisions.append(precision)

    if not precisions:
        raise ValueError("No query embeddings found in the dataset.")

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)

    result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)

def evaluate_hnsw_ef(k_steps: int, hnsw_ef_values: List[int], test_dataset: dict[str, Any]):
    """ TO DO """
    # Exact k-NN search
    # Extract the IDs of the retrieved points from the exact search and store them in the ground_truth dictionary

    # Approximate nearest neighbor search with different ef values
    # Extract the IDs of the retrieved points from the approximate search and store them in the ann_ids dictionary

    # Find the intersection of the approximate search IDs and ground truth IDs, and calculate the precision

    # Calculate the average precision for each ef value.
    # Calculate the average precision and the average query time.
    # Make a dictionary with these parameters and append them to a results list.
    pass

if __name__ == "__main__":
    FILE_PATH = "C:/Users/Frank/Documents/Backup/AI Sp3 data/queries_embeddings.json"
    path = FILE_PATH or os.getenv("FILE_PATH")
    print(f"Main> Path to use={path}")
    test_dataset_to_use = load_query_embeddings (FILE_PATH)
    EF_VALUES = [10, 20, 50, 100, 200]
    evaluate_hnsw_ef(10, EF_VALUES, test_dataset_to_use)
    precision_at_10(path)