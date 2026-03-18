import json
import time
from qdrant_client import QdrantClient, models

# Constants
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
QUERIES_FILE = '../dataset/queries_embeddings.json'
K = 10

def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):
    """
    Evaluates the impact of different hnsw_ef values on search precision and time.
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ground_truth = {}
    
    print(f"Performing exact search (k={k}) for {len(test_dataset)} queries to establish ground truth...")
    # Perform exact search for every embedding to get ground truth
    for query_text, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        ground_truth[query_text] = set(item.id for item in knn_result)

    results = []

    for ef in hnsw_ef_values:
        print(f"Evaluating hnsw_ef = {ef}...")
        ef_times = []
        ef_precisions = []

        for query_text, embedding in test_dataset.items():
            start_time = time.time()
            ann_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=k,
                search_params=models.SearchParams(hnsw_ef=ef)
            ).points
            elapsed_ms = (time.time() - start_time) * 1000
            ef_times.append(elapsed_ms)

            ann_ids = set(item.id for item in ann_result)
            knn_ids = ground_truth[query_text]
            
            if k > 0:
                precision = len(ann_ids.intersection(knn_ids)) / k
            else:
                precision = 0
            ef_precisions.append(precision)

        avg_precision = sum(ef_precisions) / len(ef_precisions) if ef_precisions else 0
        avg_query_time = sum(ef_times) / len(ef_times) if ef_times else 0
        
        results.append({
            "hnsw_ef": ef,
            "avg_precision": avg_precision,
            "avg_query_time_ms": avg_query_time
        })

    return results

def main():
    # Load the test dataset
    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
            test_dataset = json.load(file)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {QUERIES_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {QUERIES_FILE}")
        return

    hnsw_ef_values = [10, 20, 50, 100, 200]
    
    results = evaluate_hnsw_ef(K, hnsw_ef_values, test_dataset)

    print("\nEvaluation Results:")
    print(json.dumps(results, indent=4))
    
    # Reflection
    print("\nReflection:")
    print("As hnsw_ef increases, the search explores more nodes in the HNSW graph.")
    print("This generally leads to higher precision at the cost of increased query execution time.")
    print("The results above demonstrate this trade-off between speed and accuracy.")

if __name__ == "__main__":
    main()
