import json
import time
from qdrant_client import QdrantClient, models

# Constants
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
QUERIES_FILE = '../dataset/queries_embeddings.json'
K = 10

def setup_quantization(client):
    """
    Updates the collection with scalar quantization config.
    """
    print(f"Updating collection '{COLLECTION_NAME}' with scalar quantization...")
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
    
    # Wait for the collection to be indexed/optimized if needed
    # (In a real scenario, we might wait for the status to become 'green')
    while True:
        info = client.get_collection(COLLECTION_NAME)
        if info.status == models.CollectionStatus.GREEN:
            break
        print(f"Waiting for collection to be ready (current status: {info.status})...")
        time.sleep(2)
    print("Collection is ready.")

def evaluate_quantization(client, test_dataset, rescore=True):
    """
    Evaluates the performance and precision of quantized search.
    """
    ground_truth = {}
    
    print(f"Establishing ground truth (k={K}) for {len(test_dataset)} queries (ignoring quantization)...")
    for query_id, embedding in test_dataset.items():
        # Using query_points with quantization ignored to get ground truth
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=models.SearchParams(
                exact=True,
                quantization=models.QuantizationSearchParams(ignore=True)
            ),
        ).points
        ground_truth[query_id] = set(item.id for item in knn_result)

    print(f"Evaluating quantized search with rescore={rescore}...")
    times = []
    precisions = []

    search_params = models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            oversampling=2.0 if rescore else None,
        )
    )

    for query_id, embedding in test_dataset.items():
        start_time = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=K,
            search_params=search_params
        ).points
        elapsed_ms = (time.time() - start_time) * 1000
        times.append(elapsed_ms)

        ann_ids = set(item.id for item in ann_result)
        knn_ids = ground_truth[query_id]
        
        precision = len(ann_ids.intersection(knn_ids)) / K if K > 0 else 0
        precisions.append(precision)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_query_time = sum(times) / len(times) if times else 0
    
    return {
        "rescore": rescore,
        "avg_precision": avg_precision,
        "avg_query_time_ms": avg_query_time
    }

def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Load test dataset
    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
            test_dataset = json.load(file)
            # Take only a subset if it's too large, but for now use all or first 100
            if len(test_dataset) > 100:
                test_dataset = dict(list(test_dataset.items())[:100])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Setup quantization
    setup_quantization(client)

    # Run evaluations
    results = []
    for rescore in [True, False]:
        res = evaluate_quantization(client, test_dataset, rescore=rescore)
        results.append(res)

    print("\nEvaluation Results (Quantization INT8):")
    print(json.dumps(results, indent=4))
    
    # Reflection
    print("\nReflection:")
    print("Scalar Quantization (INT8) reduces storage and can speed up search.")
    print("Rescoring (rescore=True) uses original vectors to re-rank the top candidates found via quantized search.")
    print("This usually improves precision at the cost of some additional time.")
    print("When rescore=False, the search relies solely on quantized vectors, which is faster but may be less accurate.")

if __name__ == "__main__":
    main()
