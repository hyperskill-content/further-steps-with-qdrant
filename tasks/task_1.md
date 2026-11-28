# Catching up
In this project, we will reflect on the optimizations that can be applied to an existing Qdrant database to balance search speed and the retrieval accuracy.

We assume you have successfully implemented the Vector database with Qdrant project and have an existing `arxiv_papers` collection to work with. If this is not the case, navigate to the first project and we will see you in a short time!

## Theory

By default, Qdrant uses the HNSW index for fast Approximate Nearest Neighbor (ANN) searches. This method returns results that are close to the actual nearest neighbors, but it is not exact. It is suitable for large datasets where doing an exact search would be expensive.

Exact k-Nearest Neighbor (k-NN) search involves searching the entire dataset to find the true nearest neighbors. While this approach is slower and is not suitable for production use, it provides precise results. The exact k-NN search serves as a baseline for evaluating the accuracy of ANN search, allowing for comparisons of the trade-offs between speed and precision.

In this task, we will evaluate the accuracy of the approximate search for our existing collection by calculating the average precision@k and compare the speed of the exact search and the approximate search with a [small test dataset](../dataset/queries_embeddings.json). This is only a single component of the search evaluation and the results should be interpreted with caution, but it does serve as a good sanity check and a way to understand whether the expectations match the current results.

Precision@k is a metric that measures the quality of the search results and focuses on the relevance of items retrieved during the ANN search, and is calculated as

$$\text{Precision@k}=\frac{∣\text{ANN results}∩\text{Exact results}|}{k}​$$

A value of 1 indicates that all approximate search results are identical to the exact search results. High precision means fewer irrelevant or false-positive items are included. k here corresponds to the number of retrieved items (we use 10). Precision measures the quality of the retrieved results by assessing the proportion of retrieved items that are actually relevant. It answers the question:

"Out of all the items the search retrieved, how many are true nearest neighbors?"

Note that precision@k and recall@k coincide here, since recall@k is

$$\text{Recall@k}=\frac{|\text{ANN results} \cap \text{Exact results}|}{|\text{Exact results}|}$$

and $|\text{Exact results}| = k$ by construction — the exact top-k is used as the golden set. This is specific to this setup: in typical IR evaluation, relevance is independently labeled and the relevant-set size has nothing to do with $k$, so the two metrics diverge.

This is also why the exercise generalizes: the exact k-NN result is a golden dataset, and checking precision@k against it is a standard eval for a retrieval system — the same pattern applies in production with a curated relevance-labeled set in place of a brute-force baseline.

Here is how you can perform the approximate (the default) search with the Qdrant client:

```python
from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

ann_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,
    limit=k
).points
```

The exact k-NN search is done by adding exact=True to the search_params of the .query_points() method:

```python
knn_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,
    limit=k,
    search_params=models.SearchParams(exact=True),
).points
```

## Objectives

The main goal of this task is to implement a function that calculates precision@10 and displays the average precision, average time for the approximate (default) search (with limit=10), and the average time for the exact search (with the same limit=10).

To calculate the average time, you can adapt the following snippet (we assume the embeddings are being iterated through, so this snippet will calculate the time for a single query):

```python
import time

knn_times = []

start_time_knn = time.perf_counter()
knn_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,
    limit=k,
    search_params=models.SearchParams(exact=True),
).points
knn_time = time.perf_counter() - start_time_knn
knn_times.append(knn_time)
```

**A note on timing:** use `time.perf_counter()` rather than `time.time()` to measure durations. `time.time()` reads the system clock, which can be adjusted while the script runs (e.g. by NTP), so a measured interval can come out too long, too short, or even negative. `time.perf_counter()` is monotonic and uses the highest-resolution clock available; its value is only meaningful as a difference between two calls, which is exactly how it is used here.

To print your results, you can utilize the following formatting function:

```python
def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')
```

Here is the series of steps to perform:

* Run the Qdrant Docker image (the same way as it was done in stage 1 of the Vector database with Qdrant project);

* Check that the collection is ready before measuring anything:

```python
info = client.get_collection(COLLECTION_NAME)
print(info.status, info.points_count, info.indexed_vectors_count)
print(info.config.quantization_config)
```

The status should be `green`, `indexed_vectors_count` should match (or be very close to) `points_count`, and `quantization_config` should be `None`. Note:

**A small collection may have no HNSW index at all.** Qdrant only builds an HNSW graph for a segment once its vectors exceed the optimizer's `indexing_threshold` (in KB, see `info.config.optimizer_config.indexing_threshold`). Smaller segments are searched by brute force, so the "approximate" search is actually exact and the comparison in this task measures nothing. The full `arxiv_papers` collection (~411k points) is well above the threshold; a truncated one with a few thousand points is not, and will show `indexed_vectors_count` far below `points_count`.

* Download the [test dataset](../dataset/queries_embeddings.json) that will be used for the calculation. It is a JSON file with 100 queries and their corresponding text-embedding-ada-002 embeddings. The queries are stored as the keys, and the embeddings are stored as the values.

* Load the dataset with the script. This can be done as follows:

```python
import json 

with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)
```

* Before timing, run a few throwaway queries (both approximate and exact) so the first timed queries don't absorb the one-time cost of opening the connection and loading data into the cache. Pass `with_payload=False` to `.query_points()`: you only need the IDs, and returning the payload adds the same transfer cost to both searches, which hides part of the difference you are trying to measure.

* For each embedding in `test_dataset.values()` (or each `query, embedding` pair via `test_dataset.items()`), run the exact and the approximate searches, calculate the precision, and log the time for a single query. The precision of a single query can be calculated as

```python
ann_ids = set(item.id for item in ann_result)
knn_ids = set(item.id for item in knn_result)  
precision = len(ann_ids.intersection(knn_ids)) / k
```

* Append the logged times and the precision to their corresponding lists.

* Calculate the averages of the obtained logged time and the precision.

* Display the averages and write up a reflection on the obtained results. A written reflection is a required part of your submission. The final answer for the task should follow the format outlined in [the solution template](SOLUTION.md).

## Useful resources 

### Docs
1. [Precision and recall at K in ranking and recommendations from Evidently](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k)     
2. [Hierarchical Navigable Small Worlds (HNSW) from Pinecone](https://www.pinecone.io/learn/series/faiss/hnsw/)    
3. [Understanding the approximate nearest neighbor (ANN) algorithm from Elastic](https://www.elastic.co/blog/understanding-ann)           
4. [kNN search from Elastic documentation](https://www.elastic.co/docs/solutions/search/vector/knn)             
5. [exact = True in models.SearchParams()](https://qdrant.tech/documentation/guides/optimize/?q=exact%3Dtrue#fine-tuning-search-parameters)