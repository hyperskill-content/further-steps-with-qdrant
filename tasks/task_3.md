# Quantization

Quantization is a way to reduce the precision of high-dimensional vectors to make the database more efficient in terms of storage and inference speed. It involves converting floating-point numbers to lower-precision types and trying to compress the vectors without losing too much accuracy.

## Theory

Qdrant supports 3 types of quantization out of the box (meaning that you don't have to preprocess the embeddings separately, and can quantize them inside of Qdrant itself, and also store the quantized vectors alongside the original ones without much configuration). Here are their brief descriptions:

Scalar quantization reduces the precision of vector elements to a set of predefined values. It's also known as uniform quantization and it maps floating-point numbers to integers like 8-bit. This type is perhaps the most universal one in terms of retaining high accuracy, but it can only speed up the search up to 2 times and reduce the storage by a factor of 4.

Product quantization divides a high-dimensional vector into multiple sub-vectors and quantizes each sub-vector independently. This is the most compressing type out of 3, but it might drop the accuracy significantly. You can read more about the inner workings of [product quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) in the Faiss book at Pinecone.

Binary quantization is a type of quantization where each vector component is represented by just 1 bit, either 0 or 1. This is more extreme than other forms of quantization, which might use 4 or 8 bits. Reducing each number to a single bit drastically reduces storage needs and speeds up the queries, but also induces a significant loss in accuracy because so much information is lost. Binary quantization does not work universally well with any [embedding model](https://qdrant.tech/documentation/guides/quantization/#binary-quantization) (although it does seem to work with the text-embedding-ada-002, which we are using, and Cohere embeddings), so binary quantization should be applied carefully and mostly to the tested models that can retain accuracy.

## Objectives

In this task, you will also modify the k-NN vs ANN accuracy and speed function from the previous tasks. Here, we ask you to update the collection with a quantization config and scalar quantization. Here is the code you can use:


```python
from qdrant_client import QdrantClient, models
# the initialization code here

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
```

Here, we quantize into `int8` type. `quantile=0.99` clips the values beyond the 99th percentile (the top 1% extreme values). This prevents outliers from skewing the quantization range.

`always_ram` controls where the *quantized* vectors live, independently of where the *original* vectors live. If a collection's original vectors are stored on disk (`on_disk=True`, done to save memory on a large dataset), `always_ram=True` still pins the small quantized vectors in RAM — so the cheap approximate part of the search runs entirely in memory, and only the final rescoring step has to touch disk for the original vectors. `always_ram=False` (used here) doesn't force this: quantized vectors are cached the same way as the rest of the segment, so they can be evicted and re-read from disk like anything else. The common production pattern is actually the opposite of this default: `always_ram=True` combined with original vectors on disk, since that's what gives you both a small memory footprint (only the quantized vectors are guaranteed resident) and fast search (RAM-resident candidate lookup, disk touched only for the reduced rescoring set).

Once you quantize the embeddings, your searches will be in the quantized embeddings by default. In case you wish to disable this behavior, use the following code (this snippet is relevant when computing the exact k-NN baseline against a quantized collection, since the exact search needs to run on real vectors to remain a meaningful baseline):

```python
result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=k,
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(ignore=True) # ignore is False by default
    ),
).points
```

`ignore=True` tells Qdrant to skip the quantized vectors for this search and use only the original, full-precision vectors — for both graph traversal and scoring. Once a collection has quantization configured, searches use the quantized vectors by default, so this is the override you reach for whenever you need a real, non-quantized search without removing the quantization config from the collection.

For the ANN search, you can use the following `search_params`:

```python
search_params = models.SearchParams(
    quantization=models.QuantizationSearchParams(
        rescore=True,
        oversampling=2.0,
    )
)
```

Quantized search here is a two-stage process. First, Qdrant uses the cheap quantized vectors to quickly retrieve `limit * oversampling` candidates — with `oversampling=2.0` and `limit=k=10`, that's 20 candidates. 
Then, if `rescore=True`, Qdrant recomputes the exact distance for each of those 20 candidates using the original, full-precision vectors, and returns the top 10 from that refined ranking, recovering most of the accuracy quantization would otherwise cost, at the price of extra reads against the original vectors. If `rescore=False`, that refinement is skipped and the top 10 are returned straight from the quantized scores — faster, but lower accuracy, since quantized similarity is a coarser ranking signal than exact similarity. `oversampling` controls how large a candidate pool the coarse quantized stage hands to rescoring: a bigger pool is more likely to contain the true top-k, at the cost of more work during rescoring.

Run the function two times: first, with `rescore = True`, and then, with `rescore = False`. Observe the results and reflect on their meaning. The final answer for the task should follow the format outlined in [the solution template](SOLUTION.md).


## Useful resources 

### Docs
1. [Quantization from Qdrant](https://qdrant.tech/documentation/manage-data/quantization/)     