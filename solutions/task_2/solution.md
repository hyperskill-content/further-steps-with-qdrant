### Observed values

```
[
  {
    "hnsw_ef": 10,
    "avg_precision": 0.9330000000000002,
    "avg_query_time_ms": 17.883565425872803,
    "total_absolute_error": 67,
    "max_absolute_error": 4
  },
  {
    "hnsw_ef": 20,
    "avg_precision": 0.9830000000000004,
    "avg_query_time_ms": 19.01235580444336,
    "total_absolute_error": 17,
    "max_absolute_error": 2
  },
  {
    "hnsw_ef": 50,
    "avg_precision": 0.9980000000000001,
    "avg_query_time_ms": 18.40559959411621,
    "total_absolute_error": 2,
    "max_absolute_error": 1
  },
  {
    "hnsw_ef": 100,
    "avg_precision": 0.9980000000000001,
    "avg_query_time_ms": 17.952051162719727,
    "total_absolute_error": 2,
    "max_absolute_error": 1
  },
  {
    "hnsw_ef": 200,
    "avg_precision": 0.9990000000000001,
    "avg_query_time_ms": 20.17603874206543,
    "total_absolute_error": 1,
    "max_absolute_error": 1
  }
]

### Absolute Error Summary
hnsw_ef=10
Total absolute error: 67
Max absolute error: 4

hnsw_ef=20
Total absolute error: 17
Max absolute error: 2

hnsw_ef=50
Total absolute error: 2
Max absolute error: 1

hnsw_ef=100
Total absolute error: 2
Max absolute error: 1

hnsw_ef=200
Total absolute error: 1
Max absolute error: 1

### Collection Defaults
Default hnsw_ef: 100
```

### Reflection

#### A technical summary of results

##### Accuracy results

The accuracy of Approximate Nearest Neighbor (ANN) search as compared
against exact k-nearest neighbor search (k-NN) tended to improve as the
exploration effort (hnsw_ef) was increased. It is worth noting that
hnsw_ef 50 produced the same accuracy as hnsw_ef 100, which is Qdrant's
default value. The highest hnsw_ef value tested (200) did provide some
improvement over this default, though.

##### Timing results

The timing results are highly ambiguous. While the lowest hnsw_ef value
was indeed faster than the highest hnsw_ef value, that trend doesn't hold
across the range of hnsw_ef values tested. A hnsw_ef 100 is faster than 50,
which is faster than 20.

#### Explaining the results

##### Explaining accuracy results

The difference in accuracy between the extremes of ef values tested is highly
visible. However, the returns of improved accuracy for increased effort 
diminish quickly. Exploration factors of 50 and 100 appear the same, and  
then 200 halves the remaining error. I think this is most likely because 
the test conditions don't stress the system enough to reveal such fine 
details, and less likely that there's an inflection point between 100 and 200.

##### Explaining timing results

The total spread of the average time results - the highest average minus
the lowest average - is only about 2.3 ms or about 11.4% of the highest
value. Such small differences in both absolute and proportional terms are
bound to be noisy. If I were really concerned with measuring them
accurately, I'd need to run the tests multiple times and average the
results. Ideally, I'd also run a larger set of test queries. I don't think
that's necessary here. The different hnsw_ef values are all still many
times faster than the exact search results from the previous task, and only
slightly different from each other. I already took one simple step to make
the results more reliable, which I will explain in the additional
explorations section below.

#### Explaining the trends, meaning explaining the theory

At a high level of abstraction, hnsw_ef changes how intensely the target
neighborhood of the HNSW graph is searched. This means that it mostly
applies to the base layer, after navigating the higher, sparser layers has
found the entrypoint to the most promising neighborhood.

How does hnsw_ef control this? What does the "intensity" of a search mean
in this context?

HNSW shares some concepts with other graph algorithms that can help us
understand. You might be familiar with Dijkstra's algorithm either
academically for its own sake or as a part of network routing protocols like
Open Shortest Path First (OSPF) which uses it.

Dijkstra's algorithm finds the lowest-cost path through a graph with
weighted edges from a given starting node to a destination node, or to
every node in the graph. In a physical example the edge weights are
usually distances, and the algorithm is finding the shortest distance path.  
But they could just as easily be travel times, and the algorithm would be
finding the quickest itinerary instead. In a computer network, weights
might be latency times or bandwidth costs.

To do this, Dijkstra's algorithm maintains a priority list of the best
candidate nodes it can reach, but that it hasn't explored yet. The list of
unexplored nodes starts with the entry node. Then it is removed to a list
of visited nodes, and all its immediate neighbors are added in order of
their edge costs to the root node from least to greatest. The most
promising node is visited next. This process repeats until the destination
node itself is visited or until the entire reachable graph has been explored.

Note that an unexplored node's position in the priority queue might change.
A node with multiple neighbors can be reached multiple ways from the
starting node. Visiting one of its neighbors might reveal a "shortcut" to
that node and update its position in the queue.

HNSW maintains data structures very similar to those maintained by
Dijkstra's algorithm: A list of visited nodes, a priority queue of the
best nodes it has found so far, and a priority queue of the most promising
nodes it might want to visit. The variable hnsw_ef controls the length of
the queue for best known nodes.

HNSW removes the best node from the list of candidates to visit and finds
the distance to the destination point (the query). If the candidate node
is closer than the furthest entry in the list of best known results, it
takes that place. If the most promising candidate node is further from the
destination than the worst entry in that list, the algorithm stops. The
process is very similar to Dijkstra's algorithm, but the
stopping conditions are different. Dijkstra's algorithm for finding a
specific destination stops once it explores that destination. It can know
that it has finished. HNSW does not have that option, since the
destination it is trying to approach isn't an element in the graph.  
Instead, it stops as soon as the results stop improving. This is just an
approximation, it could easily be trapped in a local minimum. This
approach might miss shortcuts if the beginning of the shorter path is too
far out of the way.

The longer the list of best nodes seen so far, set by hnsw_ef, the further
from the destination, the worst entry on that list will be. This gives each
new node a lower bar to meet to be an improvement and keep the algorithm
going.

#### Additional explorations

##### Explaining additional output

As in the first task, I added some additional output after the required output.
I put the error in absolute terms again, and I had the program print the
default value of hnsw_ef.

##### Warmup search

The first several times I ran the program, the lowest exploration factor
took the longest. I suspected that this was because it was first and was
due to the costs of setup. The easiest thing to do that might address this
was to run the tests twice using the same connection object and keep the second
results, so that is what my code does. As I explained earlier, this didn't
cause the timing results to have a clear trend. If I was really worried
about it, the next step would be to run the tests multiple times and average
the results.  