### Observed values

```
Average precision@10: 0.9980
Average ANN query time: 19.30 ms
Average exact k-NN query time: 64.50 ms

### Error Summary
Total requested neighbors: 1000
Total mismatched neighbors: 2

### ANN vs Exact k-NN
ANN average: 19.30 ms
Exact k-NN average: 64.50 ms
Average difference: 45.20 ms
Time ratio (exact k-NN / ANN): 3.34x
Percent increase relative to ANN: 234.14%

### ANN Coverage of Exact Top-K
Smallest ANN limit needed to contain all exact top-K neighbors, searched from K through 20
Best case: 10
Median case: 10.0
Worst case: 10
Queries not fully covered by ANN limit 20+: 2

### Timer Comparison
Average ANN query time with time.time(): 19.30 ms
Average ANN query time with time.perf_counter(): 19.28 ms
Average ANN query time absolute difference: 0.0271 ms
Average ANN query time timer percent difference: 0.1402%

Average exact k-NN query time with time.time(): 64.50 ms
Average exact k-NN query time with time.perf_counter(): 64.51 ms
Average exact k-NN query time absolute difference: 0.0136 ms
Average exact k-NN query time timer percent difference: 0.0211%
```

### Reflection

#### A technical summary of results

As you can see from the data above, the approximate nearest neighbor search
approach was 99.8% accurate. To put this in absolute terms, of the one
thousand recommendations requested in this experiment, ANN and exact nearest
neighbor search (k-NN) disagreed about two of them. At the same time, ANN
was about three times faster.

#### An accessible explanation of the results

The points stored in the database can be located anywhere in the embedding
space. Finding the exact nearest neighbors in a continuous space is a
significant geometry problem. Calculating, say, Euclidean distance
requires exponents and square roots, landing us squarely in the domain of
floating point math with all its pitfalls. We often apply specialized hardware
like
graphics cards to this type of work even when there are only three
dimensions and embeddings sometimes have thousands. Anything we can do to
minimize the amount of floating point geometry we have to do is going to be
a big help.

At a very high level, the Hierarchical Navigable Small World (HNSW)
algorithm adds some structure to the points in the embedding space by
treating them as nodes and adding edges between them. Additionally, HNSW
doesn't just make one graph. The base layer is the most
detailed, and HNSW creates layers of sparser graphs on top of it. These  
simpler graphs allow quick navigation across large expanses of the
lower, denser layers. If the base layer is a street map, then the higher
layers show connections between neighborhoods, boroughs, cities, counties,
and so on. The simpler, higher-level maps reduce the number of decisions
required to navigate. It's not a pure graph algorithm; HNSW is always
comparing distances. But finding the distance between a handful of states,
then counties, then cities, and so on is still vastly simpler than
performing a house-to-house search from the beginning.

Why is HNSW sometimes less accurate? Just like in real life, neighborhood
boundaries can be imperfect. The best burger place in your neighborhood 
might not be the best one closest to you. However, if the better option is 
across some railroad tracks or something else that represents a boundary 
to you, it will be harder to find even though it is physically closer.

#### Speedup and its benefits

For a database experiencing any sort of demand, a factor of 3 is a
transformative speedup. It changes how widely available the technology
can be both in terms of use cases and users.

This is especially important for uses like search because search is often
speculative. If searches take a long time to run or are otherwise limited,
then users won't search unless they're already confident they'll get good
results. But if a user only searches when they're confident they already
know the results, their searches won't be effective at bringing them new
information. Put another way, search is often the most useful when it is
the most surprising.

This puts search in contrast with, say, verification. Developing a
solution to the point where it can be formally validated is a significant
effort in its own right, and changes usually mean that it would need to be
validated again. There's not really much benefit to validating a solution
before it has been heavily invested in, so even if validation is
time-consuming and computationally expensive, it's not the bottleneck.

#### Accuracy

ANN is highly, but not perfectly accurate. Do the small disagreements
between ANN and k-NN represent any sort of reduction in quality or increase
in risk from the user's perspective? Let us consider some types of users and
their use cases to examine this.

##### Users who want exact results

People who want exact results and will be disappointed if they don't get
them do exist. If someone is looking for a text they only partially
remember, then even one match missed by ANN might be the one they were
searching for and hence the reason their search fails.

Functionally, users who need an exact match from partial and unreliable
data are not worse off with approximate semantic searches than they
otherwise would have been. Hoping to run into a person who can identify
what you're looking for and hoping to find a database that can do it are
both uncertain and inexact processes.

However, technological solutions can still increase risks. In this case, I
think the greatest risks come from false perceptions about how
comprehensive computer databases are and how authoritative search results
can be. It's widely intuitively understood that there is no person who has
read every book, or seen every movie, or heard every song. So if a person
tells us that they can't think of the text we're looking for, we are
not inclined to assume that means it doesn't exist. That same intuition
about technology is less widespread.

People might not understand that the results are approximate, especially
when the results are repeatable. If a database keeps producing the same
set of recommendations over and over, it's easy to interpret that as
information about what it does, and doesn't, contain.

It's also tempting to interpret the massive scale of some databases as a
sign that they're authoritative. There are, according to the marketing,
about one-hundred million songs on Spotify. If Spotify has so much but
still doesn't have what you're looking for, that can seem like evidence it
doesn't exist. However, even 100M songs are only a small fraction of all
music, and Spotify does not contain or seek to contain a representative
sample.

This is a problem that can only be solved by improving user understanding,
but it isn't always in people's best interests to draw attention to what
their collections don't contain or how their search functionality can fail.

##### Users who don't need exact results

In keeping with what I said earlier about search often being the most
useful when it is the most surprising, many search users don't
have an exact match in mind. Whether they're looking for information  
about a topic of interest, such as novel sorting algorithms, or doing
document archaeology to learn an organization's structure and history,
what they're looking for are entrypoints, not stopping points.

In these sorts of use cases, starting points that reference further
documents or help someone improve their understanding enough to run a more
specific search later are useful. A document is "more useful" if it
shortens the path to the information being sought or to a sharper question.
It would be a mistake to try and apply the numerical measurements we can
make about the accuracy of an approximate search to this user's needs.  
There's no reason to assume they'd even be able to identify which one of
the 10 approximate results wasn't in the exact results.

It's always possible to hope that the next source you read will be the best
one you've ever seen, and providing exact matches wouldn't change that. If
someone is really concerned that the best source is the one they haven't
read, then the solution is to read more. Either they'll read enough to
have confidence that their sources are good enough, or more likely, they'll
establish a broad enough base of knowledge that they no longer want or need
a single ideal source.

### Additional explorations

#### Increasing k

When I decided to say that the solution to the desire for the single best
recommendation was to read more, I realized that I should test if increasing
the number of recommendations requested from ANN would improve its accuracy.

ANN was already very accurate to begin with, but as you can see from the
data at the top of this report, that didn't work. Even doubling the number
of requested recommendations did not cause ANN to include any of the two
exact nearest neighbors it had missed before.

After some research I realized that this is an aspect of how the
approximate search works, and the algorithm has other parameters to adjust
that are more likely to change the accuracy.

#### Using time.perf_counter

I wanted to demonstrate another option that Python has for timing code
execution.  `time.perf_counter` has the advantage of being monotonic, while
time.time() is geared towards wall-clock time and technically could even
decrease. As the results show, the difference in timing is quite small.  
The main advantage in a case like this is that `perf_counter` expresses a more
specific intention than `time.time()`.