\# V4 — Larger Corpus and Single-Machine Scalability



\## Overview



V4 extends the project from RAG-quality evaluation to \*\*retrieval-system scalability\*\*.



V1–V3 used the relatively small SciFact corpus containing 5,183 documents. That scale was sufficient for developing semantic retrieval, local RAG, and groundedness experiments, but it was too small to meaningfully investigate systems scalability.



V4 therefore introduces the much larger \*\*MS MARCO\*\* corpus and asks:



> \*\*How does the current single-machine semantic retrieval architecture behave as corpus size increases, and where do scalability bottlenecks emerge?\*\*



The experiment progressively increases corpus size while holding the evaluation queries, relevant documents, embedding model, vector-index architecture, Top-K, and hardware environment fixed.



\---



\# Research Questions



V4 investigates the following questions:



1\. How does document embedding time scale with corpus size?

2\. Does embedding throughput remain constant during increasingly large workloads?

3\. How does FAISS index size scale with corpus size?

4\. How does exact vector-search latency change as the search space grows?

5\. How does process memory usage change with corpus size?

6\. Does retrieval effectiveness degrade as more distractor documents enter the corpus?

7\. Which component becomes the dominant single-machine scalability bottleneck?

8\. At what point is distributed processing justified?



\---



\# Dataset



\## MS MARCO



V4 uses the BEIR version of \*\*MS MARCO\*\*.



Source datasets:



```text

BeIR/msmarco

BeIR/msmarco-qrels

```



The downloaded corpus contains:



```text

8,841,823 passages

```



and:



```text

509,962 queries

```



The corpus schema is:



```text

\_id

title

text

```



The qrel schema is:



```text

query-id

corpus-id

score

```



\---



\# Qrel Split Selection



Both the validation and test qrel splits were inspected before constructing the V4 benchmark.



\## Validation Split



```text

Total qrel rows:       7,437

Positive qrel rows:    7,437

Positive queries:      6,980

Unique positive docs:  7,433



Relevant documents/query:

&#x20;   min:     1

&#x20;   median:  1

&#x20;   mean:    1.07

&#x20;   max:     4

```



\## Test Split



```text

Total qrel rows:       9,260

Positive qrel rows:    4,102

Positive queries:      43

Unique positive docs:  4,102



Relevant documents/query:

&#x20;   min:      4

&#x20;   median:  75

&#x20;   mean:     95.40

&#x20;   max:      341

```



The validation split was selected because it provides a substantially larger query population and sparse relevance judgments that are better suited to the controlled retrieval-scaling experiment.



\---



\# Controlled Nested Benchmark



V4 uses a fixed sample of:



```text

300 evaluation queries

```



selected deterministically from positively judged MS MARCO validation queries using:



```text

random seed = 42

```



All known positively relevant documents for these queries are included in every experimental corpus.



Additional documents are deterministic distractors sampled from the full MS MARCO corpus.



The resulting corpora are \*\*nested\*\*:



```text

S\_1K

&#x20; ⊂

S\_25K

&#x20; ⊂

S\_50K

&#x20; ⊂

S\_100K

&#x20; ⊂

S\_250K

```



Therefore:



```text

Same 300 queries

Same gold documents

Same embedding model

Same vector index

Same Top-K

Same hardware

Same random sampling policy



&#x20;       ↓



Only corpus size changes

```



This design allows changes in retrieval behavior to be interpreted primarily as effects of increasing the number of competing distractor documents.



\---



\# Experimental Scale



The measured corpus sizes are:



| Dataset | Documents | Purpose |

|---|---:|---|

| MS MARCO 1K | 1,000 | Development / smoke-test baseline |

| MS MARCO 25K | 25,000 | Scaling experiment |

| MS MARCO 50K | 50,000 | Scaling experiment |

| MS MARCO 100K | 100,000 | Scaling experiment |

| MS MARCO 250K | 250,000 | Large single-machine validation |



The full 8.84-million-passage corpus is retained as the source population for future experiments.



\---



\# Retrieval Configuration



The retrieval architecture remains intentionally simple so that corpus-size effects can be isolated.



\## Embedding Model



```text

sentence-transformers/all-MiniLM-L6-v2

```



Embedding dimension:



```text

384

```



Embeddings are normalized before indexing.



\---



\## Vector Index



FAISS:



```text

IndexFlatIP

```



This provides exact inner-product search over normalized embeddings.



Because normalized embeddings are used, inner-product ranking corresponds to cosine-similarity ranking.



\---



\## Retrieval Depth



The baseline retrieval depth is fixed at:



```text

Top-K = 5

```



V3 identified K=5 as the strongest observed balance for the previous RAG experiments.



In V4, K is kept fixed so that corpus size remains the primary independent variable.



\---



\# Evaluation Metrics



For every corpus size, V4 records:



\## Offline Processing



\- corpus loading time

\- embedding-model loading time

\- document embedding time

\- document embedding throughput

\- FAISS index construction time

\- FAISS index size



\## Online Retrieval



\- query embedding time

\- total retrieval time

\- average query latency

\- retrieval throughput



\## Retrieval Quality



\- Recall@5



\## Resource Usage



\- maximum observed process RSS



The memory metric is sampled at major pipeline stages and should therefore be interpreted as:



```text

maximum observed RSS

```



rather than a continuously monitored true peak-memory measurement.



\---



\# Measured V4 Results



| Documents | Recall@5 | Embedding Time | Throughput | Index Build | Index Size | Query Latency | Query Throughput | Max Observed RSS |

|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| 1,000 | 0.9967 | 27.06 s | 36.96 docs/s | 0.0008 s | 1.46 MB | 0.027 ms | 37,017 q/s | 611 MB |

| 25,000 | 0.9800 | 825.46 s | 30.29 docs/s | 0.0228 s | 36.62 MB | 1.157 ms | 864 q/s | 768 MB |

| 50,000 | 0.9533 | 1,629.76 s | 30.68 docs/s | 0.0292 s | 73.24 MB | 2.569 ms | 389 q/s | 887 MB |

| 100,000 | 0.9400 | 3,171.49 s | 31.53 docs/s | 0.0687 s | 146.48 MB | 5.142 ms | 194 q/s | 1,235 MB |

| \*\*250,000\*\* | \*\*0.8833\*\* | \*\*10,695.13 s\*\* | \*\*23.38 docs/s\*\* | \*\*0.1399 s\*\* | \*\*366.21 MB\*\* | \*\*14.337 ms\*\* | \*\*69.75 q/s\*\* | \*\*1,729 MB\*\* |



The complete 250K pipeline required:



```text

10,706.18 seconds

```



or approximately:



```text

2.97 hours

```



Document embedding alone required:



```text

10,695.13 seconds

```



and therefore accounted for nearly all measured pipeline execution time.



\---



\# Result 1 — Document Embedding Dominates Offline Cost



At 250K documents:



```text

Document embedding:

10,695.13 seconds



FAISS index construction:

0.1399 seconds

```



The embedding stage therefore dominates the offline pipeline by several orders of magnitude.



Even at 100K documents:



```text

Embedding:

3,171.49 seconds



Index construction:

0.0687 seconds

```



The vector-index construction itself is negligible compared with generating document embeddings.



This identifies \*\*embedding generation as the primary single-machine preprocessing bottleneck\*\* in the current architecture.



\---



\# Result 2 — Embedding Cost Initially Scales Approximately Linearly



Measured sustained embedding throughput was:



```text

1K       36.96 docs/s

25K      30.29 docs/s

50K      30.68 docs/s

100K     31.53 docs/s

```



Between 25K and 100K, sustained throughput remained close to:



```text

30–32 documents/second

```



and embedding time increased approximately in proportion to corpus size.



This initially suggested that embedding cost could be approximated with a simple linear scaling model.



\---



\# Result 3 — Throughput Degrades at 250K



At 250K documents, measured embedding throughput fell to:



```text

23.38 documents/second

```



compared with:



```text

31.53 documents/second

```



at 100K.



This represents an approximate reduction of:



```text

25.9%

```



relative to the 100K throughput.



The experiment does not establish the cause of this degradation. Possible factors include sustained CPU behavior, thermal effects, document-length characteristics, memory behavior, or unrelated system activity.



The appropriate conclusion is therefore limited to the measured observation:



> \*\*Single-machine embedding throughput degraded at the largest tested workload.\*\*



\---



\# Result 4 — Pre-250K Projection Underestimated Embedding Time



Before running the 250K experiment, a linear model fitted to the 1K–100K measurements projected:



```text

250K embedding time ≈ 2.21 hours

```



The measured result was:



```text

250K embedding time ≈ 2.97 hours

```



The actual workload therefore required approximately:



```text

34% more time

```



than the earlier linear projection.



This provides evidence that extrapolation from smaller workloads can underestimate large single-machine embedding costs.



\---



\# Result 5 — FAISS Index Size Scales Predictably



Measured index sizes were:



```text

1K       1.46 MB

25K     36.62 MB

50K     73.24 MB

100K   146.48 MB

250K   366.21 MB

```



The 250K pre-run projection was:



```text

366.21 MB

```



which closely matched the measured result.



For a flat index of fixed-size 384-dimensional float32 vectors, storage therefore scales very predictably with the number of indexed documents.



\---



\# Result 6 — Exact Retrieval Latency Increases with Corpus Size



Average query latency increased from:



```text

1K       0.027 ms

25K      1.157 ms

50K      2.569 ms

100K     5.142 ms

250K    14.337 ms

```



The pre-250K linear projection was approximately:



```text

12.95 ms

```



while the measured value was:



```text

14.34 ms

```



Exact `IndexFlatIP` search therefore becomes progressively more expensive as the corpus grows.



However, retrieval remains relatively inexpensive compared with document embedding.



At 250K:



```text

300-query retrieval:

4.30 seconds



document embedding:

10,695.13 seconds

```



Document embedding required approximately 2,500 times as much wall-clock time as retrieving all 300 evaluation queries.



\---



\# Result 7 — Retrieval Quality Degrades as Distractors Increase



Recall@5 declined as corpus size increased:



```text

1K       0.9967

25K      0.9800

50K      0.9533

100K     0.9400

250K     0.8833

```



The same 300 evaluation queries and their gold documents were preserved in every corpus.



Therefore, the principal experimental change is the addition of increasingly many distractor documents.



From 1K to 250K:



```text

Recall@5:

0.9967 → 0.8833

```



a decrease of:



```text

0.1134

```



or approximately:



```text

11.34 percentage points

```



At 1K, approximately 299 of the 300 evaluation queries retrieved at least one gold document in the Top-5.



At 250K, approximately 265 queries did so.



This demonstrates that retrieval effectiveness can decline as the search space grows even when the embedding model, queries, gold evidence, index type, and Top-K remain unchanged.



\---



\# Result 8 — Memory Is Not Yet the Primary Bottleneck



Maximum observed RSS increased as follows:



```text

1K       611 MB

25K      768 MB

50K      887 MB

100K   1,235 MB

250K   1,729 MB

```



At 250K documents, observed process memory remained below 2 GB under the sampled measurement methodology.



On the 16 GB development machine, memory had therefore not yet become the dominant limitation.



The primary measured constraint was computation time for document embedding.



\---



\# Pre-250K Projection Validation



Before executing the 250K experiment, the 1K–100K measurements were used to generate exploratory linear projections.



| Metric | Pre-250K Projection | 250K Measurement |

|---|---:|---:|

| Embedding time | \~2.21 h | \*\*\~2.97 h\*\* |

| Index size | \~366.21 MB | \*\*366.21 MB\*\* |

| Query latency | \~12.95 ms | \*\*14.34 ms\*\* |

| Max observed RSS | \~2,162 MB | \*\*1,729 MB\*\* |

| Recall@5 | Not projected | \*\*0.8833\*\* |



The comparison shows that:



\- index size was highly predictable;

\- retrieval latency remained reasonably close to the linear estimate;

\- memory growth was lower than projected;

\- embedding time was substantially underestimated;

\- retrieval quality required direct measurement rather than linear extrapolation.



Recall@5 was intentionally not projected because retrieval-quality degradation is not assumed to be linear with corpus size.



\---



\# Why V4 Stops at 250K



A pre-250K linear model suggested approximately:



```text

500K embedding ≈ 4.4 hours

1M embedding   ≈ 8.8 hours

```



However, the 250K experiment demonstrated lower sustained embedding throughput.



Using the measured 250K throughput of approximately:



```text

23.38 documents/second

```



a simple constant-throughput estimate would instead suggest approximately:



```text

500K ≈ 5.9 hours

1M   ≈ 11.9 hours

```



These values are planning estimates, not measurements.



Running increasingly large single-machine experiments would now provide diminishing scientific value because the principal bottleneck has already been identified.



Instead, larger workloads are more useful as targets for distributed-processing experiments.



\---



\# Main V4 Finding



V4 identifies a clear single-machine scaling pattern:



```text

Corpus size increases

&#x20;       │

&#x20;       ├── embedding time increases strongly

&#x20;       │

&#x20;       ├── sustained embedding throughput degrades at 250K

&#x20;       │

&#x20;       ├── FAISS index size increases approximately linearly

&#x20;       │

&#x20;       ├── exact retrieval latency increases

&#x20;       │

&#x20;       ├── Recall@5 decreases as distractors increase

&#x20;       │

&#x20;       └── memory remains manageable at the tested scales

```



The dominant computational bottleneck is:



> \*\*Document embedding generation.\*\*



This finding provides an empirical motivation for introducing distributed processing in the next project stage.



\---



\# Relationship to V3



V3 demonstrated that:



> \*\*Increasing retrieval depth does not necessarily improve end-to-end RAG quality.\*\*



V4 demonstrates a complementary result:



> \*\*Increasing corpus size makes relevant evidence harder to retrieve and substantially increases preprocessing cost.\*\*



Together:



```text

V3:

How much context should be retrieved?

&#x20;           +

V4:

How does retrieval behave as the corpus grows?

&#x20;           ↓

Future:

Adaptive and scalable RAG

```



This establishes two independent dimensions of the broader research problem:



```text

Retrieval depth

&#x20;       ×

Corpus scale

```



\---



\# Transition to Distributed Processing



The original project roadmap anticipated Apache Spark as the distributed-processing layer.



V4 now provides a measured reason to introduce it.



The next research question becomes:



> \*\*Can distributed processing reduce the document-embedding bottleneck identified in V4, and how does distributed execution scale with corpus size and computational resources?\*\*



A future distributed architecture can investigate:



```text

Large Corpus

&#x20;     ↓

Apache Spark

&#x20;     ↓

Document Partitions

&#x20; ┌───┼───┬───┐

&#x20; ▼   ▼   ▼   ▼

Worker Worker Worker Worker

&#x20; │   │   │   │

Embed Embed Embed Embed

&#x20; └───┼───┴───┘

&#x20;     ↓

Embedding Collection

&#x20;     ↓

Vector Index

&#x20;     ↓

Retrieval

```



Potential distributed-system metrics include:



\- Spark job execution time

\- number of partitions

\- worker count

\- executor memory

\- executor CPU allocation

\- documents embedded per second

\- shuffle and serialization overhead

\- speedup

\- parallel efficiency

\- cost

\- index-construction strategy

\- retrieval latency



\---



\# Future Comparison



The single-machine measurements from V4 provide the baseline for future distributed experiments.



For example:



\\\[

Speedup =

\\frac{T\_{single}}

&#x20;    {T\_{distributed}}

\\]



and:



\\\[

Efficiency =

\\frac{Speedup}

&#x20;    {NumberOfWorkers}

\\]



Future experiments can compare distributed measurements against the V4 single-machine baseline at common corpus sizes.



This is more informative than introducing Spark without first establishing where and why the local architecture becomes expensive.



\---



\# Experimental Artifacts



V4 introduces the following major scripts:



```text

src/

├── load\_msmarco.py

├── inspect\_msmarco\_qrels.py

├── prepare\_msmarco\_subsets.py

├── evaluate\_msmarco\_scaling.py

└── analyze\_v4\_scaling.py

```



Prepared benchmark structure:



```text

data/v4/

├── msmarco\_1k/

├── msmarco\_25k/

├── msmarco\_50k/

├── msmarco\_100k/

├── msmarco\_250k/

└── msmarco\_manifest.json

```



Each prepared subset contains:



```text

corpus.jsonl

queries.jsonl

qrels.tsv

metadata.json

```



Key V4 result files include:



```text

results/

├── v4\_msmarco\_1k\_scaling.json

├── v4\_msmarco\_25k\_scaling.json

├── v4\_msmarco\_50k\_scaling.json

├── v4\_msmarco\_100k\_scaling.json

├── v4\_msmarco\_250k\_scaling.json

├── v4\_scaling\_experiments\_pre100k.csv

├── v4\_scaling\_experiments.csv

├── v4\_scaling\_summary.csv

├── v4\_scaling\_projections.csv

├── v4\_embedding\_time.png

├── v4\_retrieval\_latency.png

├── v4\_index\_size.png

├── v4\_memory.png

└── v4\_recall\_at\_5.png

```



\---



\# Limitations



The V4 results should be interpreted within the experimental configuration.



Important limitations include:



1\. Experiments were executed on one CPU-based development laptop.

2\. Only one embedding model was evaluated.

3\. Only one exact FAISS index architecture (`IndexFlatIP`) was evaluated.

4\. Top-K was fixed at 5.

5\. The evaluation used a deterministic 300-query sample from MS MARCO validation qrels.

6\. Corpora were constructed as controlled nested samples rather than using the complete MS MARCO corpus.

7\. Maximum observed RSS was sampled at major pipeline stages rather than continuously monitored.

8\. The 250K embedding-throughput reduction was observed but its cause was not experimentally isolated.

9\. Larger single-machine corpus sizes such as 500K and 1M were not directly measured.

10\. Linear projections should be treated as planning estimates rather than experimental observations.

11\. V4 evaluates retrieval scalability and does not repeat the full LLM-groundedness experiments from V3 at every corpus size.



\---



\# V4 Conclusions



V4 establishes the following experimental observations:



1\. \*\*Document embedding is the dominant single-machine preprocessing cost.\*\*

2\. \*\*Embedding time grows strongly with corpus size.\*\*

3\. \*\*Sustained embedding throughput degraded at the 250K workload.\*\*

4\. \*\*FAISS flat-index construction remains inexpensive at the tested scales.\*\*

5\. \*\*FAISS index storage grows predictably with corpus size.\*\*

6\. \*\*Exact-search latency increases with corpus size but remains much smaller than embedding cost.\*\*

7\. \*\*Recall@5 declines as increasingly many distractor documents enter the corpus.\*\*

8\. \*\*Observed memory usage remains manageable through 250K documents on the tested machine.\*\*

9\. \*\*Simple linear extrapolation underestimated the 250K embedding cost.\*\*

10\. \*\*The measured embedding bottleneck provides an empirical justification for distributed processing.\*\*



\---



\# Next Stage



The next stage will move from:



```text

Single-machine embedding

```



toward:



```text

Distributed embedding and preprocessing

```



using Apache Spark.



The central next-stage question is:



> \*\*How much speedup and scalability can distributed processing provide for the embedding workload identified as the dominant V4 bottleneck?\*\*



This marks the transition from a local RAG research prototype toward the project's original objective:



> \*\*Scalable and optimized AI-powered data systems for large document collections.\*\*

