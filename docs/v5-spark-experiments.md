\# V5 — Apache Spark Local Parallelism and Embedding Architecture



\## Overview



V5 extends the scalability work from V4 by introducing \*\*Apache Spark\*\* as the distributed-processing framework for document embedding.



V4 established that document embedding is the dominant single-machine preprocessing bottleneck. At 250,000 MS MARCO documents, document embedding required approximately 2.97 hours, while FAISS index construction required only approximately 0.14 seconds.



This motivated the central V5 question:



> \*\*Can Apache Spark improve document-embedding scalability, and how do Spark parallelism, PyTorch threading, and data-ingestion architecture affect performance?\*\*



V5 deliberately begins with Spark in local mode before moving to a true multi-node cluster.



This allows Spark execution behavior, worker configuration, model replication, CPU oversubscription, and data-ingestion architecture to be studied under controlled conditions.



\---



\# Relationship to V4



V4 measured the following direct-Python MiniLM embedding baselines:



| Corpus | Documents | Embedding Time | Throughput |

|---|---:|---:|---:|

| MS MARCO 1K | 1,000 | 27.06 s | 36.96 docs/s |

| MS MARCO 25K | 25,000 | 825.46 s | 30.29 docs/s |

| MS MARCO 50K | 50,000 | 1,629.76 s | 30.68 docs/s |

| MS MARCO 100K | 100,000 | 3,171.49 s | 31.53 docs/s |

| MS MARCO 250K | 250,000 | 10,695.13 s | 23.38 docs/s |



These measurements provide the single-machine baseline against which V5 Spark experiments are compared.



For a Spark execution time \\(T\_{Spark}\\), speedup relative to V4 is defined as:



\\\[

Speedup =

\\frac{T\_{V4}}

&#x20;    {T\_{Spark}}

\\]



A value:



```text

Speedup > 1

```



indicates Spark is faster.



A value:



```text

Speedup < 1

```



indicates Spark is slower.



\---



\# V5 Research Questions



V5 investigates the following questions:



1\. Can Sentence Transformers execute correctly inside Spark Python workers?

2\. How much overhead does local Spark introduce for small embedding workloads?

3\. Does increasing Spark partition count improve embedding throughput?

4\. How does PyTorch's internal CPU parallelism interact with Spark worker parallelism?

5\. Does Spark approach or exceed direct-Python performance as workload size increases?

6\. Does driver-mediated corpus ingestion become a scalability problem?

7\. Can Spark-native ingestion provide a more scalable architecture?

8\. What limitations remain when all Spark workers execute on the same physical machine?

9\. What experimental evidence justifies moving to a genuine multi-node Spark cluster?



\---



\# Environment



V5 was executed using:



```text

Python:       3.11.9

Java:         OpenJDK 17

Java vendor:  Eclipse Temurin

PySpark:      3.5.9

Spark:        3.5.9

Embedding:    sentence-transformers/all-MiniLM-L6-v2

Dimension:    384

```



Spark local mode detected:



```text

Default parallelism: 8

```



corresponding to the logical processors available on the development machine.



\---



\# Spark Environment Configuration



\## Initial Java Environment



The machine initially contained Java 8:



```text

Java 8

JAVA\_HOME = C:\\Program Files\\Java\\jdk1.8.0\_231

```



V5 upgraded the project environment to Java 17.



The final Java configuration was:



```text

JAVA\_HOME =

C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.20.101-hotspot

```



and:



```text

java -version

```



reported OpenJDK 17.



\---



\# Legacy Spark Conflict



An existing Spark installation was discovered at:



```text

C:\\spark

```



with:



```text

Spark 2.4.5

Scala 2.11.12

```



while the project virtual environment contained:



```text

PySpark 3.5.9

```



The environment variable:



```text

SPARK\_HOME=C:\\spark

```



caused PySpark 3.5.9 to communicate with Spark 2.4.5 JVM libraries.



This produced:



```text

Py4JError:

org.apache.spark.api.python.PythonUtils.isEncryptionEnabled

does not exist in the JVM

```



The problem was resolved by removing the legacy `SPARK\_HOME` setting for the project environment.



After correction:



```text

spark-submit --version

```



reported:



```text

Spark 3.5.9

Scala 2.12.18

Java 17

```



This established a consistent software stack:



```text

Python 3.11

&#x20;     ↓

PySpark 3.5.9

&#x20;     ↓

Spark 3.5.9

&#x20;     ↓

Java 17

```



\---



\# Spark Smoke Test



A basic Spark smoke test was implemented in:



```text

src/spark\_smoke\_test.py

```



The test:



1\. created a local Spark session;

2\. generated a small distributed dataset;

3\. repartitioned the dataset;

4\. performed a Spark transformation;

5\. aggregated the results;

6\. compared the Spark result with the expected local result.



The test produced:



```text

Spark result:    338350

Expected result: 338350

```



and:



```text

Spark smoke test PASSED.

Local parallel Spark execution is working.

```



This verified the Java, PySpark, Spark, and Python integration.



\---



\# Sentence Transformer Worker Validation



The next experiment tested whether MiniLM could execute inside Spark Python workers.



Script:



```text

src/spark\_embedding\_smoke\_test.py

```



Configuration:



```text

Documents:  40

Partitions: 2

Model:      all-MiniLM-L6-v2

Dimension:  384

```



The experiment returned:



```text

Embeddings returned: 40

Embedding dimension: 384

```



and:



```text

Spark embedding smoke test PASSED.

Sentence Transformers can execute inside Spark Python workers.

```



The elapsed time was approximately:



```text

109.37 seconds

```



or:



```text

0.37 documents/second

```



This timing was not treated as a benchmark because worker startup and independent model loading dominated such a small workload.



The experiment demonstrated an important source of Spark overhead:



```text

Spark Worker 1 → load MiniLM

Spark Worker 2 → load MiniLM

```



Each Python worker maintains its own model instance.



\---



\# Experiment 1 — Uncontrolled Spark Parallelism



The first performance experiment used the 1K MS MARCO corpus and varied Spark partitions:



```text

P = 1, 2, 4, 8

```



PyTorch's internal CPU threading was not explicitly controlled in this experiment.



The V4 baseline was:



```text

27.06 seconds

```



for 1,000 documents.



\## Results



| Method | Partitions | Time | Throughput | Speedup vs V4 | Efficiency |

|---|---:|---:|---:|---:|---:|

| V4 direct Python | — | \*\*27.06 s\*\* | \*\*36.96 docs/s\*\* | 1.000x | — |

| Spark | 1 | 81.96 s | 12.20 docs/s | 0.330x | 0.330 |

| Spark | 2 | \*\*70.18 s\*\* | \*\*14.25 docs/s\*\* | \*\*0.386x\*\* | 0.193 |

| Spark | 4 | 76.14 s | 13.13 docs/s | 0.355x | 0.089 |

| Spark | 8 | 90.02 s | 11.11 docs/s | 0.301x | 0.038 |



The best uncontrolled Spark configuration was:



```text

P = 2

```



However, even this configuration was substantially slower than direct Python.



\---



\# Initial Parallelism Finding



Performance did not improve monotonically with Spark worker count.



Observed Spark embedding times were:



```text

P=1    81.96 s

P=2    70.18 s

P=4    76.14 s

P=8    90.02 s

```



This demonstrated:



> \*\*Increasing Spark task parallelism on fixed hardware does not necessarily increase embedding throughput.\*\*



One likely explanation was nested parallelism:



```text

Spark workers

&#x20;     ×

PyTorch internal CPU threads

&#x20;     ↓

CPU oversubscription

```



This motivated a controlled CPU-thread experiment.



\---



\# Experiment 2 — Controlled Spark and PyTorch Parallelism



The benchmark was extended to explicitly control:



```text

Spark partitions

```



and:



```text

PyTorch threads per worker

```



using:



```python

torch.set\_num\_threads(...)

```



PyTorch inter-op threads were restricted to one where possible.



Three configurations were designed around an approximate four-thread CPU budget:



```text

P=1, T=4

P=2, T=2

P=4, T=1

```



A fourth configuration tested all eight logical processors:



```text

P=8, T=1

```



where:



```text

P = Spark partitions / concurrent local task slots

T = PyTorch threads per worker

```



\---



\# Controlled 1K Results



| Spark Partitions | Torch Threads/Worker | Nominal Thread Budget | Time | Throughput | Speedup vs V4 |

|---:|---:|---:|---:|---:|---:|

| \*\*1\*\* | \*\*4\*\* | 4 | \*\*55.83 s\*\* | \*\*17.91 docs/s\*\* | \*\*0.485x\*\* |

| 2 | 2 | 4 | 63.87 s | 15.66 docs/s | 0.424x |

| 4 | 1 | 4 | 76.42 s | 13.09 docs/s | 0.354x |

| 8 | 1 | 8 | 84.68 s | 11.81 docs/s | 0.320x |



The best controlled local Spark configuration at 1K was:



```text

P=1

T=4

```



\---



\# Result — PyTorch Parallelism Was More Efficient Locally



When approximately the same four-thread CPU budget was divided among more Spark workers:



```text

1 × 4

2 × 2

4 × 1

```



performance became progressively worse:



```text

55.83 s

63.87 s

76.42 s

```



This indicates that, for this workload and hardware:



> \*\*PyTorch intra-worker CPU parallelism was more efficient than dividing the same CPU resources among multiple Spark Python workers.\*\*



The eight-worker configuration was slower still:



```text

84.68 seconds

```



despite using all eight logical processors as Spark task slots.



This provides evidence that worker/model replication and CPU contention can outweigh the benefit of additional local Spark tasks.



\---



\# Experiment 3 — Workload Scaling



The configuration:



```text

P=2

T=2

```



was selected as the primary multi-worker configuration for larger workloads.



This configuration uses:



```text

2 Spark workers

×

2 PyTorch threads per worker

=

nominal four-thread budget

```



The experiment compared V4 direct-Python embedding with Spark at:



```text

1K

25K

50K

```



documents.



\---



\# 25K Repeatability Experiment



Two independent P=2/T=2 runs were performed at 25K.



Results:



```text

Run 1: 818.1185 s

Run 2: 818.8053 s

```



The difference was:



```text

0.6868 seconds

```



or approximately:



```text

0.08%

```



This demonstrated excellent run-to-run repeatability.



The mean was approximately:



```text

818.46 seconds

```



compared with the V4 baseline:



```text

825.46 seconds

```



The corresponding speedup was approximately:



```text

1.009x

```



This result is best interpreted as \*\*performance parity\*\*, not a practically meaningful acceleration.



\---



\# Driver-Mediated Spark Workload Scaling



| Documents | V4 Direct Python | Spark P=2/T=2 | Approx. Speedup |

|---:|---:|---:|---:|

| 1,000 | \*\*27.06 s\*\* | 63.87 s | 0.424x |

| 25,000 | 825.46 s | \*\*\~818.46 s\*\* | \*\*\~1.009x\*\* |

| 50,000 | \*\*1,629.76 s\*\* | 1,669.97 s | 0.976x |



The results show that Spark overhead becomes less dominant as workload size increases.



At 1K:



```text

Spark is substantially slower.

```



At 25K:



```text

Spark reaches approximate parity.

```



At 50K:



```text

Spark remains close to the direct-Python baseline

but is slightly slower.

```



There is therefore no evidence that local Spark provides a sustained compute speedup over the optimized direct-Python implementation.



\---



\# Driver-Mediated Ingestion Problem



The original Spark benchmark loaded the corpus using Python:



```text

JSONL

&#x20; ↓

Python driver

&#x20; ↓

Python list

&#x20; ↓

spark.createDataFrame(...)

&#x20; ↓

Spark

```



At 50K documents, Spark reported:



```text

Stage 0 contains a task of very large size (8459 KiB).

The maximum recommended task size is 1000 KiB.

```



This exposed an architectural scalability problem.



The corpus was being routed through the Python driver before entering Spark.



Although the embedding experiment remained correct, this is not an appropriate design for larger distributed workloads.



\---



\# Experiment 4 — Spark-Native Data Ingestion



A second-generation benchmark was implemented in:



```text

src/evaluate\_spark\_embedding\_v2.py

```



The architecture was changed to:



```text

corpus.jsonl

&#x20;     ↓

Spark DataFrameReader

&#x20;     ↓

Spark transformations

&#x20;     ↓

Spark repartition

&#x20;     ↓

Python embedding workers

&#x20;     ↓

MiniLM

```



Spark reads the JSONL corpus directly using:



```python

spark.read.json(...)

```



instead of receiving a large Python collection from the driver.



This eliminates the driver-mediated corpus-transfer architecture.



\---



\# Spark-Native Configuration



The native-ingestion experiments used:



```text

Spark partitions:      2

PyTorch threads/worker: 2

Nominal thread budget: 4

Batch size:             64

Model:                  all-MiniLM-L6-v2

Dimension:              384

```



The same V4 corpora were used.



\---



\# Spark-Native Results



| Documents | V4 Direct | Spark-Native Embedding | Native Pipeline | Embedding Throughput | Embedding Speedup |

|---:|---:|---:|---:|---:|---:|

| 1K | \*\*27.06 s\*\* | 60.90 s | 67.54 s | 16.42 docs/s | 0.444x |

| 25K | \*\*825.46 s\*\* | 837.81 s | 845.70 s | 29.84 docs/s | 0.985x |

| 50K | \*\*1,629.76 s\*\* | 1,697.24 s | 1,706.07 s | 29.46 docs/s | 0.960x |



The pipeline time includes Spark read/count plus embedding but excludes Spark JVM startup.



\---



\# Architecture Comparison



For the selected P=2/T=2 configuration:



| Documents | V4 Direct | Driver-Mediated Spark | Spark-Native |

|---:|---:|---:|---:|

| 1K | \*\*27.06 s\*\* | 63.87 s | 60.90 s |

| 25K | 825.46 s | \*\*\~818.46 s\*\* | 837.81 s |

| 50K | \*\*1,629.76 s\*\* | 1,669.97 s | 1,697.24 s |



Spark-native ingestion does not provide a local compute speedup.



However, it provides a substantially better distributed architecture by eliminating the need to load and serialize the entire corpus through the Python driver.



This distinction is important:



> \*\*Architectural scalability and single-node execution speed are not the same property.\*\*



\---



\# Result — Spark Overhead Is Workload Dependent



For Spark-native embedding:



```text

1K     speedup = 0.444x

25K    speedup = 0.985x

50K    speedup = 0.960x

```



At small workloads, framework and model initialization overhead dominate.



At larger workloads, Spark approaches the direct-Python baseline because fixed overhead is amortized over more documents.



However, Spark did not demonstrate a sustained speedup above the direct-Python implementation.



\---



\# Result — Local Spark Does Not Add Compute Resources



All V5 experiments were performed in:



```text

Spark local mode

```



on one physical development machine.



Therefore:



```text

More Spark workers

```



does not mean:



```text

More physical CPU resources

```



Instead, multiple Spark workers compete for the same underlying processor cores.



Conceptually:



```text

Direct Python

&#x20;     ↓

optimized PyTorch CPU execution

&#x20;     ↓

same physical CPU



Spark Worker 1 ─┐

Spark Worker 2 ─┼── same physical CPU

Spark Worker N ─┘

```



This explains why additional local Spark workers cannot be assumed to produce distributed-computing speedup.



\---



\# Main V5 Finding



V5 establishes:



> \*\*Apache Spark provides a scalable execution architecture, but local Spark does not materially accelerate MiniLM document embedding when compared with an efficient multithreaded direct-Python implementation on the same physical hardware.\*\*



More specifically:



1\. Spark introduced substantial overhead for small workloads.

2\. Increasing Spark worker count did not monotonically improve performance.

3\. Excessive local parallelism reduced throughput.

4\. PyTorch intra-worker parallelism was more efficient than dividing the same CPU budget among multiple Spark workers.

5\. Spark approached direct-Python performance as workload size increased.

6\. A repeated 25K experiment demonstrated approximately 0.08% timing variation.

7\. No sustained local Spark speedup was observed through 50K documents.

8\. Driver-mediated corpus ingestion created a large serialized-task warning at 50K.

9\. Spark-native ingestion removed that architectural bottleneck.

10\. Spark-native ingestion improved distributed-system design even though it did not improve single-node execution time.



\---



\# V4 and V5 Together



V4 identified the primary single-machine bottleneck:



```text

Document embedding

```



V5 investigated whether Spark local parallelism could address it.



The combined result is:



```text

V4

Corpus grows

&#x20;   ↓

Embedding becomes expensive

&#x20;   ↓

Single-machine bottleneck identified



V5

Add Spark local parallelism

&#x20;   ↓

Same physical CPU resources

&#x20;   ↓

Framework + worker + model overhead

&#x20;   ↓

No material compute speedup

```



Therefore the next logical experiment is not:



```text

More local Spark partitions

```



but:



```text

More physical compute resources

```



\---



\# Why V5 Stops at 50K



A 100K local Spark experiment was considered but not performed.



By 50K, the experiments had already established that:



\- direct Python efficiently uses the local CPU;

\- local Spark introduces orchestration overhead;

\- increasing local worker count can reduce performance;

\- controlled P=2/T=2 reaches approximate parity at larger workloads;

\- Spark-native ingestion fixes the data-ingestion architecture;

\- local Spark does not add physical compute resources.



A 100K or 250K local Spark run would consume substantial additional time while testing the same hardware limitation.



The more informative next experiment is therefore a genuine multi-worker cluster.



\---



\# V5 Analysis Artifacts



The V5 analysis is implemented in:



```text

src/analyze\_v5\_spark.py

```



It combines:



```text

results/v5\_spark\_embedding\_experiments.csv

results/v5\_spark\_thread\_experiments.csv

results/v5\_spark\_native\_ingestion\_experiments.csv

```



and produces:



```text

results/v5\_parallelism\_summary.csv

results/v5\_workload\_scaling\_summary.csv

results/v5\_architecture\_summary.csv

```



Figures:



```text

results/v5\_parallelism\_1k.png

results/v5\_architecture\_comparison.png

results/v5\_speedup\_vs\_corpus\_size.png

results/v5\_throughput\_vs\_corpus\_size.png

```



\---



\# V5 Source Files



V5 introduces:



```text

src/

├── spark\_smoke\_test.py

├── spark\_embedding\_smoke\_test.py

├── evaluate\_spark\_embedding.py

├── evaluate\_spark\_embedding\_v2.py

└── analyze\_v5\_spark.py

```



\---



\# Limitations



The V5 results should be interpreted within the experimental environment.



Important limitations include:



1\. Spark was executed in local mode.

2\. All Spark workers shared one physical machine.

3\. Additional Spark workers did not provide additional physical CPU resources.

4\. Only one embedding model was tested.

5\. Only CPU inference was evaluated.

6\. Batch size was fixed at 64.

7\. Only a limited set of Spark/PyTorch thread configurations was evaluated.

8\. Worker memory was not measured as a complete cluster-wide memory metric.

9\. Driver RSS does not represent total JVM plus worker memory.

10\. Spark startup time was measured separately from embedding time.

11\. Native pipeline time included read/count plus embedding but excluded Spark JVM startup.

12\. The native DataFrame was not persisted between the count and embedding actions.

13\. 100K and 250K local Spark experiments were intentionally not performed.

14\. No true multi-node cluster was evaluated in V5.



\---



\# Lessons for Distributed AI Systems



V5 demonstrates several broader systems principles.



\## Parallel frameworks do not create hardware



```text

More tasks ≠ more compute

```



when all tasks share the same processor.



\## Parallelism has overhead



Distributed execution introduces:



\- scheduling;

\- serialization;

\- Python worker startup;

\- model initialization;

\- model replication;

\- communication;

\- coordination.



Parallel execution is beneficial only when useful computation is large enough to justify these costs.



\## Nested parallelism matters



AI workloads may already contain internal parallelism.



In this project:



```text

Spark parallelism

&#x20;       ×

PyTorch CPU parallelism

```



must be coordinated.



Ignoring this interaction can produce oversubscription and worse performance.



\## Data architecture matters



Using Spark does not automatically make a pipeline scalable.



The transition from:



```text

Python → Spark

```



to:



```text

Storage → Spark

```



was necessary to remove the driver-mediated corpus-transfer bottleneck.



\---



\# Transition to V6



V5 shows that local Spark cannot answer the central distributed-scaling question because all workers still share the same physical compute resources.



V6 therefore moves to \*\*true distributed Spark execution\*\*.



The central V6 research question is:



> \*\*When Spark receives genuinely additional compute resources, how does MiniLM document embedding scale with worker count?\*\*



\---



\# Proposed V6 Strong-Scaling Experiment



Hold workload size fixed while increasing actual worker resources:



```text

1 worker

&#x20;  ↓

2 workers

&#x20;  ↓

4 workers

```



For a fixed workload:



\\\[

Speedup(p)=

\\frac{T\_1}

&#x20;    {T\_p}

\\]



and:



\\\[

Efficiency(p)=

\\frac{Speedup(p)}

&#x20;    {p}

\\]



where \\(p\\) represents genuinely additional worker resources.



Primary candidate workloads:



```text

25K    distributed validation

100K   primary strong-scaling benchmark

250K   large distributed benchmark

```



Optional future workloads:



```text

500K

1M

```



\---



\# V6 Architecture



The Spark-native V5 architecture becomes the starting point for V6:



```text

&#x20;                Shared Storage

&#x20;                      │

&#x20;                      ▼

&#x20;                 Spark Driver

&#x20;                      │

&#x20;         ┌────────────┼────────────┐

&#x20;         ▼            ▼            ▼

&#x20;     Worker 1     Worker 2     Worker N

&#x20;         │            │            │

&#x20;      MiniLM       MiniLM       MiniLM

&#x20;         │            │            │

&#x20;         └────────────┼────────────┘

&#x20;                      ▼

&#x20;               Embedding Output

&#x20;                      │

&#x20;                      ▼

&#x20;                   FAISS

&#x20;                      │

&#x20;                      ▼

&#x20;                    RAG

```



Unlike V5 local mode, V6 workers should have genuinely separate compute resources.



\---



\# V6 Metrics



Future distributed experiments should record:



```text

dataset

documents

worker\_count

cores\_per\_worker

total\_worker\_cores

memory\_per\_worker

partitions

batch\_size

embedding\_model



data\_read\_time

embedding\_time

write\_time

total\_job\_time



documents\_per\_second

speedup

parallel\_efficiency

```



Where available, cluster-level metrics should also include:



```text

executor CPU utilization

executor memory

task duration

serialization overhead

shuffle traffic

failed tasks

retried tasks

```



\---



\# V5 Conclusion



V5 successfully introduced Apache Spark into the scalable RAG architecture and experimentally evaluated local parallelism, CPU-resource allocation, worker/model overhead, workload scaling, and data-ingestion design.



The experiments demonstrate that:



> \*\*Spark local mode provides orchestration and a path toward distributed execution, but it does not create additional compute capacity.\*\*



The optimized direct-Python MiniLM pipeline remained highly competitive because it already used the available local CPU resources efficiently.



Spark became approximately competitive at larger workloads, but no sustained local speedup was observed.



The most important V5 architectural result was the transition to Spark-native corpus ingestion, which removes the Python-driver data-transfer bottleneck and provides the appropriate foundation for true distributed execution.



The project is therefore ready for V6:



> \*\*Multi-worker Spark scaling using genuinely additional physical compute resources.\*\*

