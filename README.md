# Scalable RAG with Apache Spark

A research-oriented implementation of **Retrieval-Augmented Generation (RAG)** focused on retrieval quality, evidence grounding, corpus scalability, and distributed AI processing.

The project progressively evolves from a local semantic-retrieval prototype into an experimentally evaluated, scalable RAG architecture using:

- Sentence Transformers
- FAISS
- Large Language Models
- MS MARCO
- Apache Spark
- distributed embedding
- systematic retrieval and groundedness evaluation

The broader goal is to study how **retrieval quality, RAG reliability, computational cost, and distributed-system design interact as document collections grow**.

> **Current status:** V5 completed. The project now includes semantic retrieval, local RAG, systematic RAG evaluation, controlled MS MARCO scalability experiments from 1K to 250K documents, and Apache Spark embedding experiments. V5 demonstrates that local Spark provides a scalable execution architecture but does not materially accelerate MiniLM embedding when workers share the same physical CPU resources. Spark-native corpus ingestion now provides the foundation for true distributed execution in V6.

---

# Research Questions

The project investigates several related questions.

## Retrieval Quality

How do embedding models, retrieval depth, corpus size, and competing documents affect the probability of retrieving relevant evidence?

## RAG Groundedness

When relevant evidence is retrieved, does the generated answer remain grounded in that evidence?

## Citation Reliability

Are generated citations valid and faithful to the retrieved context?

## Corpus Scalability

How do embedding time, vector-index size, memory consumption, retrieval latency, and retrieval quality change as the corpus grows?

## Distributed Processing

Can Apache Spark reduce the document-embedding bottleneck when additional compute resources are available?

## Adaptive Optimization

Can future RAG systems automatically optimize retrieval depth, distributed-system configuration, latency, cost, and answer quality?

---

# Project Evolution

```text
V1  Semantic Retrieval                         ✅
        ↓
V2  Local End-to-End RAG                       ✅
        ↓
V3  RAG Evaluation & Top-K Experiments         ✅
        ↓
V4  Larger Corpus & Single-Machine Scalability ✅
        ↓
V5  Spark Local Parallelism & Architecture     ✅
        ↓
V6  True Distributed / Multi-Worker Spark      ← NEXT
        ↓
V7  Million-Scale Retrieval Experiments
        ↓
V8  Adaptive & Evolutionary RAG Optimization
```

---

# System Architecture

The current architecture has evolved through several stages.

## Local RAG Architecture

```text
Question
   ↓
Sentence Transformer
   ↓
Query Embedding
   ↓
FAISS
   ↓
Top-K Documents
   ↓
Prompt Construction
   ↓
LLM
   ↓
Grounded Answer + Citations
```

## Spark-Native Embedding Architecture

V5 introduced a Spark-native preprocessing architecture:

```text
Document Corpus
      ↓
Spark DataFrameReader
      ↓
Spark Transformations
      ↓
Spark Partitions
      ↓
Python Workers
      ↓
Sentence Transformer / MiniLM
      ↓
Document Embeddings
      ↓
Vector Index
      ↓
Retrieval
      ↓
RAG
```

V6 will extend this architecture from local Spark execution to genuinely distributed worker resources.

---

# Datasets

## SciFact

V1–V3 use the SciFact scientific-document dataset.

The local corpus contains approximately:

```text
5,183 documents
```

SciFact provides a manageable benchmark for developing and evaluating:

- semantic retrieval;
- Top-K retrieval;
- local RAG;
- evidence retrieval;
- groundedness;
- citation faithfulness;
- citation validity;
- abstention behavior.

---

## MS MARCO

V4 and V5 use the BEIR version of **MS MARCO**.

The source corpus contains:

```text
8,841,823 passages
```

and approximately:

```text
509,962 queries
```

The project uses the MS MARCO validation qrels to construct a controlled scalability benchmark.

---

# Controlled MS MARCO Benchmark

V4 selects:

```text
300 evaluation queries
```

using deterministic sampling with:

```text
random seed = 42
```

All positively relevant documents for these queries are retained.

Additional documents are deterministic distractors sampled from the full MS MARCO corpus.

The resulting corpora are nested:

```text
S_1K
  ⊂
S_25K
  ⊂
S_50K
  ⊂
S_100K
  ⊂
S_250K
```

Therefore:

```text
Same queries
Same gold documents
Same embedding model
Same FAISS architecture
Same Top-K
Same hardware

        ↓

Only corpus size changes
```

This allows corpus-size effects to be studied under controlled conditions.

---

# Embedding Model

The current embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

Embeddings are normalized before indexing.

---

# Vector Search

The current vector index is:

```text
FAISS IndexFlatIP
```

Because embeddings are normalized, inner-product ranking corresponds to cosine-similarity ranking.

The use of exact flat search also makes V4 scalability behavior easier to interpret.

---

# V1.0 — Semantic Retrieval

V1 established the basic retrieval pipeline:

```text
Documents
   ↓
Sentence Transformer
   ↓
Document Embeddings
   ↓
FAISS Index
   ↓
Query Embedding
   ↓
Top-K Semantic Retrieval
```

V1 validated:

- Sentence Transformer embedding;
- FAISS indexing;
- semantic nearest-neighbor retrieval;
- document-ID mapping;
- retrieval over SciFact.

This established the retrieval foundation used by later versions.

---

# V2.0 — Local End-to-End RAG

V2 extended semantic retrieval into a complete RAG pipeline:

```text
Question
   ↓
Retrieve Top-K Documents
   ↓
Construct Evidence Context
   ↓
LLM
   ↓
Evidence-Grounded Answer
```

V2 added:

- local end-to-end RAG;
- retrieved-context construction;
- LLM generation;
- source citations;
- abstention behavior when evidence is insufficient.

The central design principle was:

> **Generation should depend on retrieved evidence rather than unsupported model knowledge.**

---

# V3.0 — RAG Evaluation and Retrieval-Depth Experiments

V3 introduced systematic RAG evaluation.

The experiments examine the relationship between:

```text
Retrieval depth
        ↓
Evidence availability
        ↓
Context size
        ↓
Generation cost
        ↓
Groundedness
        ↓
Citation quality
```

---

# V3 Top-K Pilot Experiment

A pilot experiment evaluated:

```text
K = 1, 3, 5, 10
```

using five queries per K.

| K | Evidence Retrieval | Groundedness | Citation Faithfulness | Citation Validity | Avg. Generation Time | Avg. Context Characters |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.40 | 0.70 | 0.75 | 1.00 | 24.81 s | 1,361 |
| 3 | 0.60 | 0.70 | 0.70 | 1.00 | 38.10 s | 4,051 |
| 5 | **1.00** | **0.70** | 0.70 | 1.00 | 43.07 s | 6,549 |
| 10 | **1.00** | 0.60 | 0.667 | 1.00 | 91.71 s | 14,438 |

These pilot results contain only five queries per K and should not be interpreted as final estimates.

They nevertheless motivated a larger stratified experiment.

---

# V3 Stratified Retrieval Experiment

Queries were divided into four strata based on the rank of the relevant document:

```text
A = relevant document rank 1–3
B = relevant document rank 4–5
C = relevant document rank 6–10
D = relevant document not found in Top-10
```

The experiment evaluated:

```text
K = 3, 5, 10
```

with:

```text
60 total RAG runs
```

## Overall Results

| K | Runs | Evidence Retrieval | Groundedness | Citation Faithfulness | Citation Validity | Generation Failure | Avg. Time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 20 | 0.25 | 0.475 | 0.417 | 1.00 | 0.00 | 47.33 s |
| 5 | 20 | 0.50 | **0.525** | 0.417 | 1.00 | 0.00 | 48.51 s |
| 10 | 20 | **0.75** | 0.425 | 0.375 | 1.00 | **0.15** | **120.84 s** |

The main V3 observation is:

> **Retrieving more documents improves evidence recall but does not necessarily improve end-to-end RAG quality.**

Increasing K from 5 to 10 improved evidence retrieval:

```text
0.50 → 0.75
```

but groundedness decreased:

```text
0.525 → 0.425
```

while generation time increased substantially:

```text
48.51 s → 120.84 s
```

This motivates adaptive retrieval rather than assuming that larger K is always better.

Detailed V3 methodology is available in:

```text
docs/v3-experiments.md
```

---

# V4.0 — Larger Corpus and Single-Machine Scalability

V4 extends the project from RAG-quality evaluation to **retrieval-system scalability** using MS MARCO.

The controlled nested benchmark evaluates:

```text
1K
25K
50K
100K
250K
```

documents while preserving the same 300 evaluation queries and relevant documents.

---

# V4 Measured Results

| Documents | Recall@5 | Embedding Time | Throughput | Index Size | Query Latency | Max Observed RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 0.9967 | 27.06 s | 36.96 docs/s | 1.46 MB | 0.027 ms | 611 MB |
| 25K | 0.9800 | 825.46 s | 30.29 docs/s | 36.62 MB | 1.157 ms | 768 MB |
| 50K | 0.9533 | 1,629.76 s | 30.68 docs/s | 73.24 MB | 2.569 ms | 887 MB |
| 100K | 0.9400 | 3,171.49 s | 31.53 docs/s | 146.48 MB | 5.142 ms | 1,235 MB |
| **250K** | **0.8833** | **10,695.13 s** | **23.38 docs/s** | **366.21 MB** | **14.337 ms** | **1,729 MB** |

At 250K documents:

```text
Total pipeline time:
10,706.18 seconds

Document embedding:
10,695.13 seconds

FAISS index construction:
0.1399 seconds

300-query retrieval:
4.30 seconds
```

Document embedding therefore dominates preprocessing cost.

---

# V4 Embedding Bottleneck

From 1K to 100K, sustained embedding throughput remained approximately:

```text
30–32 documents/second
```

At 250K it fell to:

```text
23.38 documents/second
```

The 250K embedding workload required approximately:

```text
2.97 hours
```

This identifies document embedding as the dominant single-machine scalability bottleneck.

---

# V4 Retrieval-Quality Scaling

Recall@5 declined as increasingly many distractor documents entered the search space:

```text
1K       0.9967
25K      0.9800
50K      0.9533
100K     0.9400
250K     0.8833
```

The queries and relevant documents remain fixed.

Therefore:

> **Larger corpora can make relevant evidence more difficult to retain in the Top-5 even when the retrieval model and evaluation queries remain unchanged.**

From 1K to 250K, Recall@5 decreased by:

```text
0.1134
```

or approximately:

```text
11.34 percentage points
```

---

# V4 Scaling Projections

Using the five measured scale points, exploratory scaling projections estimate approximately:

| Documents | Projected Embedding Time | Projected Index Size | Projected Query Latency | Projected RSS |
|---:|---:|---:|---:|---:|
| 250K | 2.89 h | 366.21 MB | 14.19 ms | 1,776.65 MB |
| 500K | 5.89 h | 732.42 MB | 28.66 ms | 2,884.66 MB |
| 1M | 11.89 h | 1,464.84 MB | 57.60 ms | 5,100.67 MB |

These are projections rather than measurements.

Recall@5 is intentionally not projected because retrieval-quality degradation is not assumed to remain linear.

Detailed V4 methodology and results are available in:

```text
docs/v4-scalability.md
```

---

# V5.0 — Apache Spark Local Parallelism and Architecture

V5 introduces **Apache Spark** and investigates whether Spark parallelism can address the document-embedding bottleneck identified in V4.

The V5 software environment is:

```text
Python       3.11.9
Java         17
PySpark      3.5.9
Spark        3.5.9
Embedding    all-MiniLM-L6-v2
Dimension    384
```

---

# V5 Environment Validation

V5 first validates:

```text
Java 17
   ↓
PySpark 3.5.9
   ↓
Spark 3.5.9
   ↓
Local SparkSession
   ↓
Parallel Spark execution
```

A legacy Spark 2.4.5 installation was discovered through:

```text
SPARK_HOME=C:\spark
```

which conflicted with PySpark 3.5.9.

Removing the legacy `SPARK_HOME` setting allowed PySpark to use the matching Spark 3.5.9 libraries.

The final environment successfully passed the Spark smoke test.

---

# V5 Sentence Transformer Worker Validation

MiniLM was then executed inside Spark Python workers.

The smoke test used:

```text
Documents:  40
Partitions: 2
Dimension:  384
```

All 40 documents returned correct 384-dimensional embeddings.

This verified:

> **Sentence Transformers can execute correctly inside Spark Python workers.**

---

# V5 Uncontrolled Parallelism

The first 1K experiment varied Spark partitions without explicitly controlling PyTorch CPU threads.

| Method | Partitions | Embedding Time | Throughput | Speedup vs V4 |
|---|---:|---:|---:|---:|
| V4 direct Python | — | **27.06 s** | **36.96 docs/s** | 1.000x |
| Spark | 1 | 81.96 s | 12.20 docs/s | 0.330x |
| Spark | 2 | **70.18 s** | **14.25 docs/s** | **0.386x** |
| Spark | 4 | 76.14 s | 13.13 docs/s | 0.355x |
| Spark | 8 | 90.02 s | 11.11 docs/s | 0.301x |

Increasing local Spark worker count did not monotonically improve performance.

---

# V5 Controlled Spark and PyTorch Parallelism

The next experiment explicitly controlled both:

```text
Spark partitions
```

and:

```text
PyTorch threads per worker
```

The primary configurations were:

```text
P=1, T=4
P=2, T=2
P=4, T=1
P=8, T=1
```

where:

```text
P = Spark partitions
T = PyTorch threads per worker
```

---

# V5 Controlled 1K Results

| Spark Partitions | Torch Threads / Worker | Thread Budget | Embedding Time | Throughput | Speedup vs V4 |
|---:|---:|---:|---:|---:|---:|
| **1** | **4** | 4 | **55.83 s** | **17.91 docs/s** | **0.485x** |
| 2 | 2 | 4 | 63.87 s | 15.66 docs/s | 0.424x |
| 4 | 1 | 4 | 76.42 s | 13.09 docs/s | 0.354x |
| 8 | 1 | 8 | 84.68 s | 11.81 docs/s | 0.320x |

For approximately the same four-thread CPU budget:

```text
1 × 4 = 4
2 × 2 = 4
4 × 1 = 4
```

performance became progressively worse as the CPU resources were divided among additional Spark workers.

This suggests:

> **PyTorch intra-worker CPU parallelism is more efficient than excessive Spark-level parallelism on the tested single machine.**

---

# V5 Workload Scaling

The configuration:

```text
P=2
T=2
```

was selected as the primary multi-worker configuration for larger workloads.

## Results

| Documents | V4 Direct Python | Spark Driver-Mediated | Spark-Native |
|---:|---:|---:|---:|
| 1K | **27.06 s** | 63.87 s | 60.90 s |
| 25K | 825.46 s | **~818.46 s** | 837.81 s |
| 50K | **1,629.76 s** | 1,669.97 s | 1,697.24 s |

At 1K, Spark overhead dominates.

At 25K, Spark approaches performance parity.

At 50K, Spark remains close to the direct-Python baseline but does not provide a material speedup.

---

# V5 Repeatability

Two independent 25K P=2/T=2 driver-mediated runs produced:

```text
818.1185 s
818.8053 s
```

The difference was approximately:

```text
0.08%
```

demonstrating excellent run-to-run repeatability.

---

# V5 Spark-Native Ingestion

The initial Spark architecture used:

```text
JSONL
  ↓
Python driver
  ↓
Python list
  ↓
Spark
  ↓
embedding workers
```

At 50K documents, Spark warned that a serialized task had reached approximately:

```text
8.5 MB
```

The architecture was redesigned as:

```text
JSONL
  ↓
Spark DataFrameReader
  ↓
Spark transformations
  ↓
Spark partitions
  ↓
Python embedding workers
  ↓
MiniLM
```

This removes the need to route the full corpus through the Python driver.

---

# V5 Spark-Native Results

| Documents | V4 Direct | Native Embedding | Native Pipeline | Throughput | Embedding Speedup |
|---:|---:|---:|---:|---:|---:|
| 1K | **27.06 s** | 60.90 s | 67.54 s | 16.42 docs/s | 0.444x |
| 25K | **825.46 s** | 837.81 s | 845.70 s | 29.84 docs/s | 0.985x |
| 50K | **1,629.76 s** | 1,697.24 s | 1,706.07 s | 29.46 docs/s | 0.960x |

Spark-native ingestion improves the distributed architecture but does not make single-node execution faster.

---

# Key V5 Finding

> **Spark local mode provides orchestration and a scalable execution architecture, but it does not create additional physical compute resources.**

The experiments show that:

- small workloads are dominated by Spark and model-startup overhead;
- increasing Spark worker count can reduce performance;
- PyTorch internal CPU parallelism can outperform additional Spark workers on fixed hardware;
- Spark approaches direct-Python performance as workload size increases;
- no sustained local embedding speedup was observed;
- Spark-native ingestion eliminates the Python-driver corpus-transfer bottleneck.

This motivates the next stage:

> **Add genuinely additional physical compute resources rather than merely increasing local Spark tasks.**

Detailed V5 methodology and results are available in:

```text
docs/v5-spark-experiments.md
```

---

# Main Findings Across V1–V5

The project has now established several experimental findings.

## Finding 1 — Retrieval Depth Has a Tradeoff

V3 shows:

```text
Higher K
  ↓
Better evidence retrieval
```

but also:

```text
Higher K
  ↓
Larger context
  ↓
Higher generation cost
  ↓
Potentially lower groundedness
```

Therefore:

> **More retrieved context is not always better.**

---

## Finding 2 — Corpus Size Affects Retrieval Quality

V4 shows:

```text
Corpus grows
  ↓
More distractors
  ↓
Harder semantic retrieval
  ↓
Recall@5 decreases
```

Therefore:

> **Retrieval quality must be evaluated at realistic corpus scale.**

---

## Finding 3 — Embedding Is the Dominant Offline Bottleneck

At 250K documents:

```text
Embedding:
~2.97 hours

FAISS index build:
~0.14 seconds
```

Therefore:

> **Document embedding dominates the current preprocessing pipeline.**

---

## Finding 4 — Parallel Frameworks Do Not Create Hardware

V5 shows:

```text
More Spark tasks
      ≠
More physical compute
```

Increasing local Spark worker count can actually reduce performance because of:

- Spark scheduling overhead;
- Python worker startup;
- model replication;
- serialization;
- nested PyTorch parallelism;
- CPU contention.

---

## Finding 5 — Data Architecture Matters

The transition:

```text
Python → Spark
```

to:

```text
Storage → Spark
```

removes a driver-side data-transfer bottleneck and provides the correct architecture for future distributed execution.

---

# V6 — True Distributed / Multi-Worker Spark

V6 will move beyond Spark local mode.

The central research question is:

> **When Spark receives genuinely additional compute resources, how does MiniLM document embedding scale with worker count?**

The proposed strong-scaling experiment holds workload size fixed while increasing actual worker resources:

```text
1 worker
   ↓
2 workers
   ↓
4 workers
```

For \(p\) workers:

\[
Speedup(p)
=
\frac{T_1}
     {T_p}
\]

and:

\[
Efficiency(p)
=
\frac{Speedup(p)}
     {p}
\]

Primary candidate workloads:

```text
25K     validation
100K    primary strong-scaling benchmark
250K    large distributed benchmark
```

Optional future workloads:

```text
500K
1M
```

---

# Future Distributed Architecture

```text
                 Shared Storage
                       │
                       ▼
                  Spark Driver
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Worker 1     Worker 2     Worker N
          │            │            │
       MiniLM       MiniLM       MiniLM
          │            │            │
          └────────────┼────────────┘
                       ▼
                Embedding Output
                       │
                       ▼
                 Vector Index
                       │
                       ▼
                    Retrieval
                       │
                       ▼
                      RAG
```

V6 will distinguish true distributed scaling from local task parallelism.

---

# V7 — Million-Scale Retrieval

After distributed embedding is validated, future experiments can extend the corpus toward:

```text
500K
1M
```

documents.

Potential research questions include:

- distributed embedding throughput;
- large-index storage;
- exact versus approximate vector search;
- FAISS IVF/HNSW alternatives;
- distributed index construction;
- retrieval latency;
- Recall@K degradation;
- cost/performance tradeoffs.

---

# V8 — Adaptive and Evolutionary RAG Optimization

The project ultimately aims to move beyond fixed configurations.

A future system may optimize:

```text
Top-K
worker count
partition count
batch size
threads per worker
memory allocation
retrieval architecture
```

using measured objectives such as:

```text
latency
cost
memory
retrieval quality
groundedness
citation faithfulness
```

A multi-objective formulation could optimize:

\[
\min_x
[
Latency(x),
Cost(x),
Memory(x)
]
\]

subject to quality constraints such as:

\[
Recall@K \geq R_{min}
\]

and:

\[
Groundedness \geq G_{min}
\]

This creates a natural path toward evolutionary and adaptive optimization of RAG systems.

---

# Repository Structure

A simplified project structure is:

```text
scalable-rag-spark/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── v3-experiments.md
│   ├── v4-scalability.md
│   └── v5-spark-experiments.md
│
├── src/
│   ├── config.py
│   ├── load_scifact.py
│   ├── build_index.py
│   ├── retrieve.py
│   ├── rag.py
│   ├── evaluate_retrieval.py
│   │
│   ├── evaluate_topk.py
│   ├── evaluate_rag_topk.py
│   ├── prepare_groundedness_evaluation.py
│   ├── analyze_groundedness.py
│   ├── evaluate_rag_stratified.py
│   ├── prepare_stratified_groundedness.py
│   ├── analyze_stratified_groundedness.py
│   │
│   ├── load_msmarco.py
│   ├── inspect_msmarco_qrels.py
│   ├── prepare_msmarco_subsets.py
│   ├── evaluate_msmarco_scaling.py
│   ├── analyze_v4_scaling.py
│   │
│   ├── spark_smoke_test.py
│   ├── spark_embedding_smoke_test.py
│   ├── evaluate_spark_embedding.py
│   ├── evaluate_spark_embedding_v2.py
│   └── analyze_v5_spark.py
│
├── results/
│   ├── ...
│   ├── v4_scaling_summary.csv
│   ├── v4_scaling_projections.csv
│   ├── v4_embedding_time.png
│   ├── v4_retrieval_latency.png
│   ├── v4_index_size.png
│   ├── v4_memory.png
│   ├── v4_recall_at_5.png
│   │
│   ├── v5_spark_embedding_experiments.csv
│   ├── v5_spark_thread_experiments.csv
│   ├── v5_spark_native_ingestion_experiments.csv
│   ├── v5_parallelism_summary.csv
│   ├── v5_workload_scaling_summary.csv
│   ├── v5_architecture_summary.csv
│   ├── v5_parallelism_1k.png
│   ├── v5_architecture_comparison.png
│   ├── v5_speedup_vs_corpus_size.png
│   └── v5_throughput_vs_corpus_size.png
│
└── data/
    └── v4/
        ├── msmarco_1k/
        ├── msmarco_25k/
        ├── msmarco_50k/
        ├── msmarco_100k/
        └── msmarco_250k/
```

Generated MS MARCO subsets and indexes under:

```text
data/v4/
```

are excluded from Git because they are reproducible and potentially large.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/BigDataAISystems/scalable-rag-spark.git
cd scalable-rag-spark
```

Create a virtual environment.

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

# Core Dependencies

The project currently uses:

```text
sentence-transformers
faiss-cpu
numpy
pandas
datasets
python-dotenv
openai
tqdm
requests
psutil
matplotlib
pyspark
```

V5 uses:

```text
pyspark==3.5.9
```

and Java 17.

---

# Spark Environment

Verify Java:

```powershell
java -version
```

V5 uses Java 17.

Verify PySpark:

```powershell
python -c "import pyspark; print(pyspark.__version__)"
```

Expected:

```text
3.5.9
```

Verify Spark:

```powershell
spark-submit --version
```

Expected:

```text
Spark 3.5.9
```

If an older standalone Spark installation exists, make sure an incompatible `SPARK_HOME` does not override the PySpark installation.

---

# Reproducing V4

Prepare the MS MARCO subsets:

```powershell
python .\src\prepare_msmarco_subsets.py
```

Run a scalability experiment:

```powershell
python .\src\evaluate_msmarco_scaling.py --subset msmarco_1k
```

Analyze V4:

```powershell
python .\src\analyze_v4_scaling.py
```

---

# Reproducing V5

Run the basic Spark smoke test:

```powershell
python .\src\spark_smoke_test.py
```

Run the Sentence Transformer worker smoke test:

```powershell
python .\src\spark_embedding_smoke_test.py
```

Run a controlled Spark embedding experiment:

```powershell
python .\src\evaluate_spark_embedding.py --subset msmarco_1k --partitions 2 --torch-threads 2
```

Run the Spark-native ingestion experiment:

```powershell
python .\src\evaluate_spark_embedding_v2.py --subset msmarco_25k --partitions 2 --torch-threads 2
```

Analyze V5:

```powershell
python .\src\analyze_v5_spark.py
```

---

# Experimental Methodology

The project follows several principles intended to make results reproducible and interpretable.

## Controlled Variables

Where possible, experiments hold constant:

- evaluation queries;
- gold documents;
- embedding model;
- embedding dimension;
- retrieval depth;
- vector-index architecture;
- random seed;
- hardware environment.

## Measured vs Projected Results

Measured values and projected values are stored separately.

Projected results are explicitly labeled and are not treated as experimental observations.

## Human Evaluation

V3 groundedness and citation-faithfulness scores use human ratings:

```text
0
1
2
```

which are normalized to:

```text
0–1
```

for aggregate reporting.

## Repeated Measurements

Repeated experiments are retained when useful.

For example, the two 25K Spark runs in V5 are preserved because they demonstrate timing repeatability.

---

# Limitations

Current limitations include:

1. V1–V3 use a relatively small SciFact corpus.
2. V3 human-scored experiments remain limited in sample size.
3. V4 uses controlled nested subsets rather than the full MS MARCO corpus.
4. Only one embedding model has been evaluated.
5. V4 uses exact FAISS `IndexFlatIP`.
6. V4 memory measurements represent maximum observed RSS rather than continuous process-wide peak memory.
7. V5 runs Spark in local mode.
8. V5 workers share one physical machine.
9. Driver RSS does not represent complete Spark JVM and worker memory.
10. GPU embedding has not yet been evaluated.
11. A true multi-node Spark cluster has not yet been evaluated.
12. Million-scale embedding and retrieval remain future work.

---

# Version History

## v1.0 — Semantic Retrieval

Completed:

- SciFact ingestion
- Sentence Transformer embeddings
- FAISS indexing
- semantic retrieval

---

## v2.0 — Local RAG

Completed:

- end-to-end RAG
- retrieved-context construction
- LLM generation
- evidence citations
- abstention behavior

---

## v3.0 — RAG Evaluation

Completed:

- Top-K retrieval experiments
- evidence-retrieval analysis
- groundedness evaluation
- citation-faithfulness evaluation
- citation-validity evaluation
- stratified retrieval experiment
- generation-failure analysis

Key result:

> Increasing retrieval depth improves evidence availability but does not necessarily improve groundedness or end-to-end RAG quality.

---

## v4.0 — Larger Corpus and Single-Machine Scalability

Completed:

- MS MARCO integration
- qrel inspection
- deterministic 300-query benchmark
- nested 1K–250K corpora
- embedding scalability experiments
- FAISS storage measurements
- retrieval-latency measurements
- Recall@5 scaling analysis
- memory measurements
- scalability projections

Key result:

> **Document embedding emerged as the dominant single-machine scalability bottleneck.**

At 250K:

```text
Embedding time       = 10,695.13 s (~2.97 h)
Embedding throughput = 23.38 docs/s
FAISS index size     = 366.21 MB
Query latency        = 14.34 ms
Recall@5             = 0.8833
Observed RSS         = 1,729 MB
```

---

## v5.0 — Apache Spark Local Parallelism and Architecture

Completed:

- Java 17 / Spark 3.5.9 environment
- PySpark 3.5.9 integration
- Spark smoke testing
- Sentence Transformer execution inside Spark workers
- local partition-scaling experiments
- controlled PyTorch-thread experiments
- 25K repeatability testing
- workload scaling through 50K
- driver-mediated ingestion evaluation
- Spark-native JSONL ingestion
- direct Python vs Spark comparison
- V5 analysis and visualization

Key findings:

> **Increasing Spark workers on fixed hardware does not automatically improve embedding performance.**

and:

> **Spark-native ingestion provides the appropriate architecture for distributed execution, but meaningful compute speedup requires additional physical compute resources.**

The best controlled 1K local Spark configuration was:

```text
Spark partitions      = 1
Torch threads/worker  = 4
Embedding time        = 55.83 s
Speedup vs V4         = 0.485x
```

For P=2/T=2:

```text
             V4 Direct       Spark Native

1K              27.06 s          60.90 s
25K            825.46 s         837.81 s
50K          1,629.76 s       1,697.24 s
```

V5 establishes the experimental foundation for true distributed Spark scaling in V6.

---

# Next Milestone — V6

V6 will evaluate Spark using **genuinely additional worker resources**.

Primary goals:

- establish a one-worker distributed baseline;
- run 2-worker and 4-worker configurations;
- measure strong scaling;
- calculate speedup;
- calculate parallel efficiency;
- evaluate distributed embedding throughput;
- measure data-read and output-write overhead;
- compare distributed results against V4 and V5;
- scale toward 100K and 250K documents.

The central question is:

> **How much acceleration does Spark provide when additional Spark workers correspond to genuinely additional physical compute resources?**

---

# Repository

GitHub:

```text
https://github.com/BigDataAISystems/scalable-rag-spark
```

Project website:

```text
https://mebrahimii.github.io/
```

---

# Author

**Mahdi Ebrahimi, Ph.D.**  
Associate Professor of Computer Science  
California State University, Northridge

Research interests include:

- Big Data
- distributed systems
- cloud computing
- AI-powered data systems
- information retrieval
- optimization
- scalable machine learning
- Retrieval-Augmented Generation