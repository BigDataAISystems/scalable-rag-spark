\# V3 — RAG Evaluation and Retrieval Experiments



\## Overview



V3 extends the project from building a working Retrieval-Augmented Generation (RAG) pipeline to \*\*systematically evaluating retrieval depth, groundedness, citation behavior, context size, and generation latency\*\*.



The central question is:



> \*\*How does retrieval depth (Top-K) affect retrieval effectiveness and downstream RAG behavior?\*\*



A second question emerged from the experiments:



> \*\*Does retrieving more documents necessarily improve RAG answer quality?\*\*



The experiments show that these are not equivalent objectives.



Increasing K generally improves retrieval recall, but larger contexts can increase latency, introduce irrelevant evidence, reduce groundedness, and increase generation failures.



\---



\# Experimental Environment



The V3 experiments use the V2 local RAG architecture.



\## Dataset



\*\*SciFact\*\*



\- Corpus documents: 5,183

\- Queries loaded: 1,109

\- Retrieval evaluation queries: 300



SciFact provides relevance judgments that allow retrieved documents to be compared with known relevant documents.



\---



\## Retrieval Stack



Embedding model:



```text

sentence-transformers/all-MiniLM-L6-v2

```



Embedding dimension:



```text

384

```



Vector search:



```text

FAISS

```



\---



\## Generation Stack



Local LLM:



```text

Qwen3 1.7B

```



Runtime:



```text

Ollama

```



The experiments were executed locally on CPU-based commodity hardware.



No paid hosted LLM API was required.



\---



\# V3 Experiment 1 — Retrieval Depth



\## Research Question



> How does increasing Top-K affect the probability of retrieving at least one known relevant SciFact document?



The following values were evaluated:



```text

K = {1, 3, 5, 10, 20}

```



The same 300 SciFact evaluation queries were used for every K.



\---



\## Results



| K | Queries | Recall@K | Evaluation Time |

|---:|---:|---:|---:|

| 1 | 300 | 0.5033 | 6.39 s |

| 3 | 300 | 0.6833 | 6.25 s |

| 5 | 300 | 0.7533 | 6.16 s |

| 10 | 300 | 0.7933 | 6.12 s |

| 20 | 300 | \*\*0.8467\*\* | 6.19 s |



\---



\## Observation



Retrieval recall increases as K increases.



The largest gain occurs between:



```text

K = 1 → K = 3

```



with:



```text

0.5033 → 0.6833

```



Increasing K beyond 5 continues to improve recall, but the improvements become smaller.



For this relatively small FAISS index, retrieval evaluation time remains approximately constant across the tested K values.



Therefore, retrieval itself is not the primary performance bottleneck in the current system.



\---



\# Retrieval Recall Is Not RAG Quality



Experiment 1 shows:



```text

larger K

&#x20;  ↓

higher probability of retrieving relevant evidence

```



However, a RAG system must also process the retrieved documents with an LLM.



Increasing K therefore also causes:



```text

larger K

&#x20;  ↓

more retrieved documents

&#x20;  ↓

larger context

&#x20;  ↓

more potential evidence

&#x20;  +

more potential noise

&#x20;  ↓

longer generation

```



This motivated the remaining V3 experiments.



\---



\# V3 Experiment 2 — Initial RAG Top-K Pilot



\## Research Question



> Does increasing K improve actual RAG behavior, or only retrieval recall?



A pilot experiment used:



```text

K = {1, 3, 5, 10}

```



with five reproducible SciFact queries.



This resulted in:



```text

5 queries × 4 K values = 20 RAG runs

```



For every run, the experiment recorded:



\- query ID

\- Top-K

\- retrieved document IDs

\- retrieval similarity scores

\- whether a known relevant document was retrieved

\- rank of the first relevant document

\- context size

\- generation latency

\- generated citations

\- generated answer



\---



\## Pilot Results



| K | Relevant Evidence Retrieved | Avg. Context Characters | Avg. Generation Time | Avg. Citations |

|---:|---:|---:|---:|---:|

| 1 | 0.40 | 1,361 | 24.81 s | 0.8 |

| 3 | 0.60 | 4,051 | 38.10 s | 1.4 |

| 5 | \*\*1.00\*\* | 6,549 | 43.07 s | 1.4 |

| 10 | \*\*1.00\*\* | 14,438 | \*\*91.71 s\*\* | 0.8 |



The five-query sample is intentionally small and should not be interpreted as an estimate of general SciFact retrieval performance.



Its purpose was to validate the experimental methodology.



\---



\## Pilot Observation



K=5 retrieved known relevant evidence for all five pilot queries.



Moving from K=5 to K=10 did not improve evidence coverage in this sample, while average context size increased by approximately 120% and average generation latency increased substantially.



The experiment also revealed that citation presence alone does not guarantee that the cited document actually supports the generated claim.



This motivated explicit groundedness and citation-faithfulness evaluation.



\---



\# V3 Experiment 3 — Human Groundedness Evaluation



\## Motivation



A RAG answer may contain a syntactically valid citation while still making a claim that is not supported by the cited evidence.



Therefore, four dimensions were manually evaluated.



\---



\## Groundedness



Scores:



```text

2 = substantive claims directly supported by retrieved evidence



1 = partially supported, but the model makes an inference

&#x20;   or stronger claim than the evidence establishes



0 = central answer unsupported, contradictory, or off-topic

```



Scores were normalized to:



```text

0–1

```



for analysis.



\---



\## Citation Faithfulness



Scores:



```text

2 = cited evidence directly supports the associated claim



1 = cited evidence is related but provides only partial

&#x20;   or indirect support



0 = citation does not support the associated claim

```



Scores were normalized to:



```text

0–1

```



\---



\## Citation Validity



```text

1 = citation refers to an actual retrieved document



0 = citation refers to an invalid/nonexistent document

```



Citation validity is different from citation faithfulness.



A citation may be valid while still failing to support the claim attributed to it.



\---



\## Abstention Correctness



```text

1 = model appropriately answers when evidence is sufficient

&#x20;   or appropriately reports insufficient evidence



0 = model confidently answers despite insufficient evidence

&#x20;   or unnecessarily abstains despite adequate evidence

```



\---



\## Pilot Groundedness Results



| K | Evidence Retrieval | Groundedness | Citation Faithfulness | Citation Validity | Abstention Correctness | Avg. Time | Avg. Context |

|---:|---:|---:|---:|---:|---:|---:|---:|

| 1 | 0.40 | 0.70 | \*\*0.75\*\* | 1.00 | 0.40 | 24.81 s | 1,361 |

| 3 | 0.60 | 0.70 | 0.70 | 1.00 | 0.40 | 38.10 s | 4,051 |

| 5 | \*\*1.00\*\* | \*\*0.70\*\* | 0.70 | 1.00 | 0.40 | 43.07 s | 6,549 |

| 10 | \*\*1.00\*\* | 0.60 | 0.6667 | 1.00 | 0.40 | \*\*91.71 s\*\* | 14,438 |



These results suggested that K=5 was a promising operating point, but five queries were insufficient for a stronger conclusion.



This motivated a larger controlled experiment.



\---



\# V3 Experiment 4 — Stratified RAG Evaluation



\## Motivation



Randomly selecting queries makes it difficult to determine whether changes in answer quality occur because relevant evidence entered the context.



Experiment 4 therefore stratified queries according to the rank of the first known relevant document.



\---



\## Retrieval Strata



Queries were classified using Top-10 retrieval.



| Stratum | Relevant Document Rank | Interpretation |

|---|---|---|

| A | 1–3 | Relevant evidence available at K=3 |

| B | 4–5 | Relevant evidence first becomes available at K=5 |

| C | 6–10 | Relevant evidence first becomes available at K=10 |

| D | >10 / missing | Relevant evidence unavailable for all tested K |



Five queries were selected from each stratum.



Therefore:



```text

4 strata × 5 queries = 20 queries

```



Each query was evaluated using:



```text

K = {3, 5, 10}

```



giving:



```text

20 queries × 3 K values = 60 RAG runs

```



\---



\# Overall Stratified Results



| K | Runs | Evidence Retrieval | Groundedness | Citation Faithfulness | Citation Validity | Abstention Correctness | Generation Failure | Avg. Time | Avg. Context |

|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| 3 | 20 | 0.25 | 0.475 | 0.4167 | 1.00 | 0.35 | 0.00 | 47.33 s | 4,252 |

| \*\*5\*\* | 20 | \*\*0.50\*\* | \*\*0.525\*\* | \*\*0.4167\*\* | \*\*1.00\*\* | \*\*0.50\*\* | \*\*0.00\*\* | \*\*48.51 s\*\* | \*\*6,982\*\* |

| 10 | 20 | \*\*0.75\*\* | 0.425 | 0.3750 | 1.00 | 0.40 | \*\*0.15\*\* | \*\*120.84 s\*\* | \*\*14,321\*\* |



\---



\# Result 1 — K=5 Provides the Best Observed Balance



Moving from:



```text

K=3 → K=5

```



increased evidence retrieval:



```text

0.25 → 0.50

```



and increased groundedness:



```text

0.475 → 0.525

```



while average generation time changed only slightly:



```text

47.33 s → 48.51 s

```



No generation failures were observed for either K=3 or K=5.



For the current model and experimental sample, K=5 therefore provides the strongest observed balance between retrieval coverage, groundedness, and generation cost.



This should be interpreted as a V3 experimental result rather than a universal optimal K.



\---



\# Result 2 — Higher Retrieval Recall Does Not Guarantee Better RAG



K=10 provides the highest evidence retrieval rate:



```text

0.75

```



but groundedness decreases:



```text

0.525 at K=5

&#x20;       ↓

0.425 at K=10

```



Citation faithfulness also decreases:



```text

0.4167

&#x20;  ↓

0.3750

```



Average generation latency increases from:



```text

48.51 s

```



to:



```text

120.84 s

```



and the generation failure rate increases to:



```text

15%

```



Therefore:



> \*\*Maximizing retrieval depth or retrieval recall does not necessarily maximize end-to-end RAG quality.\*\*



\---



\# Result 3 — Stratum A: Excess Context Can Hurt



Stratum A already contains relevant evidence within the first three documents.



| K | Retrieval | Groundedness | Faithfulness | Failure | Avg. Time |

|---:|---:|---:|---:|---:|---:|

| 3 | 1.00 | \*\*0.70\*\* | \*\*0.70\*\* | 0.00 | 42.59 s |

| 5 | 1.00 | \*\*0.70\*\* | \*\*0.70\*\* | 0.00 | 49.23 s |

| 10 | 1.00 | 0.50 | 0.50 | 0.20 | 118.69 s |



Increasing K from 3 to 10 provides no retrieval benefit because the relevant evidence was already available.



However:



\- groundedness decreases

\- citation faithfulness decreases

\- generation latency increases substantially

\- generation failures appear



This provides evidence of \*\*context dilution or context pollution\*\* for the current local model.



\---



\# Result 4 — Stratum B: Increasing K Can Help When It Introduces Evidence



For Stratum B, relevant evidence appears at ranks 4–5.



Therefore:



```text

K=3 → relevant evidence absent

K=5 → relevant evidence present

```



Results:



| K | Retrieval | Groundedness | Faithfulness | Abstention | Avg. Time |

|---:|---:|---:|---:|---:|---:|

| 3 | 0.00 | 0.40 | 0.40 | 0.00 | 41.46 s |

| 5 | \*\*1.00\*\* | \*\*0.60\*\* | \*\*0.60\*\* | 0.40 | 41.25 s |

| 10 | 1.00 | 0.70 | 0.625 | 0.60 | 102.87 s |



When relevant evidence enters the context at K=5, groundedness improves:



```text

0.40 → 0.60

```



without increasing average generation latency in this small stratum.



This demonstrates that increasing K can improve RAG quality \*\*when the additional retrieval actually introduces useful evidence\*\*.



\---



\# Result 5 — Stratum C: Retrieval Success Is Not Sufficient



For Stratum C:



```text

K=3  → evidence absent

K=5  → evidence absent

K=10 → evidence present

```



Results:



| K | Retrieval | Groundedness | Faithfulness | Failure | Avg. Time |

|---:|---:|---:|---:|---:|---:|

| 3 | 0.00 | 0.20 | 0.00 | 0.00 | 49.62 s |

| 5 | 0.00 | 0.30 | 0.00 | 0.00 | 44.25 s |

| 10 | \*\*1.00\*\* | \*\*0.30\*\* | 0.1667 | \*\*0.20\*\* | \*\*127.12 s\*\* |



The relevant document finally enters the context at K=10.



However, groundedness does not improve relative to K=5:



```text

0.30 → 0.30

```



while latency increases dramatically and generation failures appear.



This demonstrates an important distinction:



> \*\*Retrieving relevant evidence is not sufficient to guarantee that the generator will successfully use it.\*\*



The relevant document may be diluted by a larger amount of competing context.



\---



\# Result 6 — Stratum D: More Irrelevant Context Can Degrade RAG



Stratum D contains queries where known relevant evidence is not retrieved within Top-10.



| K | Retrieval | Groundedness | Faithfulness | Failure | Avg. Time |

|---:|---:|---:|---:|---:|---:|

| 3 | 0.00 | \*\*0.60\*\* | \*\*0.50\*\* | 0.00 | 55.65 s |

| 5 | 0.00 | 0.50 | 0.25 | 0.00 | 59.29 s |

| 10 | 0.00 | \*\*0.20\*\* | \*\*0.00\*\* | 0.20 | \*\*134.67 s\*\* |



As more documents are added without introducing the known relevant evidence, groundedness progressively decreases.



This supports the hypothesis that indiscriminately increasing context can introduce noise rather than useful information.



\---



\# Major V3 Finding



The experiments suggest the following relationship:



```text

Increasing K

&#x20;    │

&#x20;    ├── can improve retrieval coverage

&#x20;    │

&#x20;    ├── can improve RAG when useful evidence enters context

&#x20;    │

&#x20;    └── can hurt RAG when it primarily introduces noise

```



Therefore:



> \*\*The optimal retrieval depth for RAG is not determined by Recall@K alone.\*\*



End-to-end RAG behavior depends on both:



```text

retrieval quality

&#x20;       +

generator ability to identify and use evidence

```



\---



\# From Fixed-K RAG to Adaptive RAG



The experiments also raise a broader question:



> Why should every query use the same K?



For some queries:



```text

relevant evidence rank ≤ 3

→ K=3 may be sufficient

```



For others:



```text

relevant evidence rank = 4–5

→ K=5 may be beneficial

```



For more difficult queries:



```text

relevant evidence rank = 6–10

→ increasing K retrieves the evidence

&#x20; but may create excessive context

```



And for some queries:



```text

relevant evidence > 10

→ simply increasing K may add noise

```



This motivates an adaptive policy:



```text

Question

&#x20;  ↓

Initial Retrieval

&#x20;  ↓

Evaluate retrieval confidence

&#x20;  │

&#x20;  ├── sufficient

&#x20;  │      ↓

&#x20;  │   Generate

&#x20;  │

&#x20;  └── insufficient

&#x20;         ↓

&#x20;     Expand retrieval

&#x20;         ↓

&#x20;     Rerank / filter

&#x20;         ↓

&#x20;     Generate or abstain

```



Instead of:



```text

K = constant

```



future systems can investigate:



```text

K = f(

&#x20;   query,

&#x20;   retrieval scores,

&#x20;   score gaps,

&#x20;   evidence confidence,

&#x20;   context cost

)

```



\---



\# Multi-Objective RAG Optimization



V3 also establishes the basis for treating RAG configuration as a multi-objective optimization problem.



Possible objectives include:



\\\[

\\max \\; Recall

\\]



\\\[

\\max \\; Groundedness

\\]



\\\[

\\max \\; CitationFaithfulness

\\]



while simultaneously:



\\\[

\\min \\; GenerationLatency

\\]



\\\[

\\min \\; ContextSize

\\]



\\\[

\\min \\; GenerationFailures

\\]



These objectives can conflict.



For example, increasing K may improve Recall while worsening latency and groundedness.



This motivates future investigation using \*\*evolutionary and multi-objective optimization algorithms\*\*.



\---



\# Experimental Artifacts



V3 introduces the following scripts:



```text

src/

├── evaluate\_topk.py

├── evaluate\_rag\_topk.py

├── prepare\_groundedness\_evaluation.py

├── analyze\_groundedness.py

├── evaluate\_rag\_stratified.py

├── prepare\_stratified\_groundedness.py

└── analyze\_stratified\_groundedness.py

```



Key result files include:



```text

results/

├── topk\_experiment.csv

├── rag\_topk\_experiment.csv

├── rag\_topk\_experiment.jsonl

├── rag\_groundedness\_evaluation\_with\_evidence.csv

├── rag\_groundedness\_evaluation\_scored.csv

├── groundedness\_summary.csv

├── rag\_stratified\_queries.csv

├── rag\_stratified\_experiment.csv

├── rag\_stratified\_experiment.jsonl

├── rag\_stratified\_groundedness\_evaluation.csv

├── rag\_stratified\_groundedness\_evaluation\_scored.csv

├── stratified\_summary\_by\_k.csv

└── stratified\_summary\_by\_group\_k.csv

```



\---



\# Limitations



The V3 results should be interpreted as experimental baselines rather than universal conclusions.



Important limitations include:



1\. The RAG quality experiments use relatively small query samples.

2\. The generator is a small local Qwen3 1.7B model.

3\. Generation was performed on CPU-only commodity hardware.

4\. Groundedness and citation-faithfulness scores are initial human-style annotations.

5\. A publication-quality evaluation should use multiple independent human evaluators and report inter-rater agreement.

6\. Results may change with different embedding models, generators, prompts, rerankers, datasets, or hardware.

7\. The stratified experiment intentionally changes the query distribution and therefore its retrieval percentages should not be interpreted as corpus-level Recall@K.

8\. SciFact contains only 5,183 documents and does not represent a large-scale Big Data workload.



\---



\# V3 Conclusions



V3 establishes several important observations:



1\. \*\*Recall increases with K.\*\*

2\. \*\*Higher Recall does not automatically produce better RAG answers.\*\*

3\. \*\*K=5 provides the strongest observed overall balance in the current experiments.\*\*

4\. \*\*Increasing K helps when it introduces genuinely useful evidence.\*\*

5\. \*\*Increasing K can hurt when relevant evidence is already available.\*\*

6\. \*\*Relevant evidence can be retrieved without being effectively used by the generator.\*\*

7\. \*\*Large contexts substantially increase local generation latency.\*\*

8\. \*\*Excess context can reduce groundedness and citation faithfulness.\*\*

9\. \*\*Fixed-K retrieval may be inferior to adaptive retrieval strategies.\*\*

10\. \*\*RAG configuration naturally forms a multi-objective optimization problem.\*\*



\---



\# Next Direction



V1 established semantic retrieval.



V2 established local end-to-end RAG.



V3 established an experimental framework for evaluating retrieval depth and downstream RAG quality.



The next stage will investigate whether these observations continue to hold as the document collection becomes substantially larger.



The project will therefore begin transitioning toward:



```text

larger corpus

&#x20;    ↓

larger embedding workload

&#x20;    ↓

larger vector index

&#x20;    ↓

scalability measurements

&#x20;    ↓

distributed processing

&#x20;    ↓

Apache Spark

```



This transition moves the project from a local RAG prototype toward its broader objective:



> \*\*Scalable, optimized AI-powered data systems for large document collections.\*\*

