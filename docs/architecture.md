# Architecture

## Version 1: local baseline

```mermaid
flowchart LR
    A[BEIR SciFact] --> B[Sentence Transformer]
    B --> C[Normalized Embeddings]
    C --> D[(FAISS Index)]
    Q[User Query] --> E[Query Embedding]
    E --> D
    D --> F[Top-k Evidence]
    F --> G[Context Builder]
    G --> H[LLM]
    H --> I[Grounded Answer + Sources]
```

## Planned scale path

V1 establishes a transparent non-Spark baseline. Later versions introduce TREC-COVID, local PySpark preprocessing, MS MARCO subsets, distributed embedding/indexing experiments, and finally a distributed environment such as Databricks.
