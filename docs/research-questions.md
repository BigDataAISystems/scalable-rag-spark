# Research Questions

The repository begins as an engineering baseline and will evolve into an experimental platform.

1. At what corpus size does distributed preprocessing with Spark become beneficial compared with a single-machine Python pipeline?
2. How do chunking and batching strategies affect ingestion and embedding throughput?
3. How does corpus/index scale affect retrieval latency and retrieval effectiveness?
4. What trade-offs arise among embedding throughput, indexing cost, retrieval quality, and end-to-end RAG latency?
5. How can incremental updates be processed efficiently without rebuilding an entire retrieval corpus?
