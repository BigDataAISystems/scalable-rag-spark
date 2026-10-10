# Data

Version 1 downloads the **BEIR SciFact** corpus, queries, and relevance judgments through the Hugging Face `datasets` library at runtime.

Large generated artifacts are intentionally not committed. The FAISS index is written to `data/index/`, which is ignored by Git.

Later milestones will add TREC-COVID and MS MARCO subsets/full collections for scale experiments.
