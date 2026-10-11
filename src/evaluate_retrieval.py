"""Recall@k evaluation using SciFact test relevance judgments.

In addition to printing Recall@k, this script:
- measures evaluation time,
- appends each run to results/experiments.csv,
- writes a human-readable baseline summary.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import time

from load_scifact import load_scifact
from retrieve import Retriever


RESULTS_DIR = Path("results")
EXPERIMENTS_FILE = RESULTS_DIR / "experiments.csv"
BASELINE_FILE = RESULTS_DIR / "baseline_retrieval.txt"

DATASET = "SciFact"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
VECTOR_INDEX = "FAISS"


def save_experiment(k, tested, recall, elapsed):
    """Append the current experiment to experiments.csv."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = EXPERIMENTS_FILE.exists()

    fields = [
        "timestamp",
        "experiment",
        "dataset",
        "embedding_model",
        "embedding_dimension",
        "vector_index",
        "top_k",
        "queries_evaluated",
        "recall_at_k",
        "evaluation_time_sec",
    ]

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": f"scifact_dense_top{k}",
        "dataset": DATASET,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "vector_index": VECTOR_INDEX,
        "top_k": k,
        "queries_evaluated": tested,
        "recall_at_k": f"{recall:.4f}",
        "evaluation_time_sec": f"{elapsed:.2f}",
    }

    with EXPERIMENTS_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(f, fieldnames=fields)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_baseline(k, tested, recall, elapsed):
    """Write a human-readable summary of the latest baseline."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = f"""Scalable RAG with Apache Spark
V1 - SciFact Dense Retrieval Baseline

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DATASET
-------
Dataset: {DATASET}
Corpus documents: 5,183
Queries evaluated: {tested}

RETRIEVAL CONFIGURATION
-----------------------
Embedding model: {EMBEDDING_MODEL}
Embedding dimension: {EMBEDDING_DIMENSION}
Vector index: {VECTOR_INDEX}
Retrieval method: Dense semantic retrieval
Top-k: {k}

RESULT
------
Recall@{k}: {recall:.4f}

EVALUATION PERFORMANCE
----------------------
Evaluation time: {elapsed:.2f} seconds

Environment:
Local Windows machine
Python 3.11.9

NOTES
-----
Recall@k measures the proportion of evaluated queries for which
at least one SciFact-relevant document appeared among the top-k
retrieved documents.

This metric evaluates retrieval effectiveness.

It does NOT measure the accuracy or quality of the final
LLM-generated RAG answer.

The FAISS document index was built before this evaluation.
Index construction time is therefore not included in the
evaluation time.
"""

    BASELINE_FILE.write_text(summary, encoding="utf-8")


def main(k=5):

    _, queries, qrels = load_scifact()

    relevant = defaultdict(set)

    for qid, docid, score in qrels:
        if score > 0:
            relevant[qid].add(docid)

    retriever = Retriever()

    tested = 0
    hits = 0

    start_time = time.perf_counter()

    for qid, gold in relevant.items():

        if qid not in queries or not gold:
            continue

        predicted = {
            r["doc_id"]
            for r in retriever.search(
                queries[qid],
                k=k,
            )
        }

        tested += 1

        hits += int(bool(predicted & gold))

    elapsed = time.perf_counter() - start_time

    recall = hits / tested if tested else 0.0

    print()
    print("Retrieval Evaluation")
    print("--------------------")
    print(f"Dataset: {DATASET}")
    print(f"Queries evaluated: {tested}")
    print(
        f"Recall@{k} "
        f"(at least one relevant document retrieved): "
        f"{recall:.4f}"
    )
    print(f"Evaluation time: {elapsed:.2f} seconds")

    save_experiment(
        k,
        tested,
        recall,
        elapsed,
    )

    save_baseline(
        k,
        tested,
        recall,
        elapsed,
    )

    print()
    print(f"Experiment saved to: {EXPERIMENTS_FILE}")
    print(f"Baseline saved to: {BASELINE_FILE}")


if __name__ == "__main__":
    main()