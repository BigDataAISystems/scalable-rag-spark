"""
V4 - MS MARCO Scalability Evaluation

Build a MiniLM + FAISS index for one prepared MS MARCO subset
and evaluate retrieval performance and scalability.

Examples:
    python src/evaluate_msmarco_scaling.py --subset msmarco_1k
    python src/evaluate_msmarco_scaling.py --subset msmarco_25k
    python src/evaluate_msmarco_scaling.py --subset msmarco_50k
    python src/evaluate_msmarco_scaling.py --subset msmarco_100k
    python src/evaluate_msmarco_scaling.py --subset msmarco_250k

Metrics:
    - timestamp
    - total pipeline time
    - corpus loading time
    - model loading time
    - document embedding time
    - document embedding throughput
    - FAISS index build time
    - FAISS index size
    - query embedding time
    - retrieval time
    - average query latency
    - retrieval throughput
    - Recall@K
    - maximum observed process RSS memory

Outputs:
    results/v4_<subset>_scaling.json
    results/v4_scaling_experiments.csv

NOTE:
    max_observed_rss_mb is sampled at major pipeline stages.
    It is not a continuously monitored true peak-memory measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V4_DATA_DIR = ROOT / "data" / "v4"

RESULTS_DIR = ROOT / "results"

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

TOP_K = 5

BATCH_SIZE = 64


# ============================================================
# Data loading
# ============================================================

def load_jsonl(path: Path):
    """
    Load a JSONL file into memory.
    """

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def load_qrels(path: Path):
    """
    Load positive relevance judgments.
    """

    relevant = defaultdict(set)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            score = int(
                row["score"]
            )

            if score <= 0:
                continue

            qid = str(
                row["query_id"]
            )

            docid = str(
                row["doc_id"]
            )

            relevant[qid].add(
                docid
            )

    return relevant


# ============================================================
# CSV experiment logging
# ============================================================

def append_result_csv(
    path: Path,
    result: dict,
):
    """
    Append one experiment result to the cumulative V4 CSV.

    The CSV header must match the current result schema.
    """

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result.keys(),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            result
        )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Overall experiment timing
    # --------------------------------------------------------

    total_start = (
        time.perf_counter()
    )

    experiment_timestamp = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    # --------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MiniLM + FAISS scalability "
            "on a prepared MS MARCO subset."
        )
    )

    parser.add_argument(
        "--subset",
        default="msmarco_1k",
        help=(
            "Prepared V4 subset directory name, "
            "for example msmarco_1k or msmarco_100k."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of documents to retrieve.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Process memory monitoring
    # --------------------------------------------------------

    process = psutil.Process(
        os.getpid()
    )

    max_observed_rss_mb = (
        process.memory_info().rss
        / (1024 ** 2)
    )

    def update_memory():
        """
        Record the largest observed RSS at major stages.

        This is sampled memory usage, not a continuously
        monitored true process peak.
        """

        nonlocal max_observed_rss_mb

        current_rss_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        max_observed_rss_mb = max(
            max_observed_rss_mb,
            current_rss_mb,
        )

    # --------------------------------------------------------
    # Locate prepared subset
    # --------------------------------------------------------

    subset_dir = (
        V4_DATA_DIR
        / args.subset
    )

    if not subset_dir.exists():

        raise FileNotFoundError(
            f"Subset not found:\n"
            f"{subset_dir}"
        )

    corpus_file = (
        subset_dir
        / "corpus.jsonl"
    )

    queries_file = (
        subset_dir
        / "queries.jsonl"
    )

    qrels_file = (
        subset_dir
        / "qrels.tsv"
    )

    metadata_file = (
        subset_dir
        / "metadata.json"
    )

    required_files = [
        corpus_file,
        queries_file,
        qrels_file,
        metadata_file,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    print()
    print(
        "V4 - MS MARCO Scalability Evaluation"
    )

    print(
        "====================================="
    )

    print()

    print(
        f"Timestamp: {experiment_timestamp}"
    )

    print(
        f"Subset:    {args.subset}"
    )

    print(
        f"Top-K:     {args.top_k}"
    )

    # --------------------------------------------------------
    # Load corpus, queries, qrels, metadata
    # --------------------------------------------------------

    print()
    print(
        "Loading prepared dataset..."
    )

    load_start = (
        time.perf_counter()
    )

    corpus = load_jsonl(
        corpus_file
    )

    queries = load_jsonl(
        queries_file
    )

    relevant = load_qrels(
        qrels_file
    )

    with metadata_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    load_time = (
        time.perf_counter()
        - load_start
    )

    update_memory()

    print(
        f"Documents: {len(corpus):,}"
    )

    print(
        f"Queries:   {len(queries):,}"
    )

    print(
        f"Load time: {load_time:.2f}s"
    )

    # --------------------------------------------------------
    # Validate metadata
    # --------------------------------------------------------

    expected_documents = (
        metadata.get(
            "subset_size"
        )
    )

    if (
        expected_documents is not None
        and len(corpus)
        != expected_documents
    ):

        raise RuntimeError(
            "Corpus size does not match metadata. "
            f"Expected {expected_documents:,}, "
            f"loaded {len(corpus):,}."
        )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
    print(
        "Loading embedding model:"
    )

    print(
        f"  {MODEL_NAME}"
    )

    model_start = (
        time.perf_counter()
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    model_load_time = (
        time.perf_counter()
        - model_start
    )

    update_memory()

    print(
        f"Model load time: "
        f"{model_load_time:.2f}s"
    )

    # --------------------------------------------------------
    # Prepare document text
    # --------------------------------------------------------

    document_ids = [
        str(
            row["_id"]
        )
        for row in corpus
    ]

    document_texts = []

    for row in corpus:

        title = (
            row.get(
                "title",
                ""
            )
            or ""
        )

        text = (
            row.get(
                "text",
                ""
            )
            or ""
        )

        combined = (
            f"{title} {text}"
            .strip()
        )

        document_texts.append(
            combined
        )

    update_memory()

    # --------------------------------------------------------
    # Generate document embeddings
    # --------------------------------------------------------

    print()
    print(
        "Embedding documents..."
    )

    embedding_start = (
        time.perf_counter()
    )

    document_embeddings = (
        model.encode(
            document_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )

    document_embeddings = (
        np.asarray(
            document_embeddings,
            dtype="float32",
        )
    )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    update_memory()

    docs_per_sec = (
        len(corpus)
        / embedding_time
        if embedding_time > 0
        else float("inf")
    )

    dimension = (
        document_embeddings.shape[1]
    )

    print(
        f"Embedding time: "
        f"{embedding_time:.2f}s"
    )

    print(
        f"Embedding throughput: "
        f"{docs_per_sec:.2f} docs/s"
    )

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    # --------------------------------------------------------
    # Build FAISS index
    # --------------------------------------------------------

    print()
    print(
        "Building FAISS IndexFlatIP..."
    )

    index_start = (
        time.perf_counter()
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        document_embeddings
    )

    index_build_time = (
        time.perf_counter()
        - index_start
    )

    update_memory()

    print(
        f"Index vectors: "
        f"{index.ntotal:,}"
    )

    print(
        f"Index build time: "
        f"{index_build_time:.4f}s"
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    index_dir = (
        subset_dir
        / "index"
    )

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_file = (
        index_dir
        / "faiss.index"
    )

    faiss.write_index(
        index,
        str(index_file),
    )

    index_size_bytes = (
        index_file.stat().st_size
    )

    index_size_mb = (
        index_size_bytes
        / (1024 ** 2)
    )

    update_memory()

    print(
        f"Index size: "
        f"{index_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Prepare query data
    # --------------------------------------------------------

    query_ids = [
        str(
            row["_id"]
        )
        for row in queries
    ]

    query_texts = [
        (
            row.get(
                "text",
                ""
            )
            or ""
        )
        for row in queries
    ]

    # --------------------------------------------------------
    # Generate query embeddings
    # --------------------------------------------------------

    print()
    print(
        "Embedding queries..."
    )

    query_embedding_start = (
        time.perf_counter()
    )

    query_embeddings = (
        model.encode(
            query_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )

    query_embeddings = (
        np.asarray(
            query_embeddings,
            dtype="float32",
        )
    )

    query_embedding_time = (
        time.perf_counter()
        - query_embedding_start
    )

    update_memory()

    print(
        f"Query embedding time: "
        f"{query_embedding_time:.2f}s"
    )

    # --------------------------------------------------------
    # FAISS retrieval
    # --------------------------------------------------------

    print()
    print(
        "Running FAISS retrieval..."
    )

    retrieval_start = (
        time.perf_counter()
    )

    scores, indices = (
        index.search(
            query_embeddings,
            args.top_k,
        )
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    update_memory()

    query_count = len(
        query_embeddings
    )

    avg_query_latency_ms = (
        retrieval_time
        / query_count
        * 1000
        if query_count > 0
        else 0.0
    )

    queries_per_sec = (
        query_count
        / retrieval_time
        if retrieval_time > 0
        else float("inf")
    )

    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    tested = 0
    hits = 0

    for query_position, qid in enumerate(
        query_ids
    ):

        gold = relevant.get(
            qid,
            set(),
        )

        if not gold:
            continue

        predicted = set()

        for index_position in (
            indices[
                query_position
            ]
        ):

            if index_position < 0:
                continue

            predicted.add(
                document_ids[
                    index_position
                ]
            )

        tested += 1

        if predicted & gold:
            hits += 1

    recall_at_k = (
        hits / tested
        if tested > 0
        else 0.0
    )

    update_memory()

    # --------------------------------------------------------
    # Total pipeline time
    # --------------------------------------------------------

    total_pipeline_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # Build result record
    # --------------------------------------------------------

    result = {

        "timestamp":
            experiment_timestamp,

        "subset":
            args.subset,

        "documents":
            len(corpus),

        "queries_evaluated":
            tested,

        "embedding_model":
            MODEL_NAME,

        "embedding_dimension":
            dimension,

        "top_k":
            args.top_k,

        "total_pipeline_time_sec":
            round(
                total_pipeline_time,
                4,
            ),

        "load_time_sec":
            round(
                load_time,
                4,
            ),

        "model_load_time_sec":
            round(
                model_load_time,
                4,
            ),

        "document_embedding_time_sec":
            round(
                embedding_time,
                4,
            ),

        "document_embedding_docs_per_sec":
            round(
                docs_per_sec,
                4,
            ),

        "index_build_time_sec":
            round(
                index_build_time,
                6,
            ),

        "index_size_mb":
            round(
                index_size_mb,
                4,
            ),

        "query_embedding_time_sec":
            round(
                query_embedding_time,
                4,
            ),

        "retrieval_time_sec":
            round(
                retrieval_time,
                6,
            ),

        "avg_query_latency_ms":
            round(
                avg_query_latency_ms,
                6,
            ),

        "queries_per_sec":
            round(
                queries_per_sec,
                4,
            ),

        "recall_at_k":
            round(
                recall_at_k,
                4,
            ),

        "max_observed_rss_mb":
            round(
                max_observed_rss_mb,
                2,
            ),
    }

    # --------------------------------------------------------
    # Save per-run JSON
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_output = (
        RESULTS_DIR
        / f"v4_{args.subset}_scaling.json"
    )

    with json_output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save cumulative CSV
    # --------------------------------------------------------

    csv_output = (
        RESULTS_DIR
        / "v4_scaling_experiments.csv"
    )

    append_result_csv(
        csv_output,
        result,
    )

    # --------------------------------------------------------
    # Final console report
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "V4 Scaling Result"
    )

    print("=" * 60)

    print(
        f"Timestamp:              "
        f"{experiment_timestamp}"
    )

    print(
        f"Subset:                 "
        f"{args.subset}"
    )

    print(
        f"Documents:              "
        f"{len(corpus):,}"
    )

    print(
        f"Queries evaluated:      "
        f"{tested:,}"
    )

    print(
        f"Recall@{args.top_k}:              "
        f"{recall_at_k:.4f}"
    )

    print()

    print(
        f"Total pipeline time:    "
        f"{total_pipeline_time:.2f}s"
    )

    print(
        f"Load time:              "
        f"{load_time:.2f}s"
    )

    print(
        f"Model load time:        "
        f"{model_load_time:.2f}s"
    )

    print(
        f"Document embedding:     "
        f"{embedding_time:.2f}s"
    )

    print(
        f"Embedding throughput:   "
        f"{docs_per_sec:.2f} docs/s"
    )

    print(
        f"Index build time:       "
        f"{index_build_time:.4f}s"
    )

    print(
        f"Index size:             "
        f"{index_size_mb:.2f} MB"
    )

    print(
        f"Query embedding time:   "
        f"{query_embedding_time:.2f}s"
    )

    print(
        f"Retrieval time:         "
        f"{retrieval_time:.6f}s"
    )

    print(
        f"Avg query latency:      "
        f"{avg_query_latency_ms:.4f} ms"
    )

    print(
        f"Retrieval throughput:   "
        f"{queries_per_sec:.2f} queries/s"
    )

    print(
        f"Max observed RSS:       "
        f"{max_observed_rss_mb:.2f} MB"
    )

    print()

    print(
        "Per-run result:"
    )

    print(
        f"  {json_output}"
    )

    print()

    print(
        "Cumulative results:"
    )

    print(
        f"  {csv_output}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()