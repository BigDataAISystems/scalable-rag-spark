"""
V5 - Controlled Spark + PyTorch Embedding Evaluation

Benchmarks Sentence Transformer document embedding with Apache Spark
while explicitly controlling PyTorch CPU threads per Spark worker.

Example:

    python src/evaluate_spark_embedding.py \
        --subset msmarco_1k \
        --partitions 2 \
        --torch-threads 2

Purpose
-------
Separate two forms of parallelism:

    1. Spark worker/task parallelism
    2. PyTorch CPU parallelism inside each worker

This helps identify CPU oversubscription and determine a reasonable
configuration before running larger MS MARCO workloads.

Output:
    results/v5_spark_thread_experiments.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import pyspark
from pyspark.sql import SparkSession


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

EXPECTED_DIMENSION = 384
BATCH_SIZE = 64


# ============================================================
# V4 measured direct-Python baselines
# ============================================================

V4_BASELINES = {
    "msmarco_1k": 27.0573,
    "msmarco_25k": 825.4622,
    "msmarco_50k": 1629.7635,
    "msmarco_100k": 3171.4882,
    "msmarco_250k": 10695.1256,
}


# ============================================================
# Worker-local state
# ============================================================

_WORKER_MODEL = None
_WORKER_TORCH_THREADS = None


def get_worker_model(torch_threads: int):
    """
    Configure PyTorch CPU threading and load MiniLM once per
    Python worker process.

    Worker reuse allows later tasks executed by the same Python
    worker to reuse the loaded model.
    """

    global _WORKER_MODEL
    global _WORKER_TORCH_THREADS

    import torch

    if _WORKER_TORCH_THREADS is None:

        torch.set_num_threads(
            torch_threads
        )

        # Inter-op threading is a separate PyTorch thread pool.
        # Keep it at one to reduce nested CPU oversubscription.
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch may refuse to change this after parallel
            # work has already started in a reused worker.
            pass

        _WORKER_TORCH_THREADS = (
            torch_threads
        )

    elif (
        _WORKER_TORCH_THREADS
        != torch_threads
    ):

        raise RuntimeError(
            "A reused Spark Python worker was configured "
            f"for {_WORKER_TORCH_THREADS} PyTorch threads "
            f"but received a task requesting {torch_threads}."
        )

    if _WORKER_MODEL is None:

        from sentence_transformers import (
            SentenceTransformer,
        )

        _WORKER_MODEL = (
            SentenceTransformer(
                MODEL_NAME
            )
        )

    return _WORKER_MODEL


# ============================================================
# Partition embedding
# ============================================================

def embed_partition(
    rows,
    torch_threads: int,
):
    """
    Embed one Spark partition.

    Only lightweight validation information is returned to the
    driver; full embeddings are not collected.
    """

    partition_rows = list(rows)

    if not partition_rows:
        return iter([])

    model = get_worker_model(
        torch_threads
    )

    document_ids = [
        str(row["doc_id"])
        for row in partition_rows
    ]

    texts = [
        row["text"]
        for row in partition_rows
    ]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return iter(
        (
            doc_id,
            int(len(embedding)),
        )
        for doc_id, embedding
        in zip(
            document_ids,
            embeddings,
        )
    )


# ============================================================
# Corpus loading
# ============================================================

def load_corpus(path: Path):

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            doc_id = str(
                row["_id"]
            )

            title = (
                row.get("title", "")
                or ""
            )

            text = (
                row.get("text", "")
                or ""
            )

            records.append(
                (
                    doc_id,
                    f"{title} {text}".strip(),
                )
            )

    return records


# ============================================================
# CSV logging
# ============================================================

def append_csv(
    path: Path,
    result: dict,
):

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

        writer.writerow(result)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Spark embedding while "
            "controlling PyTorch worker threads."
        )
    )

    parser.add_argument(
        "--subset",
        default="msmarco_1k",
    )

    parser.add_argument(
        "--partitions",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--torch-threads",
        type=int,
        required=True,
        help=(
            "Number of PyTorch intra-op CPU threads "
            "per Spark Python worker."
        ),
    )

    args = parser.parse_args()

    if args.partitions < 1:
        raise ValueError(
            "--partitions must be >= 1"
        )

    if args.torch_threads < 1:
        raise ValueError(
            "--torch-threads must be >= 1"
        )

    if args.subset not in V4_BASELINES:
        raise ValueError(
            f"No V4 baseline registered for "
            f"{args.subset}."
        )

    timestamp = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    subset_dir = (
        V4_DATA_DIR
        / args.subset
    )

    corpus_file = (
        subset_dir
        / "corpus.jsonl"
    )

    if not corpus_file.exists():
        raise FileNotFoundError(
            f"Corpus not found:\n"
            f"{corpus_file}"
        )

    baseline_time = (
        V4_BASELINES[
            args.subset
        ]
    )

    nominal_thread_budget = (
        args.partitions
        * args.torch_threads
    )

    print()
    print(
        "V5 - Controlled Spark Embedding"
    )
    print(
        "==============================="
    )
    print()

    print(
        f"Timestamp:          {timestamp}"
    )
    print(
        f"Python:             "
        f"{sys.version.split()[0]}"
    )
    print(
        f"PySpark:            "
        f"{pyspark.__version__}"
    )
    print(
        f"Subset:             "
        f"{args.subset}"
    )
    print(
        f"Spark partitions:   "
        f"{args.partitions}"
    )
    print(
        f"Torch threads/worker:"
        f" {args.torch_threads}"
    )
    print(
        f"Nominal thread budget:"
        f" {nominal_thread_budget}"
    )
    print(
        f"Batch size:         "
        f"{BATCH_SIZE}"
    )
    print(
        f"V4 baseline:        "
        f"{baseline_time:.2f}s"
    )

    # --------------------------------------------------------
    # Driver memory
    # --------------------------------------------------------

    process = psutil.Process(
        os.getpid()
    )

    max_driver_rss_mb = (
        process.memory_info().rss
        / (1024 ** 2)
    )

    def update_memory():

        nonlocal max_driver_rss_mb

        rss = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        max_driver_rss_mb = max(
            max_driver_rss_mb,
            rss,
        )

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    print()
    print("Loading corpus...")

    load_start = (
        time.perf_counter()
    )

    corpus = load_corpus(
        corpus_file
    )

    load_time = (
        time.perf_counter()
        - load_start
    )

    update_memory()

    document_count = len(
        corpus
    )

    print(
        f"Documents:          "
        f"{document_count:,}"
    )
    print(
        f"Load time:          "
        f"{load_time:.2f}s"
    )

    # --------------------------------------------------------
    # Start Spark
    # --------------------------------------------------------

    master = (
        f"local[{args.partitions}]"
    )

    print()
    print(
        f"Starting Spark with "
        f"{master}..."
    )

    spark_start = (
        time.perf_counter()
    )

    spark = (
        SparkSession.builder
        .appName(
            "ScalableRAGSpark-V5-ControlledThreads"
        )
        .master(master)
        .config(
            "spark.sql.shuffle.partitions",
            str(args.partitions),
        )
        .config(
            "spark.python.worker.reuse",
            "true",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        "WARN"
    )

    spark_startup_time = (
        time.perf_counter()
        - spark_start
    )

    update_memory()

    print(
        f"Spark version:      "
        f"{spark.version}"
    )
    print(
        f"Spark master:       "
        f"{spark.sparkContext.master}"
    )
    print(
        f"Parallelism:        "
        f"{spark.sparkContext.defaultParallelism}"
    )
    print(
        f"Spark startup:      "
        f"{spark_startup_time:.2f}s"
    )

    # --------------------------------------------------------
    # DataFrame / partitions
    # --------------------------------------------------------

    df = spark.createDataFrame(
        corpus,
        [
            "doc_id",
            "text",
        ],
    )

    df = df.repartition(
        args.partitions
    )

    actual_partitions = (
        df.rdd.getNumPartitions()
    )

    print(
        f"Actual partitions:  "
        f"{actual_partitions}"
    )

    # --------------------------------------------------------
    # Timed embedding
    # --------------------------------------------------------

    print()
    print(
        "Running controlled Spark "
        "embedding workload..."
    )

    print(
        "Worker/model initialization "
        "is included in embedding time."
    )

    embedding_start = (
        time.perf_counter()
    )

    validation_rdd = (
        df.rdd.mapPartitions(
            lambda rows:
                embed_partition(
                    rows,
                    args.torch_threads,
                )
        )
    )

    validation = (
        validation_rdd
        .map(
            lambda row: (
                1,
                1
                if row[1]
                == EXPECTED_DIMENSION
                else 0,
            )
        )
        .reduce(
            lambda a, b: (
                a[0] + b[0],
                a[1] + b[1],
            )
        )
    )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    update_memory()

    embedded_count = (
        validation[0]
    )

    correct_dimensions = (
        validation[1]
    )

    if embedded_count != document_count:

        spark.stop()

        raise RuntimeError(
            "Embedding count mismatch."
        )

    if (
        correct_dimensions
        != document_count
    ):

        spark.stop()

        raise RuntimeError(
            "Incorrect embedding dimension."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    docs_per_sec = (
        document_count
        / embedding_time
    )

    speedup = (
        baseline_time
        / embedding_time
    )

    partition_efficiency = (
        speedup
        / args.partitions
    )

    thread_budget_efficiency = (
        speedup
        / nominal_thread_budget
    )

    difference_sec = (
        embedding_time
        - baseline_time
    )

    # --------------------------------------------------------
    # Stop Spark
    # --------------------------------------------------------

    stop_start = (
        time.perf_counter()
    )

    spark.stop()

    spark_stop_time = (
        time.perf_counter()
        - stop_start
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "timestamp":
            timestamp,

        "subset":
            args.subset,

        "documents":
            document_count,

        "partitions":
            args.partitions,

        "torch_threads_per_worker":
            args.torch_threads,

        "nominal_thread_budget":
            nominal_thread_budget,

        "actual_partitions":
            actual_partitions,

        "spark_master":
            master,

        "spark_version":
            pyspark.__version__,

        "embedding_model":
            MODEL_NAME,

        "embedding_dimension":
            EXPECTED_DIMENSION,

        "batch_size":
            BATCH_SIZE,

        "corpus_load_time_sec":
            round(
                load_time,
                4,
            ),

        "spark_startup_time_sec":
            round(
                spark_startup_time,
                4,
            ),

        "spark_embedding_time_sec":
            round(
                embedding_time,
                4,
            ),

        "spark_docs_per_sec":
            round(
                docs_per_sec,
                4,
            ),

        "v4_baseline_embedding_time_sec":
            round(
                baseline_time,
                4,
            ),

        "speedup_vs_v4":
            round(
                speedup,
                4,
            ),

        "partition_efficiency":
            round(
                partition_efficiency,
                4,
            ),

        "thread_budget_efficiency":
            round(
                thread_budget_efficiency,
                4,
            ),

        "time_difference_vs_v4_sec":
            round(
                difference_sec,
                4,
            ),

        "max_driver_rss_mb":
            round(
                max_driver_rss_mb,
                2,
            ),

        "spark_stop_time_sec":
            round(
                spark_stop_time,
                4,
            ),

        "worker_reuse":
            True,

        "status":
            "SUCCESS",
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        RESULTS_DIR
        / "v5_spark_thread_experiments.csv"
    )

    append_csv(
        output_file,
        result,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print(
        "V5 Controlled Spark Embedding Result"
    )
    print("=" * 64)

    print(
        f"Documents:                "
        f"{document_count:,}"
    )
    print(
        f"Spark partitions:         "
        f"{args.partitions}"
    )
    print(
        f"Torch threads/worker:     "
        f"{args.torch_threads}"
    )
    print(
        f"Nominal thread budget:    "
        f"{nominal_thread_budget}"
    )

    print()

    print(
        f"Spark embedding time:     "
        f"{embedding_time:.2f}s"
    )
    print(
        f"Spark throughput:         "
        f"{docs_per_sec:.2f} docs/s"
    )

    print()

    print(
        f"V4 baseline:              "
        f"{baseline_time:.2f}s"
    )
    print(
        f"Speedup vs V4:            "
        f"{speedup:.3f}x"
    )
    print(
        f"Partition efficiency:     "
        f"{partition_efficiency:.3f}"
    )
    print(
        f"Thread-budget efficiency: "
        f"{thread_budget_efficiency:.3f}"
    )
    print(
        f"Time difference vs V4:    "
        f"{difference_sec:+.2f}s"
    )

    print()

    print(
        f"Validated embeddings:     "
        f"{embedded_count:,}"
    )
    print(
        f"Correct dimensions:       "
        f"{correct_dimensions:,}"
    )
    print(
        f"Max driver RSS:           "
        f"{max_driver_rss_mb:.2f} MB"
    )

    print()

    print(
        "Result appended to:"
    )
    print(
        f"  {output_file}"
    )

    print("=" * 64)


if __name__ == "__main__":
    main()