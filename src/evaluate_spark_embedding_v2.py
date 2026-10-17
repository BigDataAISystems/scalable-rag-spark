"""
V5 - Spark Embedding Evaluation V2

Spark-native MS MARCO ingestion with controlled PyTorch threading.

This version improves on the original V5 benchmark by allowing Spark
to read the JSONL corpus directly instead of:

    Python -> list -> createDataFrame -> Spark

Architecture:

    corpus.jsonl
         |
         v
    Spark DataFrameReader
         |
         v
    Spark transformations
         |
         v
    repartition(P)
         |
         v
    Python embedding workers
         |
         v
    SentenceTransformer / MiniLM

Example:

    python src/evaluate_spark_embedding_v2.py \
        --subset msmarco_1k \
        --partitions 2 \
        --torch-threads 2

Output:

    results/v5_spark_native_ingestion_experiments.csv

This remains a LOCAL Spark experiment. All workers execute on the
same physical machine.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V4_DATA_DIR = (
    ROOT
    / "data"
    / "v4"
)

RESULTS_DIR = (
    ROOT
    / "results"
)

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

EXPECTED_DIMENSION = 384

BATCH_SIZE = 64


# ============================================================
# V4 direct-Python baselines
# ============================================================

V4_BASELINES = {

    "msmarco_1k":
        27.0573,

    "msmarco_25k":
        825.4622,

    "msmarco_50k":
        1629.7635,

    "msmarco_100k":
        3171.4882,

    "msmarco_250k":
        10695.1256,
}


# ============================================================
# Worker-local state
# ============================================================

_WORKER_MODEL = None
_WORKER_TORCH_THREADS = None


def get_worker_model(
    torch_threads: int,
):
    """
    Configure PyTorch threading and load MiniLM once per
    Python worker process.

    With spark.python.worker.reuse=true, later tasks assigned
    to the same Python worker may reuse the loaded model.
    """

    global _WORKER_MODEL
    global _WORKER_TORCH_THREADS

    import torch

    if _WORKER_TORCH_THREADS is None:

        torch.set_num_threads(
            torch_threads
        )

        try:

            torch.set_num_interop_threads(
                1
            )

        except RuntimeError:

            # PyTorch can refuse to change this after
            # parallel execution has already started.
            pass

        _WORKER_TORCH_THREADS = (
            torch_threads
        )

    elif (
        _WORKER_TORCH_THREADS
        != torch_threads
    ):

        raise RuntimeError(
            "Spark Python worker thread configuration "
            "changed unexpectedly. "
            f"Existing={_WORKER_TORCH_THREADS}, "
            f"requested={torch_threads}."
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
# Spark partition embedding
# ============================================================

def embed_partition(
    rows,
    torch_threads: int,
):
    """
    Embed one Spark partition.

    Full vectors are deliberately NOT returned to the driver.

    Each output record contains only:

        doc_id
        embedding dimension

    This is sufficient to force computation and validate
    correctness while avoiding unnecessary driver traffic.
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
        or ""
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
            int(
                len(embedding)
            ),
        )
        for doc_id, embedding
        in zip(
            document_ids,
            embeddings,
        )
    )


# ============================================================
# CSV logging
# ============================================================

def append_csv(
    path: Path,
    result: dict,
):
    """
    Append one benchmark result.
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
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Spark-native MS MARCO ingestion "
            "and Sentence Transformer embedding."
        )
    )

    parser.add_argument(
        "--subset",
        default="msmarco_1k",
        help=(
            "Prepared V4 subset, for example "
            "msmarco_1k or msmarco_50k."
        ),
    )

    parser.add_argument(
        "--partitions",
        type=int,
        required=True,
        help=(
            "Number of local Spark task slots and "
            "target DataFrame partitions."
        ),
    )

    parser.add_argument(
        "--torch-threads",
        type=int,
        required=True,
        help=(
            "PyTorch intra-op CPU threads "
            "per Spark Python worker."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

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
            "No V4 baseline registered for "
            f"{args.subset}."
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

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

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        RESULTS_DIR
        / "v5_spark_native_ingestion_experiments.csv"
    )

    # --------------------------------------------------------
    # Experiment metadata
    # --------------------------------------------------------

    timestamp = (
        datetime.now().isoformat(
            timespec="seconds"
        )
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

    master = (
        f"local[{args.partitions}]"
    )

    print()
    print(
        "V5 - Spark Native-Ingestion Embedding"
    )

    print(
        "====================================="
    )

    print()

    print(
        f"Timestamp:            "
        f"{timestamp}"
    )

    print(
        f"Python:               "
        f"{sys.version.split()[0]}"
    )

    print(
        f"PySpark:              "
        f"{pyspark.__version__}"
    )

    print(
        f"Subset:               "
        f"{args.subset}"
    )

    print(
        f"Corpus:               "
        f"{corpus_file}"
    )

    print(
        f"Spark partitions:     "
        f"{args.partitions}"
    )

    print(
        f"Torch threads/worker: "
        f"{args.torch_threads}"
    )

    print(
        f"Nominal thread budget:"
        f" {nominal_thread_budget}"
    )

    print(
        f"Batch size:           "
        f"{BATCH_SIZE}"
    )

    print(
        f"V4 baseline:          "
        f"{baseline_time:.2f}s"
    )

    # --------------------------------------------------------
    # Driver RSS
    # --------------------------------------------------------

    driver_process = (
        psutil.Process(
            os.getpid()
        )
    )

    max_driver_rss_mb = (
        driver_process
        .memory_info()
        .rss
        / (1024 ** 2)
    )

    def update_driver_memory():

        nonlocal max_driver_rss_mb

        current_mb = (
            driver_process
            .memory_info()
            .rss
            / (1024 ** 2)
        )

        max_driver_rss_mb = max(
            max_driver_rss_mb,
            current_mb,
        )

    # --------------------------------------------------------
    # Start Spark
    # --------------------------------------------------------

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
            "ScalableRAGSpark-V5-NativeIngestion"
        )
        .master(
            master
        )
        .config(
            "spark.sql.shuffle.partitions",
            str(args.partitions),
        )
        .config(
            "spark.default.parallelism",
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

    update_driver_memory()

    print(
        f"Spark version:        "
        f"{spark.version}"
    )

    print(
        f"Spark master:         "
        f"{spark.sparkContext.master}"
    )

    print(
        f"Default parallelism:  "
        f"{spark.sparkContext.defaultParallelism}"
    )

    print(
        f"Spark startup:        "
        f"{spark_startup_time:.2f}s"
    )

    # --------------------------------------------------------
    # Spark-native corpus ingestion
    # --------------------------------------------------------

    print()
    print(
        "Reading corpus directly with Spark..."
    )

    read_start = (
        time.perf_counter()
    )

    # Explicit schema avoids unnecessary type inference and
    # ensures consistent field types across experiments.

    corpus_schema = T.StructType([
        T.StructField(
            "_id",
            T.StringType(),
            False,
        ),
        T.StructField(
            "title",
            T.StringType(),
            True,
        ),
        T.StructField(
            "text",
            T.StringType(),
            True,
        ),
    ])

    raw_df = (
        spark.read
        .schema(
            corpus_schema
        )
        .json(
            str(corpus_file)
        )
    )

    # Build the exact same logical document text used by V4:
    #
    #     title + " " + text
    #
    # Empty/null titles are handled safely.

    df = (
        raw_df
        .select(
            F.col("_id")
            .cast("string")
            .alias("doc_id"),

            F.trim(
                F.concat_ws(
                    " ",
                    F.coalesce(
                        F.col("title"),
                        F.lit(""),
                    ),
                    F.coalesce(
                        F.col("text"),
                        F.lit(""),
                    ),
                )
            )
            .alias("text"),
        )
    )

    # Explicitly repartition to the experiment configuration.

    df = df.repartition(
        args.partitions
    )

    # Count forces Spark to execute the file read and partitioning.
    #
    # This is intentionally measured separately from embedding.

    document_count = (
        df.count()
    )

    spark_read_time = (
        time.perf_counter()
        - read_start
    )

    actual_partitions = (
        df.rdd.getNumPartitions()
    )

    update_driver_memory()

    print(
        f"Documents:            "
        f"{document_count:,}"
    )

    print(
        f"Actual partitions:    "
        f"{actual_partitions}"
    )

    print(
        f"Spark read/count time:"
        f" {spark_read_time:.2f}s"
    )

    # --------------------------------------------------------
    # Timed embedding workload
    # --------------------------------------------------------

    print()
    print(
        "Running Spark-native embedding workload..."
    )

    print(
        "Worker/model initialization is included "
        "in embedding time."
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

    # Force execution without returning all embeddings.

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

    spark_embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    update_driver_memory()

    embedded_count = (
        validation[0]
    )

    correct_dimensions = (
        validation[1]
    )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    if (
        embedded_count
        != document_count
    ):

        spark.stop()

        raise RuntimeError(
            "Spark embedding count mismatch. "
            f"Expected {document_count:,}, "
            f"got {embedded_count:,}."
        )

    if (
        correct_dimensions
        != document_count
    ):

        spark.stop()

        raise RuntimeError(
            "One or more embeddings had "
            "incorrect dimensions."
        )

    # --------------------------------------------------------
    # Performance metrics
    # --------------------------------------------------------

    docs_per_sec = (
        document_count
        / spark_embedding_time
        if spark_embedding_time > 0
        else 0.0
    )

    speedup = (
        baseline_time
        / spark_embedding_time
        if spark_embedding_time > 0
        else 0.0
    )

    partition_efficiency = (
        speedup
        / args.partitions
    )

    thread_budget_efficiency = (
        speedup
        / nominal_thread_budget
    )

    difference_vs_v4 = (
        spark_embedding_time
        - baseline_time
    )

    # Spark-native end-to-end time excludes Spark JVM startup,
    # but includes Spark corpus read/count plus embedding.

    spark_pipeline_time = (
        spark_read_time
        + spark_embedding_time
    )

    pipeline_speedup_vs_v4 = (
        baseline_time
        / spark_pipeline_time
        if spark_pipeline_time > 0
        else 0.0
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
    # Result
    # --------------------------------------------------------

    result = {

        "timestamp":
            timestamp,

        "architecture":
            "spark_native_jsonl",

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

        "spark_startup_time_sec":
            round(
                spark_startup_time,
                4,
            ),

        "spark_read_count_time_sec":
            round(
                spark_read_time,
                4,
            ),

        "spark_embedding_time_sec":
            round(
                spark_embedding_time,
                4,
            ),

        "spark_pipeline_time_sec":
            round(
                spark_pipeline_time,
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

        "embedding_speedup_vs_v4":
            round(
                speedup,
                4,
            ),

        "pipeline_speedup_vs_v4":
            round(
                pipeline_speedup_vs_v4,
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

        "embedding_time_difference_vs_v4_sec":
            round(
                difference_vs_v4,
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

    append_csv(
        output_file,
        result,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 68)

    print(
        "V5 Spark Native-Ingestion Result"
    )

    print("=" * 68)

    print(
        f"Documents:                 "
        f"{document_count:,}"
    )

    print(
        f"Spark partitions:          "
        f"{args.partitions}"
    )

    print(
        f"Torch threads/worker:      "
        f"{args.torch_threads}"
    )

    print(
        f"Nominal thread budget:     "
        f"{nominal_thread_budget}"
    )

    print()

    print(
        f"Spark read/count time:     "
        f"{spark_read_time:.2f}s"
    )

    print(
        f"Spark embedding time:      "
        f"{spark_embedding_time:.2f}s"
    )

    print(
        f"Spark pipeline time:       "
        f"{spark_pipeline_time:.2f}s"
    )

    print(
        f"Embedding throughput:      "
        f"{docs_per_sec:.2f} docs/s"
    )

    print()

    print(
        f"V4 baseline embedding:     "
        f"{baseline_time:.2f}s"
    )

    print(
        f"Embedding speedup vs V4:   "
        f"{speedup:.3f}x"
    )

    print(
        f"Pipeline speedup vs V4:    "
        f"{pipeline_speedup_vs_v4:.3f}x"
    )

    print(
        f"Partition efficiency:      "
        f"{partition_efficiency:.3f}"
    )

    print(
        f"Thread-budget efficiency:  "
        f"{thread_budget_efficiency:.3f}"
    )

    print(
        f"Embedding difference:      "
        f"{difference_vs_v4:+.2f}s"
    )

    print()

    print(
        f"Validated embeddings:      "
        f"{embedded_count:,}"
    )

    print(
        f"Correct dimensions:        "
        f"{correct_dimensions:,}"
    )

    print(
        f"Max driver RSS:            "
        f"{max_driver_rss_mb:.2f} MB"
    )

    print()

    print(
        "Result appended to:"
    )

    print(
        f"  {output_file}"
    )

    print("=" * 68)


if __name__ == "__main__":
    main()