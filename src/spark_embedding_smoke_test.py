"""
V5 - Spark Sentence Transformer Embedding Smoke Test

Purpose
-------
Verify that Sentence Transformers can run successfully inside
PySpark Python workers before performing timed scalability experiments.

This test:
    1. Loads a very small sample from the prepared MS MARCO 1K corpus.
    2. Creates a Spark DataFrame.
    3. Repartitions the documents.
    4. Loads all-MiniLM-L6-v2 inside each Python worker partition.
    5. Generates embeddings.
    6. Verifies that every document produces a 384-dimensional vector.

This is a correctness test, NOT a performance benchmark.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CORPUS_FILE = (
    ROOT
    / "data"
    / "v4"
    / "msmarco_1k"
    / "corpus.jsonl"
)

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

SAMPLE_SIZE = 40

NUM_PARTITIONS = 2

EXPECTED_DIMENSION = 384

BATCH_SIZE = 16


# ============================================================
# Local sample loader
# ============================================================

def load_sample(path: Path, limit: int):
    """
    Load a small number of MS MARCO documents.
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

            row = json.loads(line)

            doc_id = str(
                row["_id"]
            )

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

            combined_text = (
                f"{title} {text}"
                .strip()
            )

            records.append(
                (
                    doc_id,
                    combined_text,
                )
            )

            if len(records) >= limit:
                break

    return records


# ============================================================
# Spark partition embedding function
# ============================================================

def embed_partition(rows):
    """
    Embed one Spark partition.

    The model is loaded once per partition invocation rather than
    once per document.

    This is intentionally simple for the smoke test.
    """

    from sentence_transformers import SentenceTransformer

    partition_rows = list(rows)

    if not partition_rows:
        return iter([])

    model = SentenceTransformer(
        MODEL_NAME
    )

    document_ids = [
        row["doc_id"]
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

    output = []

    for doc_id, embedding in zip(
        document_ids,
        embeddings,
    ):

        output.append(
            (
                doc_id,
                embedding.tolist(),
            )
        )

    return iter(output)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "V5 - Spark Embedding Smoke Test"
    )

    print(
        "==============================="
    )

    print()

    print(
        f"Python:     "
        f"{sys.version.split()[0]}"
    )

    print(
        f"PySpark:    "
        f"{pyspark.__version__}"
    )

    print(
        f"Model:      "
        f"{MODEL_NAME}"
    )

    print(
        f"Documents:  "
        f"{SAMPLE_SIZE}"
    )

    print(
        f"Partitions: "
        f"{NUM_PARTITIONS}"
    )

    # --------------------------------------------------------
    # Validate source corpus
    # --------------------------------------------------------

    if not CORPUS_FILE.exists():

        raise FileNotFoundError(
            f"Corpus file not found:\n"
            f"{CORPUS_FILE}"
        )

    # --------------------------------------------------------
    # Load tiny local sample
    # --------------------------------------------------------

    sample = load_sample(
        CORPUS_FILE,
        SAMPLE_SIZE,
    )

    if len(sample) != SAMPLE_SIZE:

        raise RuntimeError(
            f"Expected {SAMPLE_SIZE} documents, "
            f"loaded {len(sample)}."
        )

    print()
    print(
        f"Loaded {len(sample)} "
        f"documents from MS MARCO."
    )

    # --------------------------------------------------------
    # Create Spark session
    # --------------------------------------------------------

    spark = (
        SparkSession.builder
        .appName(
            "ScalableRAGSpark-V5-EmbeddingSmokeTest"
        )
        .master("local[*]")
        .config(
            "spark.sql.shuffle.partitions",
            str(NUM_PARTITIONS),
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        "WARN"
    )

    print()
    print(
        f"Spark:      "
        f"{spark.version}"
    )

    print(
        f"Master:     "
        f"{spark.sparkContext.master}"
    )

    print(
        f"Parallelism:"
        f" {spark.sparkContext.defaultParallelism}"
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = spark.createDataFrame(
        sample,
        [
            "doc_id",
            "text",
        ],
    )

    df = df.repartition(
        NUM_PARTITIONS
    )

    print(
        f"Actual partitions: "
        f"{df.rdd.getNumPartitions()}"
    )

    # --------------------------------------------------------
    # Run distributed embedding
    # --------------------------------------------------------

    print()
    print(
        "Running Sentence Transformer "
        "inside Spark workers..."
    )

    start = time.perf_counter()

    embedded_rdd = (
        df.rdd.mapPartitions(
            embed_partition
        )
    )

    results = (
        embedded_rdd.collect()
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Validate results
    # --------------------------------------------------------

    print()
    print(
        f"Embeddings returned: "
        f"{len(results)}"
    )

    if len(results) != SAMPLE_SIZE:

        spark.stop()

        raise RuntimeError(
            "Spark embedding test failed: "
            f"expected {SAMPLE_SIZE} embeddings, "
            f"received {len(results)}."
        )

    bad_dimensions = []

    for doc_id, embedding in results:

        if (
            len(embedding)
            != EXPECTED_DIMENSION
        ):

            bad_dimensions.append(
                (
                    doc_id,
                    len(embedding),
                )
            )

    if bad_dimensions:

        spark.stop()

        raise RuntimeError(
            "Incorrect embedding dimensions: "
            f"{bad_dimensions[:5]}"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    docs_per_sec = (
        SAMPLE_SIZE
        / elapsed
        if elapsed > 0
        else 0.0
    )

    print(
        f"Embedding dimension: "
        f"{EXPECTED_DIMENSION}"
    )

    print(
        f"Elapsed time:        "
        f"{elapsed:.2f}s"
    )

    print(
        f"Throughput:          "
        f"{docs_per_sec:.2f} docs/s"
    )

    print()
    print(
        "First result:"
    )

    first_doc_id = (
        results[0][0]
    )

    first_embedding = (
        results[0][1]
    )

    print(
        f"  Document ID: "
        f"{first_doc_id}"
    )

    print(
        f"  Dimensions:  "
        f"{len(first_embedding)}"
    )

    print(
        f"  First values:"
        f" {first_embedding[:5]}"
    )

    print()
    print(
        "Spark embedding smoke test PASSED."
    )

    print(
        "Sentence Transformers can execute "
        "inside Spark Python workers."
    )

    print()
    print(
        "NOTE: This is a correctness test."
    )

    print(
        "Do not compare this timing directly "
        "with the V4 scalability measurements."
    )

    # --------------------------------------------------------
    # Stop Spark
    # --------------------------------------------------------

    spark.stop()


if __name__ == "__main__":
    main()