"""
V4 - MS MARCO Scalability Analysis

Combines measured V4 scaling experiments and analyzes how corpus size
affects:

    - document embedding time
    - embedding throughput
    - FAISS index size
    - retrieval latency
    - retrieval throughput
    - Recall@K
    - observed process memory

Measured experiments:
    1K
    25K
    50K
    100K

Projected corpus sizes:
    250K
    500K
    1M

IMPORTANT:
Projected values are extrapolations from measured scaling behavior.
They are NOT experimental measurements.

Inputs:
    results/v4_scaling_experiments_pre100k.csv
    results/v4_scaling_experiments.csv

Outputs:
    results/v4_scaling_summary.csv
    results/v4_scaling_projections.csv

    results/v4_embedding_time.png
    results/v4_retrieval_latency.png
    results/v4_index_size.png
    results/v4_memory.png
    results/v4_recall_at_5.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

OLD_RESULTS_FILE = (
    RESULTS_DIR
    / "v4_scaling_experiments_pre100k.csv"
)

NEW_RESULTS_FILE = (
    RESULTS_DIR
    / "v4_scaling_experiments.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "v4_scaling_summary.csv"
)

PROJECTIONS_FILE = (
    RESULTS_DIR
    / "v4_scaling_projections.csv"
)


# ============================================================
# Projection configuration
# ============================================================

PROJECTION_SIZES = [
    250_000,
    500_000,
    1_000_000,
]


# ============================================================
# CSV utilities
# ============================================================

def read_csv(path: Path):
    """
    Read a CSV file and return rows as dictionaries.
    """

    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def to_int(value, default=0):
    """
    Safely convert a value to int.
    """

    if value is None:
        return default

    value = str(value).strip()

    if value == "":
        return default

    return int(float(value))


def to_float(value, default=0.0):
    """
    Safely convert a value to float.
    """

    if value is None:
        return default

    value = str(value).strip()

    if value == "":
        return default

    return float(value)


def normalize_row(row):
    """
    Normalize old and new V4 CSV schemas.

    The pre-100K file does not contain:
        timestamp
        total_pipeline_time_sec

    These are left blank for older measurements.
    """

    return {

        "timestamp":
            row.get(
                "timestamp",
                "",
            ),

        "subset":
            row.get(
                "subset",
                "",
            ),

        "documents":
            to_int(
                row.get("documents")
            ),

        "queries_evaluated":
            to_int(
                row.get(
                    "queries_evaluated"
                )
            ),

        "embedding_model":
            row.get(
                "embedding_model",
                "",
            ),

        "embedding_dimension":
            to_int(
                row.get(
                    "embedding_dimension"
                )
            ),

        "top_k":
            to_int(
                row.get("top_k")
            ),

        "total_pipeline_time_sec":
            (
                to_float(
                    row.get(
                        "total_pipeline_time_sec"
                    )
                )
                if str(
                    row.get(
                        "total_pipeline_time_sec",
                        "",
                    )
                ).strip()
                else ""
            ),

        "load_time_sec":
            to_float(
                row.get(
                    "load_time_sec"
                )
            ),

        "model_load_time_sec":
            to_float(
                row.get(
                    "model_load_time_sec"
                )
            ),

        "document_embedding_time_sec":
            to_float(
                row.get(
                    "document_embedding_time_sec"
                )
            ),

        "document_embedding_docs_per_sec":
            to_float(
                row.get(
                    "document_embedding_docs_per_sec"
                )
            ),

        "index_build_time_sec":
            to_float(
                row.get(
                    "index_build_time_sec"
                )
            ),

        "index_size_mb":
            to_float(
                row.get(
                    "index_size_mb"
                )
            ),

        "query_embedding_time_sec":
            to_float(
                row.get(
                    "query_embedding_time_sec"
                )
            ),

        "retrieval_time_sec":
            to_float(
                row.get(
                    "retrieval_time_sec"
                )
            ),

        "avg_query_latency_ms":
            to_float(
                row.get(
                    "avg_query_latency_ms"
                )
            ),

        "queries_per_sec":
            to_float(
                row.get(
                    "queries_per_sec"
                )
            ),

        "recall_at_k":
            to_float(
                row.get(
                    "recall_at_k"
                )
            ),

        "max_observed_rss_mb":
            to_float(
                row.get(
                    "max_observed_rss_mb"
                )
            ),
    }


# ============================================================
# Measured result preparation
# ============================================================

def load_measured_results():
    """
    Combine the pre-100K and current V4 experiment files.
    """

    rows = []

    for source in [
        OLD_RESULTS_FILE,
        NEW_RESULTS_FILE,
    ]:

        for row in read_csv(source):

            rows.append(
                normalize_row(row)
            )

    if not rows:

        raise RuntimeError(
            "No V4 scaling experiment results found."
        )

    # --------------------------------------------------------
    # Remove duplicate corpus-size measurements if present.
    #
    # Keep the last occurrence so the newer run wins.
    # --------------------------------------------------------

    by_documents = {}

    for row in rows:

        by_documents[
            row["documents"]
        ] = row

    measured = list(
        by_documents.values()
    )

    measured.sort(
        key=lambda row:
            row["documents"]
    )

    return measured


# ============================================================
# Scaling metrics
# ============================================================

def calculate_scaling_metrics(rows):
    """
    Add useful derived metrics to measured results.
    """

    enriched = []

    previous = None

    for row in rows:

        current = dict(row)

        documents = (
            current["documents"]
        )

        embedding_time = (
            current[
                "document_embedding_time_sec"
            ]
        )

        index_size = (
            current[
                "index_size_mb"
            ]
        )

        latency = (
            current[
                "avg_query_latency_ms"
            ]
        )

        memory = (
            current[
                "max_observed_rss_mb"
            ]
        )

        # ----------------------------------------------------
        # Per-document metrics
        # ----------------------------------------------------

        current[
            "embedding_sec_per_1k_docs"
        ] = (
            embedding_time
            / documents
            * 1000
            if documents > 0
            else 0.0
        )

        current[
            "index_mb_per_1k_docs"
        ] = (
            index_size
            / documents
            * 1000
            if documents > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Scaling relative to previous measured point
        # ----------------------------------------------------

        if previous is None:

            current[
                "corpus_growth_factor"
            ] = ""

            current[
                "embedding_time_growth_factor"
            ] = ""

            current[
                "latency_growth_factor"
            ] = ""

            current[
                "index_size_growth_factor"
            ] = ""

            current[
                "memory_growth_factor"
            ] = ""

            current[
                "recall_change"
            ] = ""

        else:

            previous_docs = (
                previous["documents"]
            )

            current[
                "corpus_growth_factor"
            ] = (
                documents
                / previous_docs
            )

            previous_embedding = (
                previous[
                    "document_embedding_time_sec"
                ]
            )

            current[
                "embedding_time_growth_factor"
            ] = (
                embedding_time
                / previous_embedding
                if previous_embedding > 0
                else ""
            )

            previous_latency = (
                previous[
                    "avg_query_latency_ms"
                ]
            )

            current[
                "latency_growth_factor"
            ] = (
                latency
                / previous_latency
                if previous_latency > 0
                else ""
            )

            previous_index = (
                previous[
                    "index_size_mb"
                ]
            )

            current[
                "index_size_growth_factor"
            ] = (
                index_size
                / previous_index
                if previous_index > 0
                else ""
            )

            previous_memory = (
                previous[
                    "max_observed_rss_mb"
                ]
            )

            current[
                "memory_growth_factor"
            ] = (
                memory
                / previous_memory
                if previous_memory > 0
                else ""
            )

            current[
                "recall_change"
            ] = (
                current["recall_at_k"]
                - previous["recall_at_k"]
            )

        enriched.append(
            current
        )

        previous = current

    return enriched


# ============================================================
# Linear projections
# ============================================================

def linear_fit(
    x,
    y,
):
    """
    Fit y = a*x + b.

    Returns:
        slope
        intercept
        R^2
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    predicted = (
        slope * x
        + intercept
    )

    ss_res = np.sum(
        (y - predicted) ** 2
    )

    ss_tot = np.sum(
        (
            y
            - np.mean(y)
        ) ** 2
    )

    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = (
            1
            - ss_res
            / ss_tot
        )

    return (
        float(slope),
        float(intercept),
        float(r_squared),
    )


def project_scaling(rows):
    """
    Project selected scalability metrics to larger corpus sizes.

    We intentionally do NOT project Recall@5 because retrieval-quality
    degradation does not have to remain linear with corpus size.

    Projected metrics:
        embedding time
        index size
        retrieval latency
        observed RSS

    These projections are exploratory planning estimates only.
    """

    documents = [
        row["documents"]
        for row in rows
    ]

    embedding_times = [
        row[
            "document_embedding_time_sec"
        ]
        for row in rows
    ]

    index_sizes = [
        row[
            "index_size_mb"
        ]
        for row in rows
    ]

    latencies = [
        row[
            "avg_query_latency_ms"
        ]
        for row in rows
    ]

    memory_values = [
        row[
            "max_observed_rss_mb"
        ]
        for row in rows
    ]

    (
        embedding_slope,
        embedding_intercept,
        embedding_r2,
    ) = linear_fit(
        documents,
        embedding_times,
    )

    (
        index_slope,
        index_intercept,
        index_r2,
    ) = linear_fit(
        documents,
        index_sizes,
    )

    (
        latency_slope,
        latency_intercept,
        latency_r2,
    ) = linear_fit(
        documents,
        latencies,
    )

    (
        memory_slope,
        memory_intercept,
        memory_r2,
    ) = linear_fit(
        documents,
        memory_values,
    )

    projections = []

    for size in PROJECTION_SIZES:

        embedding_sec = max(
            0.0,
            (
                embedding_slope
                * size
                + embedding_intercept
            ),
        )

        index_mb = max(
            0.0,
            (
                index_slope
                * size
                + index_intercept
            ),
        )

        latency_ms = max(
            0.0,
            (
                latency_slope
                * size
                + latency_intercept
            ),
        )

        memory_mb = max(
            0.0,
            (
                memory_slope
                * size
                + memory_intercept
            ),
        )

        projections.append({

            "documents":
                size,

            "projected_embedding_time_sec":
                round(
                    embedding_sec,
                    2,
                ),

            "projected_embedding_time_min":
                round(
                    embedding_sec / 60,
                    2,
                ),

            "projected_embedding_time_hr":
                round(
                    embedding_sec / 3600,
                    2,
                ),

            "projected_index_size_mb":
                round(
                    index_mb,
                    2,
                ),

            "projected_index_size_gb":
                round(
                    index_mb / 1024,
                    3,
                ),

            "projected_avg_query_latency_ms":
                round(
                    latency_ms,
                    3,
                ),

            "projected_max_observed_rss_mb":
                round(
                    memory_mb,
                    2,
                ),

            "embedding_linear_fit_r2":
                round(
                    embedding_r2,
                    6,
                ),

            "index_linear_fit_r2":
                round(
                    index_r2,
                    6,
                ),

            "latency_linear_fit_r2":
                round(
                    latency_r2,
                    6,
                ),

            "memory_linear_fit_r2":
                round(
                    memory_r2,
                    6,
                ),

            "projection_status":
                "PROJECTED_NOT_MEASURED",
        })

    return projections


# ============================================================
# CSV output
# ============================================================

def write_csv(
    path,
    rows,
):
    """
    Write dictionaries to CSV.
    """

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Plotting
# ============================================================

def save_line_plot(
    x,
    y,
    xlabel,
    ylabel,
    title,
    output_path,
):
    """
    Save one measured-data line plot.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x,
        y,
        marker="o",
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


def create_figures(rows):
    """
    Create V4 measured scaling figures.
    """

    documents = [
        row["documents"]
        for row in rows
    ]

    # --------------------------------------------------------
    # Embedding time
    # --------------------------------------------------------

    embedding_minutes = [
        row[
            "document_embedding_time_sec"
        ]
        / 60
        for row in rows
    ]

    save_line_plot(
        documents,
        embedding_minutes,
        "Corpus size (documents)",
        "Embedding time (minutes)",
        "V4: Corpus Size vs Document Embedding Time",
        RESULTS_DIR
        / "v4_embedding_time.png",
    )

    # --------------------------------------------------------
    # Retrieval latency
    # --------------------------------------------------------

    latency = [
        row[
            "avg_query_latency_ms"
        ]
        for row in rows
    ]

    save_line_plot(
        documents,
        latency,
        "Corpus size (documents)",
        "Average query latency (ms)",
        "V4: Corpus Size vs FAISS Retrieval Latency",
        RESULTS_DIR
        / "v4_retrieval_latency.png",
    )

    # --------------------------------------------------------
    # Index size
    # --------------------------------------------------------

    index_size = [
        row[
            "index_size_mb"
        ]
        for row in rows
    ]

    save_line_plot(
        documents,
        index_size,
        "Corpus size (documents)",
        "FAISS index size (MB)",
        "V4: Corpus Size vs FAISS Index Size",
        RESULTS_DIR
        / "v4_index_size.png",
    )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    memory = [
        row[
            "max_observed_rss_mb"
        ]
        for row in rows
    ]

    save_line_plot(
        documents,
        memory,
        "Corpus size (documents)",
        "Maximum observed RSS (MB)",
        "V4: Corpus Size vs Observed Process Memory",
        RESULTS_DIR
        / "v4_memory.png",
    )

    # --------------------------------------------------------
    # Recall@5
    # --------------------------------------------------------

    recall = [
        row[
            "recall_at_k"
        ]
        for row in rows
    ]

    save_line_plot(
        documents,
        recall,
        "Corpus size (documents)",
        "Recall@5",
        "V4: Corpus Size vs Recall@5",
        RESULTS_DIR
        / "v4_recall_at_5.png",
    )


# ============================================================
# Console reporting
# ============================================================

def print_measured_table(rows):
    """
    Print compact measured scaling table.
    """

    print()
    print(
        "Measured V4 Scaling Results"
    )

    print(
        "==========================="
    )

    print()

    header = (
        f"{'Docs':>10} "
        f"{'Recall@5':>10} "
        f"{'Embed min':>11} "
        f"{'Docs/s':>9} "
        f"{'Index MB':>10} "
        f"{'Latency ms':>11} "
        f"{'RSS MB':>10}"
    )

    print(header)

    print(
        "-" * len(header)
    )

    for row in rows:

        embedding_minutes = (
            row[
                "document_embedding_time_sec"
            ]
            / 60
        )

        print(
            f"{row['documents']:>10,} "
            f"{row['recall_at_k']:>10.4f} "
            f"{embedding_minutes:>11.2f} "
            f"{row['document_embedding_docs_per_sec']:>9.2f} "
            f"{row['index_size_mb']:>10.2f} "
            f"{row['avg_query_latency_ms']:>11.3f} "
            f"{row['max_observed_rss_mb']:>10.2f}"
        )


def print_projection_table(rows):
    """
    Print projection table.
    """

    print()
    print(
        "Exploratory Linear Projections"
    )

    print(
        "=============================="
    )

    print()

    print(
        "WARNING: These values are projections, "
        "not measurements."
    )

    print()

    header = (
        f"{'Docs':>10} "
        f"{'Embed hr':>10} "
        f"{'Index MB':>10} "
        f"{'Latency ms':>11} "
        f"{'RSS MB':>10}"
    )

    print(header)

    print(
        "-" * len(header)
    )

    for row in rows:

        print(
            f"{row['documents']:>10,} "
            f"{row['projected_embedding_time_hr']:>10.2f} "
            f"{row['projected_index_size_mb']:>10.2f} "
            f"{row['projected_avg_query_latency_ms']:>11.3f} "
            f"{row['projected_max_observed_rss_mb']:>10.2f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load and combine measured experiments
    # --------------------------------------------------------

    measured = (
        load_measured_results()
    )

    if len(measured) < 2:

        raise RuntimeError(
            "At least two measured scaling points "
            "are required for V4 analysis."
        )

    # --------------------------------------------------------
    # Derived scaling metrics
    # --------------------------------------------------------

    summary = (
        calculate_scaling_metrics(
            measured
        )
    )

    # --------------------------------------------------------
    # Exploratory projections
    # --------------------------------------------------------

    projections = (
        project_scaling(
            measured
        )
    )

    # --------------------------------------------------------
    # Save consolidated measured summary
    # --------------------------------------------------------

    write_csv(
        SUMMARY_FILE,
        summary,
    )

    # --------------------------------------------------------
    # Save projections separately
    # --------------------------------------------------------

    write_csv(
        PROJECTIONS_FILE,
        projections,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    create_figures(
        measured
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print(
        "V4 - MS MARCO Scaling Analysis"
    )

    print(
        "=============================="
    )

    print(
        f"Measured scale points: "
        f"{len(measured)}"
    )

    print_measured_table(
        measured
    )

    print_projection_table(
        projections
    )

    # --------------------------------------------------------
    # Main observations
    # --------------------------------------------------------

    smallest = measured[0]
    largest = measured[-1]

    corpus_factor = (
        largest["documents"]
        / smallest["documents"]
    )

    embedding_factor = (
        largest[
            "document_embedding_time_sec"
        ]
        / smallest[
            "document_embedding_time_sec"
        ]
    )

    latency_factor = (
        largest[
            "avg_query_latency_ms"
        ]
        / smallest[
            "avg_query_latency_ms"
        ]
    )

    index_factor = (
        largest[
            "index_size_mb"
        ]
        / smallest[
            "index_size_mb"
        ]
    )

    recall_difference = (
        largest[
            "recall_at_k"
        ]
        - smallest[
            "recall_at_k"
        ]
    )

    print()
    print(
        "Measured End-to-End Change"
    )

    print(
        "=========================="
    )

    print(
        f"Corpus growth:          "
        f"{corpus_factor:.2f}x"
    )

    print(
        f"Embedding-time growth: "
        f"{embedding_factor:.2f}x"
    )

    print(
        f"Index-size growth:     "
        f"{index_factor:.2f}x"
    )

    print(
        f"Latency growth:        "
        f"{latency_factor:.2f}x"
    )

    print(
        f"Recall@5 change:       "
        f"{recall_difference:+.4f}"
    )

    print()
    print(
        "Measured data and projected data "
        "are stored separately."
    )

    print()

    print(
        "Measured summary:"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print()

    print(
        "Projections:"
    )

    print(
        f"  {PROJECTIONS_FILE}"
    )

    print()

    print(
        "Figures:"
    )

    for filename in [
        "v4_embedding_time.png",
        "v4_retrieval_latency.png",
        "v4_index_size.png",
        "v4_memory.png",
        "v4_recall_at_5.png",
    ]:

        print(
            f"  {RESULTS_DIR / filename}"
        )

    print()
    print(
        "NOTE:"
    )

    print(
        "  Recall@5 is intentionally NOT projected."
    )

    print(
        "  Retrieval quality need not degrade "
        "linearly with corpus size."
    )


if __name__ == "__main__":
    main()