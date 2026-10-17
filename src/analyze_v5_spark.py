"""
V5 - Apache Spark Embedding Analysis

Analyzes the local Apache Spark embedding experiments performed in V5.

Experiment families
-------------------

1. Original Spark parallelism experiments
   results/v5_spark_embedding_experiments.csv

2. Controlled Spark + PyTorch threading experiments
   results/v5_spark_thread_experiments.csv

3. Spark-native JSONL ingestion experiments
   results/v5_spark_native_ingestion_experiments.csv

The analysis evaluates:

    - Spark parallelism overhead
    - PyTorch thread allocation
    - workload-size effects
    - Spark vs V4 direct-Python embedding
    - driver-mediated vs Spark-native ingestion
    - throughput
    - speedup
    - repeatability

Outputs
-------

CSV:
    results/v5_parallelism_summary.csv
    results/v5_workload_scaling_summary.csv
    results/v5_architecture_summary.csv

Figures:
    results/v5_parallelism_1k.png
    results/v5_architecture_comparison.png
    results/v5_speedup_vs_corpus_size.png
    results/v5_throughput_vs_corpus_size.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

ORIGINAL_FILE = (
    RESULTS_DIR
    / "v5_spark_embedding_experiments.csv"
)

THREAD_FILE = (
    RESULTS_DIR
    / "v5_spark_thread_experiments.csv"
)

NATIVE_FILE = (
    RESULTS_DIR
    / "v5_spark_native_ingestion_experiments.csv"
)

PARALLELISM_SUMMARY_FILE = (
    RESULTS_DIR
    / "v5_parallelism_summary.csv"
)

WORKLOAD_SUMMARY_FILE = (
    RESULTS_DIR
    / "v5_workload_scaling_summary.csv"
)

ARCHITECTURE_SUMMARY_FILE = (
    RESULTS_DIR
    / "v5_architecture_summary.csv"
)


# ============================================================
# V4 direct-Python baselines
# ============================================================

V4_BASELINES = {
    1_000: {
        "embedding_time_sec": 27.0573,
        "docs_per_sec": 36.9586,
    },

    25_000: {
        "embedding_time_sec": 825.4622,
        "docs_per_sec": 30.2861,
    },

    50_000: {
        "embedding_time_sec": 1629.7635,
        "docs_per_sec": 30.6793,
    },

    100_000: {
        "embedding_time_sec": 3171.4882,
        "docs_per_sec": 31.5309,
    },

    250_000: {
        "embedding_time_sec": 10695.1256,
        "docs_per_sec": 23.3751,
    },
}


# ============================================================
# Utility functions
# ============================================================

def read_csv(path: Path):
    """
    Read a CSV file.

    Missing files return an empty list so the analysis can
    still operate on the experiment families that exist.
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

    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    return int(
        float(value)
    )


def to_float(value, default=0.0):

    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    return float(value)


def write_csv(
    path: Path,
    rows: list[dict],
):

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


def sample_stdev(values):

    if len(values) < 2:
        return 0.0

    return stdev(values)


# ============================================================
# Normalize experiment rows
# ============================================================

def normalize_original(row):

    return {
        "timestamp":
            row.get("timestamp", ""),

        "documents":
            to_int(
                row.get("documents")
            ),

        "partitions":
            to_int(
                row.get("partitions")
            ),

        "embedding_time_sec":
            to_float(
                row.get(
                    "spark_embedding_time_sec"
                )
            ),

        "docs_per_sec":
            to_float(
                row.get(
                    "spark_docs_per_sec"
                )
            ),

        "speedup_vs_v4":
            to_float(
                row.get(
                    "speedup_vs_v4"
                )
            ),

        "architecture":
            "driver_mediated_uncontrolled",

        "torch_threads":
            None,
    }


def normalize_thread(row):

    return {
        "timestamp":
            row.get("timestamp", ""),

        "documents":
            to_int(
                row.get("documents")
            ),

        "partitions":
            to_int(
                row.get("partitions")
            ),

        "torch_threads":
            to_int(
                row.get(
                    "torch_threads_per_worker"
                )
            ),

        "thread_budget":
            to_int(
                row.get(
                    "nominal_thread_budget"
                )
            ),

        "embedding_time_sec":
            to_float(
                row.get(
                    "spark_embedding_time_sec"
                )
            ),

        "docs_per_sec":
            to_float(
                row.get(
                    "spark_docs_per_sec"
                )
            ),

        "speedup_vs_v4":
            to_float(
                row.get(
                    "speedup_vs_v4"
                )
            ),

        "architecture":
            "driver_mediated_controlled",
    }


def normalize_native(row):

    return {
        "timestamp":
            row.get("timestamp", ""),

        "documents":
            to_int(
                row.get("documents")
            ),

        "partitions":
            to_int(
                row.get("partitions")
            ),

        "torch_threads":
            to_int(
                row.get(
                    "torch_threads_per_worker"
                )
            ),

        "thread_budget":
            to_int(
                row.get(
                    "nominal_thread_budget"
                )
            ),

        "embedding_time_sec":
            to_float(
                row.get(
                    "spark_embedding_time_sec"
                )
            ),

        "pipeline_time_sec":
            to_float(
                row.get(
                    "spark_pipeline_time_sec"
                )
            ),

        "read_time_sec":
            to_float(
                row.get(
                    "spark_read_count_time_sec"
                )
            ),

        "docs_per_sec":
            to_float(
                row.get(
                    "spark_docs_per_sec"
                )
            ),

        "speedup_vs_v4":
            to_float(
                row.get(
                    "embedding_speedup_vs_v4"
                )
            ),

        "pipeline_speedup_vs_v4":
            to_float(
                row.get(
                    "pipeline_speedup_vs_v4"
                )
            ),

        "architecture":
            "spark_native_jsonl",
    }


# ============================================================
# Parallelism analysis
# ============================================================

def analyze_parallelism(
    thread_rows,
):
    """
    Analyze the controlled 1K thread-allocation experiment.
    """

    rows = [
        row
        for row in thread_rows
        if row["documents"] == 1_000
    ]

    rows.sort(
        key=lambda row:
            (
                row["partitions"],
                row["torch_threads"],
            )
    )

    summary = []

    for row in rows:

        summary.append({

            "documents":
                row["documents"],

            "partitions":
                row["partitions"],

            "torch_threads_per_worker":
                row["torch_threads"],

            "nominal_thread_budget":
                row["thread_budget"],

            "embedding_time_sec":
                round(
                    row["embedding_time_sec"],
                    4,
                ),

            "docs_per_sec":
                round(
                    row["docs_per_sec"],
                    4,
                ),

            "speedup_vs_v4":
                round(
                    row["speedup_vs_v4"],
                    4,
                ),

            "v4_embedding_time_sec":
                V4_BASELINES[
                    1_000
                ][
                    "embedding_time_sec"
                ],
        })

    return summary


# ============================================================
# Workload scaling analysis
# ============================================================

def analyze_workload_scaling(
    thread_rows,
    native_rows,
):
    """
    Compare the selected P=2/T=2 configuration across
    increasing workload sizes.

    Repeated runs are summarized using mean and sample
    standard deviation.
    """

    controlled = [
        row
        for row in thread_rows
        if (
            row["partitions"] == 2
            and row["torch_threads"] == 2
        )
    ]

    native = [
        row
        for row in native_rows
        if (
            row["partitions"] == 2
            and row["torch_threads"] == 2
        )
    ]

    controlled_by_size = defaultdict(
        list
    )

    native_by_size = defaultdict(
        list
    )

    for row in controlled:

        controlled_by_size[
            row["documents"]
        ].append(row)

    for row in native:

        native_by_size[
            row["documents"]
        ].append(row)

    sizes = sorted(
        set(
            controlled_by_size.keys()
        )
        | set(
            native_by_size.keys()
        )
    )

    summary = []

    for size in sizes:

        baseline = (
            V4_BASELINES.get(size)
        )

        if baseline is None:
            continue

        controlled_runs = (
            controlled_by_size.get(
                size,
                [],
            )
        )

        native_runs = (
            native_by_size.get(
                size,
                [],
            )
        )

        controlled_times = [
            row["embedding_time_sec"]
            for row in controlled_runs
        ]

        native_times = [
            row["embedding_time_sec"]
            for row in native_runs
        ]

        controlled_mean = (
            mean(controlled_times)
            if controlled_times
            else None
        )

        native_mean = (
            mean(native_times)
            if native_times
            else None
        )

        controlled_speedup = (
            baseline[
                "embedding_time_sec"
            ]
            / controlled_mean
            if controlled_mean
            else None
        )

        native_speedup = (
            baseline[
                "embedding_time_sec"
            ]
            / native_mean
            if native_mean
            else None
        )

        summary.append({

            "documents":
                size,

            "v4_embedding_time_sec":
                round(
                    baseline[
                        "embedding_time_sec"
                    ],
                    4,
                ),

            "v4_docs_per_sec":
                round(
                    baseline[
                        "docs_per_sec"
                    ],
                    4,
                ),

            "driver_mediated_runs":
                len(
                    controlled_runs
                ),

            "driver_mediated_mean_embedding_sec":
                (
                    round(
                        controlled_mean,
                        4,
                    )
                    if controlled_mean
                    is not None
                    else ""
                ),

            "driver_mediated_stdev_sec":
                (
                    round(
                        sample_stdev(
                            controlled_times
                        ),
                        4,
                    )
                    if controlled_times
                    else ""
                ),

            "driver_mediated_speedup_vs_v4":
                (
                    round(
                        controlled_speedup,
                        4,
                    )
                    if controlled_speedup
                    is not None
                    else ""
                ),

            "native_runs":
                len(
                    native_runs
                ),

            "native_mean_embedding_sec":
                (
                    round(
                        native_mean,
                        4,
                    )
                    if native_mean
                    is not None
                    else ""
                ),

            "native_stdev_sec":
                (
                    round(
                        sample_stdev(
                            native_times
                        ),
                        4,
                    )
                    if native_times
                    else ""
                ),

            "native_speedup_vs_v4":
                (
                    round(
                        native_speedup,
                        4,
                    )
                    if native_speedup
                    is not None
                    else ""
                ),
        })

    return summary


# ============================================================
# Architecture comparison
# ============================================================

def analyze_architecture(
    thread_rows,
    native_rows,
):
    """
    Compare direct Python, driver-mediated Spark, and
    Spark-native ingestion for common workload sizes.
    """

    selected_thread_rows = [
        row
        for row in thread_rows
        if (
            row["partitions"] == 2
            and row["torch_threads"] == 2
        )
    ]

    selected_native_rows = [
        row
        for row in native_rows
        if (
            row["partitions"] == 2
            and row["torch_threads"] == 2
        )
    ]

    thread_by_size = defaultdict(
        list
    )

    native_by_size = defaultdict(
        list
    )

    for row in selected_thread_rows:

        thread_by_size[
            row["documents"]
        ].append(
            row
        )

    for row in selected_native_rows:

        native_by_size[
            row["documents"]
        ].append(
            row
        )

    common_sizes = sorted(
        set(
            thread_by_size.keys()
        )
        & set(
            native_by_size.keys()
        )
    )

    summary = []

    for size in common_sizes:

        baseline = (
            V4_BASELINES[size]
        )

        driver_times = [
            row["embedding_time_sec"]
            for row
            in thread_by_size[size]
        ]

        native_embedding_times = [
            row["embedding_time_sec"]
            for row
            in native_by_size[size]
        ]

        native_pipeline_times = [
            row["pipeline_time_sec"]
            for row
            in native_by_size[size]
        ]

        driver_mean = mean(
            driver_times
        )

        native_embedding_mean = mean(
            native_embedding_times
        )

        native_pipeline_mean = mean(
            native_pipeline_times
        )

        summary.append({

            "documents":
                size,

            "v4_direct_embedding_sec":
                round(
                    baseline[
                        "embedding_time_sec"
                    ],
                    4,
                ),

            "driver_mediated_embedding_sec":
                round(
                    driver_mean,
                    4,
                ),

            "spark_native_embedding_sec":
                round(
                    native_embedding_mean,
                    4,
                ),

            "spark_native_pipeline_sec":
                round(
                    native_pipeline_mean,
                    4,
                ),

            "driver_vs_v4_speedup":
                round(
                    baseline[
                        "embedding_time_sec"
                    ]
                    / driver_mean,
                    4,
                ),

            "native_embedding_vs_v4_speedup":
                round(
                    baseline[
                        "embedding_time_sec"
                    ]
                    / native_embedding_mean,
                    4,
                ),

            "native_pipeline_vs_v4_speedup":
                round(
                    baseline[
                        "embedding_time_sec"
                    ]
                    / native_pipeline_mean,
                    4,
                ),

            "native_vs_driver_embedding_ratio":
                round(
                    native_embedding_mean
                    / driver_mean,
                    4,
                ),
        })

    return summary


# ============================================================
# Figures
# ============================================================

def create_parallelism_figure(
    parallelism_summary,
):

    if not parallelism_summary:
        return

    labels = [
        (
            f"P={row['partitions']}\n"
            f"T={row['torch_threads_per_worker']}"
        )
        for row
        in parallelism_summary
    ]

    times = [
        row["embedding_time_sec"]
        for row
        in parallelism_summary
    ]

    baseline = (
        V4_BASELINES[
            1_000
        ][
            "embedding_time_sec"
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        labels,
        times,
    )

    plt.axhline(
        baseline,
        linestyle="--",
        label="V4 direct Python",
    )

    plt.xlabel(
        "Spark partitions / Torch threads"
    )

    plt.ylabel(
        "Embedding time (seconds)"
    )

    plt.title(
        "V5: 1K Spark Parallelism and PyTorch Threading"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "v5_parallelism_1k.png",
        dpi=160,
    )

    plt.close()


def create_architecture_figure(
    architecture_summary,
):

    if not architecture_summary:
        return

    documents = [
        row["documents"]
        for row
        in architecture_summary
    ]

    v4 = [
        row[
            "v4_direct_embedding_sec"
        ]
        for row
        in architecture_summary
    ]

    driver = [
        row[
            "driver_mediated_embedding_sec"
        ]
        for row
        in architecture_summary
    ]

    native = [
        row[
            "spark_native_embedding_sec"
        ]
        for row
        in architecture_summary
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        documents,
        v4,
        marker="o",
        label="V4 direct Python",
    )

    plt.plot(
        documents,
        driver,
        marker="o",
        label="Spark driver-mediated",
    )

    plt.plot(
        documents,
        native,
        marker="o",
        label="Spark-native JSONL",
    )

    plt.xlabel(
        "Corpus size (documents)"
    )

    plt.ylabel(
        "Embedding time (seconds)"
    )

    plt.title(
        "V5: Embedding Architecture Comparison"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "v5_architecture_comparison.png",
        dpi=160,
    )

    plt.close()


def create_speedup_figure(
    workload_summary,
):

    if not workload_summary:
        return

    rows = [
        row
        for row in workload_summary
        if (
            row[
                "native_speedup_vs_v4"
            ]
            != ""
        )
    ]

    if not rows:
        return

    documents = [
        row["documents"]
        for row in rows
    ]

    native_speedup = [
        float(
            row[
                "native_speedup_vs_v4"
            ]
        )
        for row in rows
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        documents,
        native_speedup,
        marker="o",
        label="Spark-native embedding",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="V4 parity",
    )

    plt.xlabel(
        "Corpus size (documents)"
    )

    plt.ylabel(
        "Speedup vs V4"
    )

    plt.title(
        "V5: Spark Speedup vs Corpus Size"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "v5_speedup_vs_corpus_size.png",
        dpi=160,
    )

    plt.close()


def create_throughput_figure(
    native_rows,
):

    selected = [
        row
        for row in native_rows
        if (
            row["partitions"] == 2
            and row["torch_threads"] == 2
        )
    ]

    if not selected:
        return

    by_size = defaultdict(
        list
    )

    for row in selected:

        by_size[
            row["documents"]
        ].append(
            row["docs_per_sec"]
        )

    sizes = sorted(
        by_size.keys()
    )

    spark_throughput = [
        mean(
            by_size[size]
        )
        for size in sizes
    ]

    v4_throughput = [
        V4_BASELINES[
            size
        ][
            "docs_per_sec"
        ]
        for size in sizes
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        sizes,
        v4_throughput,
        marker="o",
        label="V4 direct Python",
    )

    plt.plot(
        sizes,
        spark_throughput,
        marker="o",
        label="Spark-native P=2/T=2",
    )

    plt.xlabel(
        "Corpus size (documents)"
    )

    plt.ylabel(
        "Embedding throughput (docs/s)"
    )

    plt.title(
        "V5: Embedding Throughput vs Corpus Size"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "v5_throughput_vs_corpus_size.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# Console reporting
# ============================================================

def print_parallelism(
    rows,
):

    print()
    print(
        "Controlled 1K Parallelism"
    )

    print(
        "========================="
    )

    print()

    header = (
        f"{'P':>4} "
        f"{'T':>4} "
        f"{'Budget':>7} "
        f"{'Time(s)':>10} "
        f"{'Docs/s':>9} "
        f"{'Speedup':>9}"
    )

    print(header)

    print(
        "-" * len(header)
    )

    for row in rows:

        print(
            f"{row['partitions']:>4} "
            f"{row['torch_threads_per_worker']:>4} "
            f"{row['nominal_thread_budget']:>7} "
            f"{row['embedding_time_sec']:>10.2f} "
            f"{row['docs_per_sec']:>9.2f} "
            f"{row['speedup_vs_v4']:>9.3f}"
        )


def print_workload_scaling(
    rows,
):

    print()
    print(
        "Selected P=2/T=2 Workload Scaling"
    )

    print(
        "================================="
    )

    print()

    header = (
        f"{'Docs':>10} "
        f"{'V4(s)':>10} "
        f"{'Driver(s)':>11} "
        f"{'Native(s)':>11} "
        f"{'Native speedup':>14}"
    )

    print(header)

    print(
        "-" * len(header)
    )

    for row in rows:

        driver = (
            row[
                "driver_mediated_mean_embedding_sec"
            ]
        )

        native = (
            row[
                "native_mean_embedding_sec"
            ]
        )

        speedup = (
            row[
                "native_speedup_vs_v4"
            ]
        )

        driver_text = (
            f"{float(driver):.2f}"
            if driver != ""
            else "-"
        )

        native_text = (
            f"{float(native):.2f}"
            if native != ""
            else "-"
        )

        speedup_text = (
            f"{float(speedup):.3f}"
            if speedup != ""
            else "-"
        )

        print(
            f"{row['documents']:>10,} "
            f"{row['v4_embedding_time_sec']:>10.2f} "
            f"{driver_text:>11} "
            f"{native_text:>11} "
            f"{speedup_text:>14}"
        )


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_raw = read_csv(
        ORIGINAL_FILE
    )

    thread_raw = read_csv(
        THREAD_FILE
    )

    native_raw = read_csv(
        NATIVE_FILE
    )

    original_rows = [
        normalize_original(row)
        for row in original_raw
    ]

    thread_rows = [
        normalize_thread(row)
        for row in thread_raw
    ]

    native_rows = [
        normalize_native(row)
        for row in native_raw
    ]

    if (
        not original_rows
        and not thread_rows
        and not native_rows
    ):

        raise RuntimeError(
            "No V5 experiment results were found."
        )

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    parallelism_summary = (
        analyze_parallelism(
            thread_rows
        )
    )

    workload_summary = (
        analyze_workload_scaling(
            thread_rows,
            native_rows,
        )
    )

    architecture_summary = (
        analyze_architecture(
            thread_rows,
            native_rows,
        )
    )

    # --------------------------------------------------------
    # Save summaries
    # --------------------------------------------------------

    write_csv(
        PARALLELISM_SUMMARY_FILE,
        parallelism_summary,
    )

    write_csv(
        WORKLOAD_SUMMARY_FILE,
        workload_summary,
    )

    write_csv(
        ARCHITECTURE_SUMMARY_FILE,
        architecture_summary,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    create_parallelism_figure(
        parallelism_summary
    )

    create_architecture_figure(
        architecture_summary
    )

    create_speedup_figure(
        workload_summary
    )

    create_throughput_figure(
        native_rows
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print(
        "V5 - Apache Spark Analysis"
    )

    print(
        "=========================="
    )

    print()

    print(
        f"Original Spark runs:   "
        f"{len(original_rows)}"
    )

    print(
        f"Controlled-thread runs:"
        f" {len(thread_rows)}"
    )

    print(
        f"Native-ingestion runs: "
        f"{len(native_rows)}"
    )

    print_parallelism(
        parallelism_summary
    )

    print_workload_scaling(
        workload_summary
    )

    # --------------------------------------------------------
    # Key observations
    # --------------------------------------------------------

    if parallelism_summary:

        best = min(
            parallelism_summary,
            key=lambda row:
                row[
                    "embedding_time_sec"
                ],
        )

        print()
        print(
            "Best controlled 1K Spark configuration:"
        )

        print(
            f"  P={best['partitions']}, "
            f"T={best['torch_threads_per_worker']}"
        )

        print(
            f"  Time: "
            f"{best['embedding_time_sec']:.2f}s"
        )

        print(
            f"  Speedup vs V4: "
            f"{best['speedup_vs_v4']:.3f}x"
        )

    native_50k = [
        row
        for row in native_rows
        if (
            row["documents"] == 50_000
            and row["partitions"] == 2
            and row["torch_threads"] == 2
        )
    ]

    if native_50k:

        avg_50k = mean(
            row[
                "embedding_time_sec"
            ]
            for row
            in native_50k
        )

        baseline_50k = (
            V4_BASELINES[
                50_000
            ][
                "embedding_time_sec"
            ]
        )

        print()
        print(
            "50K Spark-native comparison:"
        )

        print(
            f"  V4 direct Python: "
            f"{baseline_50k:.2f}s"
        )

        print(
            f"  Spark native:     "
            f"{avg_50k:.2f}s"
        )

        print(
            f"  Speedup:          "
            f"{baseline_50k / avg_50k:.3f}x"
        )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  Local Spark does not create additional physical "
        "compute resources."
    )

    print(
        "  Small workloads are dominated by Spark and "
        "worker/model initialization overhead."
    )

    print(
        "  Dividing the same CPU budget among more Spark "
        "workers reduced performance in the 1K experiment."
    )

    print(
        "  At 25K-50K, Spark approaches direct-Python "
        "performance but has not shown a material local speedup."
    )

    print(
        "  Spark-native ingestion removes the driver-mediated "
        "corpus-transfer architecture and is preferable for "
        "future distributed experiments."
    )

    print()

    print(
        "Outputs:"
    )

    for path in [
        PARALLELISM_SUMMARY_FILE,
        WORKLOAD_SUMMARY_FILE,
        ARCHITECTURE_SUMMARY_FILE,
        RESULTS_DIR
        / "v5_parallelism_1k.png",
        RESULTS_DIR
        / "v5_architecture_comparison.png",
        RESULTS_DIR
        / "v5_speedup_vs_corpus_size.png",
        RESULTS_DIR
        / "v5_throughput_vs_corpus_size.png",
    ]:

        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()