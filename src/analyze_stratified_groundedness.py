"""
V3 Experiment 4:
Analyze the scored stratified RAG experiment.

Input:
    results/rag_stratified_groundedness_evaluation_scored.csv

Outputs:
    results/stratified_summary_by_k.csv
    results/stratified_summary_by_group_k.csv

Metrics:
- evidence retrieval rate
- normalized groundedness
- normalized citation faithfulness
- citation validity rate
- abstention correctness
- generation failure rate
- average generation latency
- average context size
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR
    / "rag_stratified_groundedness_evaluation_scored.csv"
)

SUMMARY_BY_K = (
    RESULTS_DIR
    / "stratified_summary_by_k.csv"
)

SUMMARY_BY_GROUP_K = (
    RESULTS_DIR
    / "stratified_summary_by_group_k.csv"
)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def parse_bool(value):
    """Convert common boolean values to 0 or 1."""

    value = str(value).strip().lower()

    return 1 if value in {
        "true",
        "1",
        "yes",
    } else 0


def parse_optional_float(value):
    """Return float or None for blank values."""

    value = str(value).strip()

    if value == "":
        return None

    return float(value)


def average(values):
    """Average non-None values."""

    valid = [
        value
        for value in values
        if value is not None
    ]

    if not valid:
        return None

    return sum(valid) / len(valid)


def safe_round(value, digits=4):
    """Round unless value is None."""

    if value is None:
        return ""

    return round(value, digits)


def format_value(value):
    """Pretty formatting for console tables."""

    if value == "" or value is None:
        return "N/A"

    return f"{float(value):.4f}"


# ---------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------

def calculate_metrics(rows):
    """
    Calculate RAG metrics for a collection of rows.
    """

    runs = len(rows)

    if runs == 0:
        return {}

    # -------------------------------
    # Retrieval success
    # -------------------------------

    retrieval_rate = (
        sum(
            parse_bool(
                row["relevant_retrieved"]
            )
            for row in rows
        )
        / runs
    )

    # -------------------------------
    # Groundedness
    # Human score 0-2 -> normalize 0-1
    # -------------------------------

    groundedness_raw = [
        parse_optional_float(
            row["groundedness_score"]
        )
        for row in rows
    ]

    groundedness_avg = average(
        groundedness_raw
    )

    groundedness = (
        groundedness_avg / 2
        if groundedness_avg
        is not None
        else None
    )

    # -------------------------------
    # Citation faithfulness
    # Human score 0-2 -> normalize 0-1
    # -------------------------------

    faithfulness_raw = [
        parse_optional_float(
            row[
                "citation_faithfulness_score"
            ]
        )
        for row in rows
    ]

    faithfulness_avg = average(
        faithfulness_raw
    )

    faithfulness = (
        faithfulness_avg / 2
        if faithfulness_avg
        is not None
        else None
    )

    # -------------------------------
    # Citation validity
    # -------------------------------

    citation_validity = average([
        parse_optional_float(
            row["citation_valid"]
        )
        for row in rows
    ])

    # -------------------------------
    # Abstention correctness
    # -------------------------------

    abstention = average([
        parse_optional_float(
            row["abstention_correct"]
        )
        for row in rows
    ])

    # -------------------------------
    # Generation failure
    # -------------------------------

    generation_failure = average([
        parse_optional_float(
            row["generation_failure"]
        )
        for row in rows
    ])

    # -------------------------------
    # Performance
    # -------------------------------

    latency = average([
        parse_optional_float(
            row["generation_time_sec"]
        )
        for row in rows
    ])

    context = average([
        parse_optional_float(
            row["context_characters"]
        )
        for row in rows
    ])

    return {

        "runs":
            runs,

        "evidence_retrieval_rate":
            safe_round(
                retrieval_rate
            ),

        "groundedness":
            safe_round(
                groundedness
            ),

        "citation_faithfulness":
            safe_round(
                faithfulness
            ),

        "citation_validity_rate":
            safe_round(
                citation_validity
            ),

        "abstention_correctness":
            safe_round(
                abstention
            ),

        "generation_failure_rate":
            safe_round(
                generation_failure
            ),

        "avg_generation_time_sec":
            safe_round(
                latency,
                2,
            ),

        "avg_context_characters":
            safe_round(
                context,
                1,
            ),
    }


# ---------------------------------------------------------
# CSV writer
# ---------------------------------------------------------

def write_csv(
    path,
    rows,
    fields,
):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    # -----------------------------------------------------
    # Load scored experiment
    # -----------------------------------------------------

    rows = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append(row)

    if not rows:

        raise RuntimeError(
            "No experiment rows found."
        )

    print()
    print(
        "V3 Stratified RAG Analysis"
    )

    print(
        "=========================="
    )

    print(
        f"Total runs: {len(rows)}"
    )

    # -----------------------------------------------------
    # Analysis 1:
    # Overall performance by K
    # -----------------------------------------------------

    grouped_by_k = defaultdict(list)

    for row in rows:

        k = int(
            row["top_k"]
        )

        grouped_by_k[k].append(
            row
        )

    summary_by_k = []

    for k in sorted(
        grouped_by_k
    ):

        metrics = calculate_metrics(
            grouped_by_k[k]
        )

        summary_by_k.append({

            "top_k":
                k,

            **metrics,
        })

    fields_by_k = [

        "top_k",
        "runs",

        "evidence_retrieval_rate",
        "groundedness",
        "citation_faithfulness",
        "citation_validity_rate",
        "abstention_correctness",
        "generation_failure_rate",

        "avg_generation_time_sec",
        "avg_context_characters",
    ]

    write_csv(
        SUMMARY_BY_K,
        summary_by_k,
        fields_by_k,
    )

    # -----------------------------------------------------
    # Analysis 2:
    # Stratum × K
    # -----------------------------------------------------

    grouped_by_group_k = defaultdict(
        list
    )

    for row in rows:

        stratum = row[
            "stratum"
        ]

        k = int(
            row["top_k"]
        )

        grouped_by_group_k[
            (stratum, k)
        ].append(
            row
        )

    summary_group_k = []

    stratum_order = [
        "A",
        "B",
        "C",
        "D",
    ]

    for stratum in stratum_order:

        for k in sorted(
            grouped_by_k
        ):

            group_rows = (
                grouped_by_group_k[
                    (stratum, k)
                ]
            )

            if not group_rows:
                continue

            metrics = (
                calculate_metrics(
                    group_rows
                )
            )

            summary_group_k.append({

                "stratum":
                    stratum,

                "top_k":
                    k,

                **metrics,
            })

    fields_group_k = [

        "stratum",
        "top_k",
        "runs",

        "evidence_retrieval_rate",
        "groundedness",
        "citation_faithfulness",
        "citation_validity_rate",
        "abstention_correctness",
        "generation_failure_rate",

        "avg_generation_time_sec",
        "avg_context_characters",
    ]

    write_csv(
        SUMMARY_BY_GROUP_K,
        summary_group_k,
        fields_group_k,
    )

    # -----------------------------------------------------
    # Console output:
    # Overall K analysis
    # -----------------------------------------------------

    print()
    print(
        "Overall Performance by K"
    )

    print(
        "------------------------"
    )

    header = (
        f"{'K':>3} "
        f"{'N':>3} "
        f"{'Retrieve':>9} "
        f"{'Ground':>8} "
        f"{'Faith':>8} "
        f"{'Valid':>8} "
        f"{'Abstain':>8} "
        f"{'Fail':>7} "
        f"{'Time':>9} "
        f"{'Context':>10}"
    )

    print(header)

    print(
        "-" * len(header)
    )

    for row in summary_by_k:

        print(
            f"{row['top_k']:>3} "
            f"{row['runs']:>3} "
            f"{format_value(row['evidence_retrieval_rate']):>9} "
            f"{format_value(row['groundedness']):>8} "
            f"{format_value(row['citation_faithfulness']):>8} "
            f"{format_value(row['citation_validity_rate']):>8} "
            f"{format_value(row['abstention_correctness']):>8} "
            f"{format_value(row['generation_failure_rate']):>7} "
            f"{row['avg_generation_time_sec']:>9.2f} "
            f"{row['avg_context_characters']:>10.1f}"
        )

    # -----------------------------------------------------
    # Console output:
    # Stratum × K
    # -----------------------------------------------------

    print()
    print(
        "Performance by Stratum x K"
    )

    print(
        "--------------------------"
    )

    header2 = (
        f"{'Group':>5} "
        f"{'K':>3} "
        f"{'N':>3} "
        f"{'Retrieve':>9} "
        f"{'Ground':>8} "
        f"{'Faith':>8} "
        f"{'Abstain':>8} "
        f"{'Fail':>7} "
        f"{'Time':>9}"
    )

    print(header2)

    print(
        "-" * len(header2)
    )

    for row in summary_group_k:

        print(
            f"{row['stratum']:>5} "
            f"{row['top_k']:>3} "
            f"{row['runs']:>3} "
            f"{format_value(row['evidence_retrieval_rate']):>9} "
            f"{format_value(row['groundedness']):>8} "
            f"{format_value(row['citation_faithfulness']):>8} "
            f"{format_value(row['abstention_correctness']):>8} "
            f"{format_value(row['generation_failure_rate']):>7} "
            f"{row['avg_generation_time_sec']:>9.2f}"
        )

    # -----------------------------------------------------
    # Output paths
    # -----------------------------------------------------

    print()

    print(
        "Results saved:"
    )

    print(
        f"  {SUMMARY_BY_K}"
    )

    print(
        f"  {SUMMARY_BY_GROUP_K}"
    )

    print()

    print(
        "Strata:"
    )

    print(
        "  A = relevant document rank 1-3"
    )

    print(
        "  B = relevant document rank 4-5"
    )

    print(
        "  C = relevant document rank 6-10"
    )

    print(
        "  D = relevant document not found "
        "in Top-10"
    )

    print()

    print(
        "NOTE: Groundedness and citation "
        "faithfulness are normalized from "
        "human scores 0-2 to 0-1."
    )


if __name__ == "__main__":
    main()