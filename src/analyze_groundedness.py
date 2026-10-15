"""
V3 Experiment 3:
Analyze human-scored RAG groundedness results by Top-K.

Input:
    results/rag_groundedness_evaluation_scored.csv

Outputs:
    results/groundedness_summary.csv

Metrics:
- number of runs
- evidence retrieval rate
- normalized groundedness
- normalized citation faithfulness
- citation validity rate
- abstention correctness rate
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
    / "rag_groundedness_evaluation_scored.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "groundedness_summary.csv"
)


def parse_bool(value: str) -> int:
    """
    Convert common boolean representations to 0/1.
    """

    value = str(value).strip().lower()

    return 1 if value in {
        "true",
        "1",
        "yes",
    } else 0


def parse_optional_float(value: str):
    """
    Convert a CSV value to float.

    Return None for blank values.
    """

    value = str(value).strip()

    if value == "":
        return None

    return float(value)


def average(values):
    """
    Average non-None values.
    """

    valid = [
        value
        for value in values
        if value is not None
    ]

    if not valid:
        return None

    return sum(valid) / len(valid)


def format_metric(value):
    """
    Format metric for console output.
    """

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    grouped = defaultdict(list)

    # -----------------------------------------------------
    # Load scored evaluation data
    # -----------------------------------------------------

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            k = int(row["top_k"])

            grouped[k].append(row)

    if not grouped:

        raise RuntimeError(
            "No evaluation rows found."
        )

    summary_rows = []

    # -----------------------------------------------------
    # Calculate metrics by K
    # -----------------------------------------------------

    for k in sorted(grouped):

        rows = grouped[k]

        runs = len(rows)

        # Retrieval success
        evidence_retrieval_rate = (
            sum(
                parse_bool(
                    row["relevant_retrieved"]
                )
                for row in rows
            )
            / runs
        )

        # Groundedness is scored 0-2.
        groundedness_raw = [
            parse_optional_float(
                row["groundedness_score"]
            )
            for row in rows
        ]

        groundedness_avg_raw = average(
            groundedness_raw
        )

        groundedness_normalized = (
            groundedness_avg_raw / 2
            if groundedness_avg_raw
            is not None
            else None
        )

        # Citation faithfulness is scored 0-2.
        faithfulness_raw = [
            parse_optional_float(
                row[
                    "citation_faithfulness_score"
                ]
            )
            for row in rows
        ]

        faithfulness_avg_raw = average(
            faithfulness_raw
        )

        faithfulness_normalized = (
            faithfulness_avg_raw / 2
            if faithfulness_avg_raw
            is not None
            else None
        )

        # Citation validity is 0/1.
        citation_validity = average([
            parse_optional_float(
                row["citation_valid"]
            )
            for row in rows
        ])

        # Abstention correctness is 0/1.
        abstention_correctness = average([
            parse_optional_float(
                row["abstention_correct"]
            )
            for row in rows
        ])

        # Performance metrics.
        generation_latency = average([
            parse_optional_float(
                row["generation_time_sec"]
            )
            for row in rows
        ])

        context_characters = average([
            parse_optional_float(
                row["context_characters"]
            )
            for row in rows
        ])

        summary = {
            "top_k": k,
            "runs": runs,

            "evidence_retrieval_rate":
                round(
                    evidence_retrieval_rate,
                    4,
                ),

            "groundedness":
                (
                    round(
                        groundedness_normalized,
                        4,
                    )
                    if groundedness_normalized
                    is not None
                    else ""
                ),

            "citation_faithfulness":
                (
                    round(
                        faithfulness_normalized,
                        4,
                    )
                    if faithfulness_normalized
                    is not None
                    else ""
                ),

            "citation_validity_rate":
                (
                    round(
                        citation_validity,
                        4,
                    )
                    if citation_validity
                    is not None
                    else ""
                ),

            "abstention_correctness":
                (
                    round(
                        abstention_correctness,
                        4,
                    )
                    if abstention_correctness
                    is not None
                    else ""
                ),

            "avg_generation_time_sec":
                round(
                    generation_latency,
                    2,
                ),

            "avg_context_characters":
                round(
                    context_characters,
                    1,
                ),
        }

        summary_rows.append(
            summary
        )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    fields = [
        "top_k",
        "runs",
        "evidence_retrieval_rate",
        "groundedness",
        "citation_faithfulness",
        "citation_validity_rate",
        "abstention_correctness",
        "avg_generation_time_sec",
        "avg_context_characters",
    ]

    with OUTPUT_FILE.open(
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
            summary_rows
        )

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    print()
    print(
        "V3 RAG Groundedness Analysis"
    )

    print(
        "============================"
    )

    print()

    header = (
        f"{'K':>3} "
        f"{'Runs':>5} "
        f"{'Retrieve':>9} "
        f"{'Ground':>9} "
        f"{'Faithful':>9} "
        f"{'CiteValid':>10} "
        f"{'Abstain':>9} "
        f"{'Time(s)':>9} "
        f"{'Context':>10}"
    )

    print(header)

    print("-" * len(header))

    for row in summary_rows:

        print(
            f"{row['top_k']:>3} "
            f"{row['runs']:>5} "
            f"{format_metric(row['evidence_retrieval_rate']):>9} "
            f"{format_metric(row['groundedness']):>9} "
            f"{format_metric(row['citation_faithfulness']):>9} "
            f"{format_metric(row['citation_validity_rate']):>10} "
            f"{format_metric(row['abstention_correctness']):>9} "
            f"{row['avg_generation_time_sec']:>9.2f} "
            f"{row['avg_context_characters']:>10.1f}"
        )

    print()

    print(
        f"Summary saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Groundedness and citation faithfulness "
        "are normalized from human scores "
        "0-2 into the range 0-1."
    )

    print(
        "These pilot results contain only "
        "5 queries per K and should not be "
        "interpreted as final estimates."
    )


if __name__ == "__main__":
    main()