"""
V3 Experiment 4:
Prepare human-groundedness evaluation worksheet
from the stratified RAG experiment.

No retrieval or LLM generation is performed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR
    / "rag_stratified_experiment.jsonl"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "rag_stratified_groundedness_evaluation.csv"
)


def format_evidence(documents):
    """Format full retrieved evidence for human evaluation."""

    blocks = []

    for doc in documents:

        block = (
            f"[{doc['rank']}]\n"
            f"Document ID: {doc['doc_id']}\n"
            f"Score: {doc['score']}\n"
            f"Title: {doc['title']}\n"
            f"Evidence:\n{doc['text']}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    rows = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            documents = record.get(
                "retrieved_documents",
                []
            )

            row = {

                # Experiment design
                "stratum":
                    record.get("stratum", ""),

                "query_id":
                    record.get("query_id", ""),

                "question":
                    record.get("question", ""),

                "top_k":
                    record.get("top_k", ""),

                # Ground truth / retrieval
                "gold_documents":
                    "|".join(
                        record.get(
                            "gold_documents",
                            []
                        )
                    ),

                "relevant_retrieved":
                    record.get(
                        "relevant_retrieved",
                        ""
                    ),

                "first_relevant_rank":
                    (
                        record.get(
                            "first_relevant_rank"
                        )
                        if record.get(
                            "first_relevant_rank"
                        ) is not None
                        else ""
                    ),

                # Performance
                "context_characters":
                    record.get(
                        "context_characters",
                        ""
                    ),

                "generation_time_sec":
                    record.get(
                        "generation_time_sec",
                        ""
                    ),

                # Citations
                "citations_generated":
                    "|".join(
                        str(x)
                        for x in record.get(
                            "citations",
                            []
                        )
                    ),

                # Generated answer
                "answer":
                    record.get(
                        "answer",
                        ""
                    ),

                # Full evidence
                "retrieved_evidence":
                    format_evidence(
                        documents
                    ),

                # Errors
                "generation_error":
                    record.get(
                        "error",
                        ""
                    ) or "",

                # Human evaluation
                "groundedness_score": "",
                "citation_faithfulness_score": "",
                "citation_valid": "",
                "abstention_correct": "",
                "generation_failure": "",
                "evaluator_notes": "",
            }

            rows.append(row)

    fields = [
        "stratum",
        "query_id",
        "question",
        "top_k",

        "gold_documents",
        "relevant_retrieved",
        "first_relevant_rank",

        "context_characters",
        "generation_time_sec",

        "citations_generated",

        "answer",
        "retrieved_evidence",

        "generation_error",

        "groundedness_score",
        "citation_faithfulness_score",
        "citation_valid",
        "abstention_correct",
        "generation_failure",

        "evaluator_notes",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "Stratified groundedness worksheet created."
    )

    print(
        f"Runs included: {len(rows)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print("Human scoring:")
    print("  groundedness_score: 0, 1, 2")
    print(
        "  citation_faithfulness_score: "
        "0, 1, 2, or blank"
    )
    print(
        "  citation_valid: "
        "0, 1, or blank"
    )
    print(
        "  abstention_correct: 0 or 1"
    )
    print(
        "  generation_failure: 0 or 1"
    )


if __name__ == "__main__":
    main()