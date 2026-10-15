"""
V3 Experiment 3:
Prepare a human RAG-groundedness evaluation worksheet
including the full retrieved evidence text.

No LLM generation is performed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from retrieve import Retriever


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR
    / "rag_topk_experiment.jsonl"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "rag_groundedness_evaluation_with_evidence.csv"
)


def format_evidence(results):
    """
    Format full retrieved documents for human evaluation.
    """

    blocks = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        block = (
            f"[{rank}]\n"
            f"Document ID: {result['doc_id']}\n"
            f"Score: {result['score']:.4f}\n"
            f"Title: {result['title']}\n"
            f"Evidence:\n"
            f"{result['text']}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE}"
        )

    print()
    print(
        "Preparing groundedness worksheet "
        "with full evidence..."
    )

    # Load the retriever only once.
    retriever = Retriever()

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

            question = record["question"]

            k = int(
                record["top_k"]
            )

            # Re-run retrieval only.
            # No LLM generation.
            results = retriever.search(
                question,
                k=k,
            )

            full_evidence = (
                format_evidence(
                    results
                )
            )

            row = {

                # ----------------------------------
                # Experiment information
                # ----------------------------------

                "query_id":
                    record.get(
                        "query_id",
                        ""
                    ),

                "question":
                    question,

                "top_k":
                    k,

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
                        )
                        is not None
                        else ""
                    ),

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

                "citations_generated":
                    "|".join(
                        str(x)
                        for x
                        in record.get(
                            "citations",
                            []
                        )
                    ),

                # ----------------------------------
                # Generated answer
                # ----------------------------------

                "answer":
                    record.get(
                        "answer",
                        ""
                    ),

                # ----------------------------------
                # Full evidence
                # ----------------------------------

                "retrieved_evidence":
                    full_evidence,

                # ----------------------------------
                # Human evaluation
                # ----------------------------------

                "groundedness_score":
                    "",

                "citation_faithfulness_score":
                    "",

                "citation_valid":
                    "",

                "abstention_correct":
                    "",

                "evaluator_notes":
                    "",
            }

            rows.append(row)

    fields = [

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

        "groundedness_score",
        "citation_faithfulness_score",
        "citation_valid",
        "abstention_correct",

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

        writer.writerows(
            rows
        )

    print()
    print(
        f"Runs included: "
        f"{len(rows)}"
    )

    print(
        f"Output: "
        f"{OUTPUT_FILE}"
    )

    print()
    print(
        "No LLM generation was performed."
    )


if __name__ == "__main__":
    main()