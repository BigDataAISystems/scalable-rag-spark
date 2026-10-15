"""
V3 Experiment 2:
Evaluate how retrieval depth (Top-K) affects RAG behavior.

For each selected SciFact query and each K value, record:
- whether known relevant evidence was retrieved
- rank of the first relevant document
- retrieval scores
- context size
- generation latency
- generated answer
- citations appearing in the answer

Results are saved after EVERY run so progress is not lost.
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from load_scifact import load_scifact
from retrieve import Retriever
from rag import build_context, build_prompt, call_ollama


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

K_VALUES = [1, 3, 5, 10]

NUM_QUERIES = 5

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

CSV_FILE = RESULTS_DIR / "rag_topk_experiment.csv"

JSONL_FILE = RESULTS_DIR / "rag_topk_experiment.jsonl"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def extract_citations(answer: str) -> list[int]:
    """
    Extract numbered citations such as [1], [2], [5]
    from the generated answer.
    """

    matches = re.findall(r"\[(\d+)\]", answer)

    return sorted(
        {int(x) for x in matches}
    )


def first_relevant_rank(
    results: list[dict],
    gold_docs: set[str],
) -> int | None:
    """
    Return the rank of the first known relevant document
    in the retrieved results.

    Returns None if no relevant document was retrieved.
    """

    for rank, result in enumerate(results, start=1):

        if str(result["doc_id"]) in gold_docs:
            return rank

    return None


def save_csv_row(row: dict) -> None:
    """
    Append one completed RAG run to the CSV file.
    """

    fields = [
        "timestamp",
        "query_id",
        "question",
        "top_k",
        "relevant_retrieved",
        "first_relevant_rank",
        "retrieved_doc_ids",
        "retrieval_scores",
        "context_characters",
        "generation_time_sec",
        "citation_count",
        "citations",
        "answer",
    ]

    file_exists = CSV_FILE.exists()

    with CSV_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_jsonl(record: dict) -> None:
    """
    Save the complete run in JSONL format.

    JSONL is useful because nested retrieval information
    can be preserved without flattening everything into CSV.
    """

    with JSONL_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    print()
    print("V3 Experiment 2 - RAG Top-K Pilot")
    print("---------------------------------")

    corpus, queries, qrels = load_scifact()

    # -----------------------------------------------------
    # Build query -> relevant document mapping
    # -----------------------------------------------------

    relevant = defaultdict(set)

    for qid, docid, score in qrels:

        if score > 0:
            relevant[str(qid)].add(str(docid))

    # -----------------------------------------------------
    # Select reproducible queries
    # -----------------------------------------------------

    candidate_query_ids = sorted(
        qid
        for qid in relevant
        if qid in queries and relevant[qid]
    )

    selected_query_ids = candidate_query_ids[:NUM_QUERIES]

    print(
        f"Selected queries: "
        f"{len(selected_query_ids)}"
    )

    print(
        f"K values: "
        f"{K_VALUES}"
    )

    print(
        f"Total planned RAG runs: "
        f"{len(selected_query_ids) * len(K_VALUES)}"
    )

    print()

    # Load retriever/model once.
    retriever = Retriever()

    completed = 0

    total_runs = (
        len(selected_query_ids)
        * len(K_VALUES)
    )

    # -----------------------------------------------------
    # Experiment loop
    # -----------------------------------------------------

    for qid in selected_query_ids:

        question = queries[qid]

        gold_docs = relevant[qid]

        print("=" * 70)

        print(
            f"Query ID: {qid}"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Relevant documents: "
            f"{sorted(gold_docs)}"
        )

        print("=" * 70)

        for k in K_VALUES:

            completed += 1

            print()
            print(
                f"Run {completed}/{total_runs} "
                f"| Query={qid} "
                f"| K={k}"
            )

            # ---------------------------------------------
            # Retrieval
            # ---------------------------------------------

            results = retriever.search(
                question,
                k=k,
            )

            rank = first_relevant_rank(
                results,
                gold_docs,
            )

            relevant_retrieved = (
                rank is not None
            )

            # ---------------------------------------------
            # Build RAG context
            # ---------------------------------------------

            context = build_context(
                results
            )

            prompt = build_prompt(
                question,
                context,
            )

            context_characters = len(context)

            # ---------------------------------------------
            # Generation
            # ---------------------------------------------

            print(
                f"Relevant retrieved: "
                f"{relevant_retrieved}"
            )

            print(
                f"First relevant rank: "
                f"{rank}"
            )

            print(
                f"Context characters: "
                f"{context_characters:,}"
            )

            print(
                "Generating..."
            )

            start = time.perf_counter()

            try:

                answer = call_ollama(
                    prompt
                )

                error = None

            except Exception as exc:

                answer = ""

                error = str(exc)

            elapsed = (
                time.perf_counter()
                - start
            )

            citations = extract_citations(
                answer
            )

            # ---------------------------------------------
            # Build result record
            # ---------------------------------------------

            retrieved_doc_ids = [
                str(r["doc_id"])
                for r in results
            ]

            retrieval_scores = [
                round(
                    float(r["score"]),
                    4,
                )
                for r in results
            ]

            timestamp = (
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
            )

            csv_row = {
                "timestamp": timestamp,
                "query_id": qid,
                "question": question,
                "top_k": k,
                "relevant_retrieved":
                    relevant_retrieved,
                "first_relevant_rank":
                    rank if rank is not None else "",
                "retrieved_doc_ids":
                    "|".join(retrieved_doc_ids),
                "retrieval_scores":
                    "|".join(
                        str(x)
                        for x in retrieval_scores
                    ),
                "context_characters":
                    context_characters,
                "generation_time_sec":
                    round(elapsed, 2),
                "citation_count":
                    len(citations),
                "citations":
                    "|".join(
                        str(x)
                        for x in citations
                    ),
                "answer":
                    answer.replace(
                        "\n",
                        " ",
                    ),
            }

            json_record = {
                "timestamp": timestamp,
                "query_id": qid,
                "question": question,
                "top_k": k,
                "gold_documents":
                    sorted(gold_docs),
                "relevant_retrieved":
                    relevant_retrieved,
                "first_relevant_rank":
                    rank,
                "context_characters":
                    context_characters,
                "generation_time_sec":
                    round(elapsed, 2),
                "citations":
                    citations,
                "error":
                    error,
                "retrieved_documents": [
                    {
                        "rank": i,
                        "doc_id":
                            str(r["doc_id"]),
                        "title":
                            r["title"],
                        "score":
                            round(
                                float(
                                    r["score"]
                                ),
                                4,
                            ),
                    }
                    for i, r
                    in enumerate(
                        results,
                        start=1,
                    )
                ],
                "answer": answer,
            }

            # ---------------------------------------------
            # Save immediately
            # ---------------------------------------------

            save_csv_row(
                csv_row
            )

            save_jsonl(
                json_record
            )

            print(
                f"Generation time: "
                f"{elapsed:.2f}s"
            )

            print(
                f"Citations: "
                f"{citations}"
            )

            if error:

                print(
                    f"ERROR: {error}"
                )

            print(
                "Saved."
            )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print()
    print("=" * 70)

    print(
        "Experiment complete."
    )

    print(
        f"CSV: {CSV_FILE}"
    )

    print(
        f"JSONL: {JSONL_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()