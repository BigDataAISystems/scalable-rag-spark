"""
V3 Experiment 4:
Stratified RAG Top-K evaluation.

1. Evaluate Top-10 retrieval for SciFact queries.
2. Stratify queries according to first relevant-document rank:
       A: rank 1-3
       B: rank 4-5
       C: rank 6-10
       D: not found in Top-10
3. Select 5 reproducible queries per group.
4. Run RAG for K = 3, 5, 10.
5. Save retrieval, evidence, generation, and citation data.

IMPORTANT:
Results are saved after every run.
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


K_VALUES = [3, 5, 10]

QUERIES_PER_GROUP = 5

MAX_RANK = 10

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

CSV_FILE = (
    RESULTS_DIR
    / "rag_stratified_experiment.csv"
)

JSONL_FILE = (
    RESULTS_DIR
    / "rag_stratified_experiment.jsonl"
)

SELECTION_FILE = (
    RESULTS_DIR
    / "rag_stratified_queries.csv"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def extract_citations(answer):

    matches = re.findall(
        r"\[(\d+)\]",
        answer,
    )

    return sorted(
        {int(x) for x in matches}
    )


def first_relevant_rank(
    results,
    gold_docs,
):

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if str(
            result["doc_id"]
        ) in gold_docs:

            return rank

    return None


def classify_rank(rank):

    if rank is None:
        return "D"

    if rank <= 3:
        return "A"

    if rank <= 5:
        return "B"

    if rank <= 10:
        return "C"

    return "D"


def append_csv(
    path,
    fields,
    row,
):

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def save_jsonl(record):

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


def main():

    print()
    print(
        "V3 Experiment 4 - "
        "Stratified RAG Evaluation"
    )

    print(
        "========================================"
    )

    _, queries, qrels = load_scifact()

    relevant = defaultdict(set)

    for qid, docid, score in qrels:

        if score > 0:

            relevant[
                str(qid)
            ].add(
                str(docid)
            )

    retriever = Retriever()

    # --------------------------------------------------
    # Stage 1:
    # Determine first relevant rank for each query
    # --------------------------------------------------

    groups = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
    }

    print()
    print(
        "Analyzing retrieval ranks..."
    )

    for qid in sorted(relevant):

        if qid not in queries:
            continue

        question = queries[qid]

        results = retriever.search(
            question,
            k=MAX_RANK,
        )

        rank = first_relevant_rank(
            results,
            relevant[qid],
        )

        group = classify_rank(
            rank
        )

        groups[group].append({
            "query_id": qid,
            "question": question,
            "first_relevant_rank": rank,
        })

    # --------------------------------------------------
    # Report population
    # --------------------------------------------------

    print()
    print(
        "Retrieval strata:"
    )

    for group in [
        "A",
        "B",
        "C",
        "D",
    ]:

        print(
            f"Group {group}: "
            f"{len(groups[group])} queries"
        )

    # --------------------------------------------------
    # Select deterministic sample
    # --------------------------------------------------

    selected = []

    for group in [
        "A",
        "B",
        "C",
        "D",
    ]:

        sample = (
            groups[group]
            [:QUERIES_PER_GROUP]
        )

        if len(sample) < QUERIES_PER_GROUP:

            print(
                f"WARNING: Group {group} "
                f"contains only "
                f"{len(sample)} queries."
            )

        for item in sample:

            item = dict(item)

            item[
                "stratum"
            ] = group

            selected.append(
                item
            )

    # --------------------------------------------------
    # Save query selection
    # --------------------------------------------------

    selection_fields = [
        "stratum",
        "query_id",
        "first_relevant_rank",
        "question",
    ]

    with SELECTION_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=selection_fields,
        )

        writer.writeheader()

        writer.writerows(
            selected
        )

    print()
    print(
        f"Selected queries: "
        f"{len(selected)}"
    )

    print(
        f"K values: "
        f"{K_VALUES}"
    )

    total_runs = (
        len(selected)
        * len(K_VALUES)
    )

    print(
        f"Planned RAG runs: "
        f"{total_runs}"
    )

    print(
        f"Selection saved: "
        f"{SELECTION_FILE}"
    )

    # --------------------------------------------------
    # CSV schema
    # --------------------------------------------------

    csv_fields = [
        "timestamp",
        "stratum",
        "query_id",
        "question",
        "top_k",
        "gold_documents",
        "relevant_retrieved",
        "first_relevant_rank",
        "context_characters",
        "generation_time_sec",
        "citation_count",
        "citations",
        "answer",
    ]

    completed = 0

    # --------------------------------------------------
    # Stage 2:
    # RAG generation
    # --------------------------------------------------

    for item in selected:

        qid = item[
            "query_id"
        ]

        question = item[
            "question"
        ]

        stratum = item[
            "stratum"
        ]

        gold_docs = (
            relevant[qid]
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"Stratum: {stratum}"
        )

        print(
            f"Query: {qid}"
        )

        print(
            question
        )

        print(
            "=" * 70
        )

        for k in K_VALUES:

            completed += 1

            print()
            print(
                f"Run "
                f"{completed}/"
                f"{total_runs}"
                f" | K={k}"
            )

            results = retriever.search(
                question,
                k=k,
            )

            rank = (
                first_relevant_rank(
                    results,
                    gold_docs,
                )
            )

            retrieved = (
                rank is not None
            )

            context = (
                build_context(
                    results
                )
            )

            prompt = (
                build_prompt(
                    question,
                    context,
                )
            )

            print(
                f"Relevant retrieved: "
                f"{retrieved}"
            )

            print(
                f"First relevant rank: "
                f"{rank}"
            )

            print(
                f"Context: "
                f"{len(context):,} chars"
            )

            print(
                "Generating..."
            )

            start = (
                time.perf_counter()
            )

            try:

                answer = (
                    call_ollama(
                        prompt
                    )
                )

                error = None

            except Exception as exc:

                answer = ""

                error = str(exc)

            elapsed = (
                time.perf_counter()
                - start
            )

            citations = (
                extract_citations(
                    answer
                )
            )

            timestamp = (
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
            )

            csv_row = {

                "timestamp":
                    timestamp,

                "stratum":
                    stratum,

                "query_id":
                    qid,

                "question":
                    question,

                "top_k":
                    k,

                "gold_documents":
                    "|".join(
                        sorted(
                            gold_docs
                        )
                    ),

                "relevant_retrieved":
                    retrieved,

                "first_relevant_rank":
                    (
                        rank
                        if rank
                        is not None
                        else ""
                    ),

                "context_characters":
                    len(context),

                "generation_time_sec":
                    round(
                        elapsed,
                        2,
                    ),

                "citation_count":
                    len(
                        citations
                    ),

                "citations":
                    "|".join(
                        str(x)
                        for x
                        in citations
                    ),

                "answer":
                    answer.replace(
                        "\n",
                        " "
                    ),
            }

            append_csv(
                CSV_FILE,
                csv_fields,
                csv_row,
            )

            # ------------------------------------------
            # Save FULL evidence for future evaluation
            # ------------------------------------------

            json_record = {

                "timestamp":
                    timestamp,

                "stratum":
                    stratum,

                "query_id":
                    qid,

                "question":
                    question,

                "top_k":
                    k,

                "gold_documents":
                    sorted(
                        gold_docs
                    ),

                "relevant_retrieved":
                    retrieved,

                "first_relevant_rank":
                    rank,

                "context_characters":
                    len(context),

                "generation_time_sec":
                    round(
                        elapsed,
                        2,
                    ),

                "citations":
                    citations,

                "error":
                    error,

                "retrieved_documents": [
                    {
                        "rank": rank_number,

                        "doc_id":
                            str(
                                result[
                                    "doc_id"
                                ]
                            ),

                        "title":
                            result[
                                "title"
                            ],

                        "score":
                            round(
                                float(
                                    result[
                                        "score"
                                    ]
                                ),
                                4,
                            ),

                        "text":
                            result[
                                "text"
                            ],
                    }

                    for (
                        rank_number,
                        result,
                    )
                    in enumerate(
                        results,
                        start=1,
                    )
                ],

                "answer":
                    answer,
            }

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
                    f"ERROR: "
                    f"{error}"
                )

            print(
                "Saved."
            )

    print()
    print(
        "=" * 70
    )

    print(
        "Experiment complete."
    )

    print(
        f"CSV: "
        f"{CSV_FILE}"
    )

    print(
        f"JSONL: "
        f"{JSONL_FILE}"
    )

    print(
        f"Selection: "
        f"{SELECTION_FILE}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()