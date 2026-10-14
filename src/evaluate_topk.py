"""V3 Experiment 1: Evaluate retrieval effectiveness across Top-K values."""

from collections import defaultdict
from pathlib import Path
from datetime import datetime
import csv
import time

from load_scifact import load_scifact
from retrieve import Retriever


K_VALUES = [1, 3, 5, 10, 20]

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "topk_experiment.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():

    _, queries, qrels = load_scifact()

    relevant = defaultdict(set)

    for qid, docid, score in qrels:
        if score > 0:
            relevant[qid].add(docid)

    retriever = Retriever()

    results = []

    print("\nV3 Experiment 1 — Retrieval Depth")
    print("---------------------------------")

    for k in K_VALUES:

        tested = 0
        hits = 0

        start = time.perf_counter()

        for qid, gold in relevant.items():

            if qid not in queries or not gold:
                continue

            predicted = {
                r["doc_id"]
                for r in retriever.search(
                    queries[qid],
                    k=k
                )
            }

            tested += 1
            hits += int(bool(predicted & gold))

        elapsed = time.perf_counter() - start

        recall = hits / tested if tested else 0.0

        results.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": "SciFact",
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_index": "FAISS",
            "top_k": k,
            "queries_evaluated": tested,
            "recall_at_k": round(recall, 4),
            "evaluation_time_sec": round(elapsed, 2),
        })

        print(
            f"K={k:<2} | "
            f"Recall@{k}: {recall:.4f} | "
            f"Time: {elapsed:.2f}s"
        )

    fields = [
        "timestamp",
        "dataset",
        "embedding_model",
        "vector_index",
        "top_k",
        "queries_evaluated",
        "recall_at_k",
        "evaluation_time_sec",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()