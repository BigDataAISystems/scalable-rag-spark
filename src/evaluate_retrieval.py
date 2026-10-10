"""Simple Recall@k evaluation using SciFact test relevance judgments."""
from collections import defaultdict
from load_scifact import load_scifact
from retrieve import Retriever


def main(k=5):
    _, queries, qrels = load_scifact()
    relevant = defaultdict(set)
    for qid, docid, score in qrels:
        if score > 0:
            relevant[qid].add(docid)

    retriever = Retriever()
    tested = hits = 0
    for qid, gold in relevant.items():
        if qid not in queries or not gold:
            continue
        predicted = {r["doc_id"] for r in retriever.search(queries[qid], k=k)}
        tested += 1
        hits += int(bool(predicted & gold))

    print(f"Queries evaluated: {tested}")
    print(f"Recall@{k} (at least one relevant document retrieved): {hits/tested:.4f}")


if __name__ == "__main__":
    main()
