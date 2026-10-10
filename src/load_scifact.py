"""Load the BEIR SciFact corpus, queries, and relevance judgments via Hugging Face."""
from datasets import load_dataset


def load_scifact():
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/scifact-qrels", split="test")

    corpus = {
        str(row["_id"]): {
            "title": row.get("title", ""),
            "text": row.get("text", ""),
        }
        for row in corpus_ds
    }
    queries = {str(row["_id"]): row["text"] for row in queries_ds}
    qrels = [(str(r["query-id"]), str(r["corpus-id"]), int(r["score"])) for r in qrels_ds]
    return corpus, queries, qrels


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    print(f"Corpus documents: {len(corpus):,}")
    print(f"Queries: {len(queries):,}")
    print(f"Qrels: {len(qrels):,}")
