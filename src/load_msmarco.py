"""
V4 Dataset Loader

Load the BEIR MS MARCO corpus, queries, and relevance judgments
from Hugging Face.

This script performs a small smoke test only.
It does NOT build embeddings or a FAISS index.
"""

from __future__ import annotations

from datasets import load_dataset


DATASET_NAME = "BeIR/msmarco"
QRELS_DATASET_NAME = "BeIR/msmarco-qrels"


def load_msmarco():

    print()
    print("Loading MS MARCO corpus...")

    corpus = load_dataset(
        DATASET_NAME,
        "corpus",
        split="corpus",
    )

    print("Loading MS MARCO queries...")

    queries = load_dataset(
        DATASET_NAME,
        "queries",
        split="queries",
    )

    print("Loading MS MARCO validation qrels...")

    qrels = load_dataset(
        QRELS_DATASET_NAME,
        split="validation",
    )

    return corpus, queries, qrels


def main():

    corpus, queries, qrels = load_msmarco()

    print()
    print("MS MARCO Dataset")
    print("================")

    print(f"Corpus documents: {len(corpus):,}")
    print(f"Queries:          {len(queries):,}")
    print(f"Test qrels:       {len(qrels):,}")

    print()
    print("Corpus columns:")
    print(corpus.column_names)

    print()
    print("Query columns:")
    print(queries.column_names)

    print()
    print("Qrel columns:")
    print(qrels.column_names)

    print()
    print("Example documents")
    print("-----------------")

    for i in range(3):

        row = corpus[i]

        print()
        print(f"Document {i + 1}")
        print(row)

    print()
    print("Example queries")
    print("---------------")

    for i in range(3):

        row = queries[i]

        print()
        print(f"Query {i + 1}")
        print(row)

    print()
    print("Example qrels")
    print("-------------")

    for i in range(3):

        print(qrels[i])


if __name__ == "__main__":
    main()