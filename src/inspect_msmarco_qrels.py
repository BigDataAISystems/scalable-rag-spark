"""Inspect MS MARCO qrel statistics for V4 dataset design."""

from collections import defaultdict
from statistics import mean, median

from datasets import load_dataset


QRELS_DATASET = "BeIR/msmarco-qrels"


def inspect_split(split):

    qrels = load_dataset(
        QRELS_DATASET,
        split=split,
    )

    relevant = defaultdict(set)

    positive_rows = 0

    for row in qrels:

        if int(row["score"]) <= 0:
            continue

        qid = str(row["query-id"])
        docid = str(row["corpus-id"])

        relevant[qid].add(docid)

        positive_rows += 1

    counts = [
        len(docs)
        for docs in relevant.values()
    ]

    unique_docs = set()

    for docs in relevant.values():
        unique_docs.update(docs)

    print()
    print(f"Split: {split}")
    print("=" * 40)

    print(
        f"Total qrel rows:       "
        f"{len(qrels):,}"
    )

    print(
        f"Positive qrel rows:    "
        f"{positive_rows:,}"
    )

    print(
        f"Positive queries:      "
        f"{len(relevant):,}"
    )

    print(
        f"Unique positive docs:  "
        f"{len(unique_docs):,}"
    )

    if counts:

        print(
            f"Relevant docs/query:"
        )

        print(
            f"  min:    {min(counts)}"
        )

        print(
            f"  median: {median(counts):.1f}"
        )

        print(
            f"  mean:   {mean(counts):.2f}"
        )

        print(
            f"  max:    {max(counts)}"
        )


def main():

    for split in [
        "validation",
        "test",
    ]:

        inspect_split(split)


if __name__ == "__main__":
    main()