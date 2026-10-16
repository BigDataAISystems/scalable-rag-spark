"""
V4 - MS MARCO Nested Dataset Preparation

Creates deterministic, qrel-preserving, nested MS MARCO subsets
for corpus-scaling experiments.

Design
------
Dataset:
    BEIR MS MARCO

Qrels:
    validation split

Evaluation queries:
    300 randomly selected positive queries

Random seed:
    42

Nested corpus sizes:
    1,000
    25,000
    50,000
    100,000
    250,000

Important property:

    S_1K ⊂ S_25K ⊂ S_50K ⊂ S_100K ⊂ S_250K

Every subset contains:
    - exactly the same evaluation queries
    - every positively relevant document for those queries
    - an increasing number of deterministic distractor documents

No embeddings or FAISS indexes are created here.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from load_msmarco import load_msmarco


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V4_DIR = ROOT / "data" / "v4"

NUM_QUERIES = 300

RANDOM_SEED = 42

SUBSET_SIZES = [
    1_000,
    25_000,
    50_000,
    100_000,
    250_000,
]


# ============================================================
# Helper functions
# ============================================================

def subset_name(size: int) -> str:
    """
    Convert subset size into a readable directory name.

    Examples:
        1000   -> msmarco_1k
        25000  -> msmarco_25k
        250000 -> msmarco_250k
    """

    if size % 1000 == 0:
        return f"msmarco_{size // 1000}k"

    return f"msmarco_{size}"


def write_jsonl(path: Path, records) -> None:
    """
    Write records to JSONL.
    """

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    start_total = time.perf_counter()

    print()
    print("V4 - MS MARCO Nested Dataset Preparation")
    print("=========================================")

    print()
    print(f"Evaluation queries: {NUM_QUERIES}")
    print(f"Random seed:        {RANDOM_SEED}")
    print(
        "Subset sizes:       "
        + ", ".join(
            f"{size:,}"
            for size in SUBSET_SIZES
        )
    )

    # --------------------------------------------------------
    # Load source dataset
    # --------------------------------------------------------

    corpus, queries, qrels = load_msmarco()

    V4_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build positive relevance mapping
    # --------------------------------------------------------

    print()
    print("Building positive relevance mapping...")

    relevant = defaultdict(set)

    positive_qrels = 0

    for row in qrels:

        score = int(
            row["score"]
        )

        if score <= 0:
            continue

        qid = str(
            row["query-id"]
        )

        docid = str(
            row["corpus-id"]
        )

        relevant[qid].add(
            docid
        )

        positive_qrels += 1

    print(
        f"Positive qrel rows: "
        f"{positive_qrels:,}"
    )

    print(
        f"Queries with positive relevance: "
        f"{len(relevant):,}"
    )

    # --------------------------------------------------------
    # Build query lookup
    # --------------------------------------------------------

    print()
    print("Building query lookup...")

    query_lookup = {}

    for row in queries:

        qid = str(
            row["_id"]
        )

        query_lookup[qid] = {
            "_id": qid,
            "title": row.get(
                "title",
                "",
            ) or "",
            "text": row.get(
                "text",
                "",
            ) or "",
        }

    # --------------------------------------------------------
    # Candidate evaluation queries
    # --------------------------------------------------------

    candidate_qids = sorted(
        qid
        for qid in relevant
        if qid in query_lookup
    )

    print(
        f"Candidate judged queries: "
        f"{len(candidate_qids):,}"
    )

    if len(candidate_qids) < NUM_QUERIES:

        raise RuntimeError(
            f"Requested {NUM_QUERIES} queries, "
            f"but only {len(candidate_qids)} "
            f"eligible queries are available."
        )

    # --------------------------------------------------------
    # Select deterministic random query sample
    # --------------------------------------------------------

    query_rng = random.Random(
        RANDOM_SEED
    )

    selected_qids = sorted(
        query_rng.sample(
            candidate_qids,
            NUM_QUERIES,
        )
    )

    # --------------------------------------------------------
    # Collect mandatory gold documents
    # --------------------------------------------------------

    gold_doc_ids = set()

    selected_positive_qrels = 0

    for qid in selected_qids:

        docs = relevant[qid]

        gold_doc_ids.update(
            docs
        )

        selected_positive_qrels += (
            len(docs)
        )

    print()
    print(
        f"Selected evaluation queries: "
        f"{len(selected_qids):,}"
    )

    print(
        f"Selected positive qrels: "
        f"{selected_positive_qrels:,}"
    )

    print(
        f"Unique mandatory gold documents: "
        f"{len(gold_doc_ids):,}"
    )

    smallest_subset = min(
        SUBSET_SIZES
    )

    if len(gold_doc_ids) > smallest_subset:

        raise RuntimeError(
            "The number of mandatory gold documents "
            f"({len(gold_doc_ids):,}) exceeds the "
            f"smallest subset size "
            f"({smallest_subset:,})."
        )

    # --------------------------------------------------------
    # Determine how many distractors are needed
    # --------------------------------------------------------

    largest_subset = max(
        SUBSET_SIZES
    )

    distractors_needed = (
        largest_subset
        - len(gold_doc_ids)
    )

    print()
    print(
        f"Distractors required for largest subset: "
        f"{distractors_needed:,}"
    )

    # --------------------------------------------------------
    # Reservoir-sample distractors ONCE
    #
    # We sample enough distractors for the largest corpus.
    # Prefixes of this same ordered sample are then used for
    # all smaller corpora.
    #
    # This guarantees nested subsets.
    # --------------------------------------------------------

    print()
    print(
        "Sampling distractor documents "
        "from full MS MARCO corpus..."
    )

    distractor_rng = random.Random(
        RANDOM_SEED
    )

    distractor_ids = []

    seen_candidates = 0

    scan_start = time.perf_counter()

    for row_number, row in enumerate(
        corpus,
        start=1,
    ):

        docid = str(
            row["_id"]
        )

        if docid in gold_doc_ids:
            continue

        seen_candidates += 1

        if len(distractor_ids) < distractors_needed:

            distractor_ids.append(
                docid
            )

        else:

            j = distractor_rng.randint(
                1,
                seen_candidates,
            )

            if j <= distractors_needed:

                distractor_ids[
                    j - 1
                ] = docid

        if row_number % 1_000_000 == 0:

            print(
                f"  scanned "
                f"{row_number:,} documents..."
            )

    scan_elapsed = (
        time.perf_counter()
        - scan_start
    )

    if len(distractor_ids) != distractors_needed:

        raise RuntimeError(
            "Distractor sampling failed. "
            f"Expected {distractors_needed:,}, "
            f"got {len(distractor_ids):,}."
        )

    print(
        f"Sampling complete in "
        f"{scan_elapsed:.2f} seconds."
    )

    # --------------------------------------------------------
    # Build selected document set for largest corpus
    # --------------------------------------------------------

    largest_doc_ids = (
        gold_doc_ids
        | set(distractor_ids)
    )

    if len(largest_doc_ids) != largest_subset:

        raise RuntimeError(
            f"Expected {largest_subset:,} unique "
            f"documents for largest subset, "
            f"got {len(largest_doc_ids):,}."
        )

    # --------------------------------------------------------
    # Extract largest selected corpus in ONE second scan
    #
    # Store only the selected <=250K documents in memory.
    # --------------------------------------------------------

    print()
    print(
        f"Extracting the selected "
        f"{largest_subset:,} documents..."
    )

    selected_documents = {}

    extraction_start = (
        time.perf_counter()
    )

    for row_number, row in enumerate(
        corpus,
        start=1,
    ):

        docid = str(
            row["_id"]
        )

        if docid not in largest_doc_ids:
            continue

        selected_documents[docid] = {
            "_id": docid,
            "title": row.get(
                "title",
                "",
            ) or "",
            "text": row.get(
                "text",
                "",
            ) or "",
        }

        if len(selected_documents) == largest_subset:
            break

        if row_number % 1_000_000 == 0:

            print(
                f"  scanned "
                f"{row_number:,} documents..."
            )

    extraction_elapsed = (
        time.perf_counter()
        - extraction_start
    )

    if len(selected_documents) != largest_subset:

        missing = (
            largest_doc_ids
            - set(
                selected_documents
            )
        )

        raise RuntimeError(
            f"Corpus extraction failed. "
            f"Expected {largest_subset:,}, "
            f"found {len(selected_documents):,}. "
            f"Missing: {len(missing):,}"
        )

    print(
        f"Extraction complete in "
        f"{extraction_elapsed:.2f} seconds."
    )

    # --------------------------------------------------------
    # Prepare shared query records
    # --------------------------------------------------------

    query_records = [
        query_lookup[qid]
        for qid in selected_qids
    ]

    # --------------------------------------------------------
    # Prepare shared qrel records
    # --------------------------------------------------------

    qrel_records = []

    for qid in selected_qids:

        for docid in sorted(
            relevant[qid]
        ):

            qrel_records.append(
                (
                    qid,
                    docid,
                    1,
                )
            )

    # --------------------------------------------------------
    # Create every nested subset
    # --------------------------------------------------------

    print()
    print("Writing nested subsets...")
    print()

    subset_summaries = []

    for size in SUBSET_SIZES:

        name = subset_name(
            size
        )

        subset_dir = (
            V4_DIR
            / name
        )

        subset_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        num_distractors = (
            size
            - len(gold_doc_ids)
        )

        if num_distractors < 0:

            raise RuntimeError(
                f"Subset {size:,} cannot contain "
                "all mandatory gold documents."
            )

        # Prefix of the SAME distractor sample.
        current_distractors = set(
            distractor_ids[
                :num_distractors
            ]
        )

        current_doc_ids = (
            gold_doc_ids
            | current_distractors
        )

        if len(current_doc_ids) != size:

            raise RuntimeError(
                f"Subset {name}: expected "
                f"{size:,} documents, "
                f"got {len(current_doc_ids):,}."
            )

        # --------------------------------------------
        # Corpus
        # --------------------------------------------

        corpus_file = (
            subset_dir
            / "corpus.jsonl"
        )

        with corpus_file.open(
            "w",
            encoding="utf-8",
        ) as f:

            for docid in sorted(
                current_doc_ids,
                key=int,
            ):

                record = (
                    selected_documents[
                        docid
                    ]
                )

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # --------------------------------------------
        # Queries
        # --------------------------------------------

        queries_file = (
            subset_dir
            / "queries.jsonl"
        )

        write_jsonl(
            queries_file,
            query_records,
        )

        # --------------------------------------------
        # Qrels
        # --------------------------------------------

        qrels_file = (
            subset_dir
            / "qrels.tsv"
        )

        with qrels_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(
                f,
                delimiter="\t",
            )

            writer.writerow([
                "query_id",
                "doc_id",
                "score",
            ])

            for (
                qid,
                docid,
                score,
            ) in qrel_records:

                writer.writerow([
                    qid,
                    docid,
                    score,
                ])

        # --------------------------------------------
        # Validate gold preservation
        # --------------------------------------------

        missing_gold = (
            gold_doc_ids
            - current_doc_ids
        )

        if missing_gold:

            raise RuntimeError(
                f"Subset {name} is missing "
                f"{len(missing_gold)} gold "
                f"documents."
            )

        # --------------------------------------------
        # Metadata
        # --------------------------------------------

        metadata = {
            "source_dataset":
                "BeIR/msmarco",

            "qrels_dataset":
                "BeIR/msmarco-qrels",

            "qrels_split":
                "validation",

            "subset_name":
                name,

            "subset_size":
                size,

            "evaluation_queries":
                len(selected_qids),

            "positive_qrels":
                len(qrel_records),

            "gold_documents":
                len(gold_doc_ids),

            "distractor_documents":
                len(current_distractors),

            "query_random_seed":
                RANDOM_SEED,

            "distractor_random_seed":
                RANDOM_SEED,

            "embedding_model":
                "sentence-transformers/"
                "all-MiniLM-L6-v2",

            "embedding_dimension":
                384,

            "baseline_top_k":
                5,

            "nested_subset":
                True,
        }

        metadata_file = (
            subset_dir
            / "metadata.json"
        )

        with metadata_file.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
            )

        subset_summaries.append(
            metadata
        )

        print(
            f"{name:15} "
            f"docs={size:>8,} | "
            f"gold={len(gold_doc_ids):>4,} | "
            f"distractors="
            f"{len(current_distractors):>8,}"
        )

    # --------------------------------------------------------
    # Save global experiment manifest
    # --------------------------------------------------------

    manifest = {
        "experiment":
            "V4 MS MARCO corpus scaling",

        "source_dataset":
            "BeIR/msmarco",

        "qrels_dataset":
            "BeIR/msmarco-qrels",

        "qrels_split":
            "validation",

        "evaluation_queries":
            len(selected_qids),

        "selected_query_ids":
            selected_qids,

        "gold_documents":
            len(gold_doc_ids),

        "subset_sizes":
            SUBSET_SIZES,

        "random_seed":
            RANDOM_SEED,

        "nested":
            True,

        "subsets":
            subset_summaries,
    }

    manifest_file = (
        V4_DIR
        / "msmarco_manifest.json"
    )

    with manifest_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Validate nesting
    # --------------------------------------------------------

    print()
    print("Validating nested corpus property...")
    print()

    previous_ids = None

    for size in SUBSET_SIZES:

        num_distractors = (
            size
            - len(gold_doc_ids)
        )

        current_ids = (
            gold_doc_ids
            | set(
                distractor_ids[
                    :num_distractors
                ]
            )
        )

        if previous_ids is not None:

            if not previous_ids.issubset(
                current_ids
            ):

                raise RuntimeError(
                    "Nested subset validation "
                    f"failed at size {size:,}."
                )

        previous_ids = (
            current_ids
        )

        print(
            f"{subset_name(size):15} "
            f"OK"
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - start_total
    )

    print()
    print("=" * 60)

    print(
        "MS MARCO nested subsets created successfully."
    )

    print()
    print(
        f"Evaluation queries: "
        f"{len(selected_qids):,}"
    )

    print(
        f"Gold documents:     "
        f"{len(gold_doc_ids):,}"
    )

    print(
        f"Largest corpus:     "
        f"{largest_subset:,}"
    )

    print(
        f"Total preparation:  "
        f"{total_elapsed:.2f} seconds"
    )

    print()
    print(
        f"Manifest:"
    )

    print(
        f"  {manifest_file}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()