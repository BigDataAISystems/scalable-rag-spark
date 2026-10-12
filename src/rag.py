"""V2: Retrieval-Augmented Generation using FAISS + Ollama."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from retrieve import Retriever


load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

TOP_K = 5

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RAG_LOG = RESULTS_DIR / "rag_runs.jsonl"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_context(results):
    """Convert retrieved documents into numbered evidence blocks."""

    blocks = []

    for i, result in enumerate(results, start=1):

        block = (
            f"[{i}]\n"
            f"Document ID: {result['doc_id']}\n"
            f"Title: {result['title']}\n"
            f"Evidence:\n{result['text']}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(question, context):
    """Create a grounded RAG prompt."""

    return f"""
You are answering a scientific question using retrieved evidence.

INSTRUCTIONS

1. Use only the evidence provided below.
2. Do not add unsupported facts from your own knowledge.
3. Cite evidence using [1], [2], etc.
4. If the evidence is insufficient to answer the question,
   explicitly say that the retrieved evidence is insufficient.
5. Give a concise answer followed by a short explanation.

QUESTION

{question}

RETRIEVED EVIDENCE

{context}

ANSWER
""".strip()


def call_ollama(prompt):
    """Send the grounded prompt to the local Ollama server."""

    endpoint = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        endpoint,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"].strip()


def save_run(question, results, answer, elapsed):
    """Save the RAG run for later evaluation."""

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "version": "v2",
        "dataset": "SciFact",
        "embedding_model": "all-MiniLM-L6-v2",
        "retriever": "FAISS",
        "top_k": TOP_K,
        "llm_provider": "ollama",
        "llm_model": MODEL,
        "question": question,
        "retrieved_documents": [
            {
                "rank": i,
                "doc_id": r["doc_id"],
                "title": r["title"],
                "score": round(float(r["score"]), 4)
            }
            for i, r in enumerate(results, start=1)
        ],
        "answer": answer,
        "generation_time_sec": round(elapsed, 2)
    }

    with RAG_LOG.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(json.dumps(record) + "\n")


def main():

    print("\nScalable RAG with Apache Spark - V2")
    print("-----------------------------------")
    print(f"Generator: Ollama / {MODEL}")
    print(f"Retriever: FAISS")
    print(f"Top-k: {TOP_K}\n")

    question = input("Question: ").strip()

    if not question:
        print("No question entered.")
        return

    print("\nRetrieving evidence...")

    retriever = Retriever()

    results = retriever.search(
        question,
        k=TOP_K
    )

    print("\nRetrieved evidence:")

    for i, result in enumerate(results, start=1):

        print(
            f"[{i}] "
            f"{result['score']:.4f} - "
            f"{result['title']}"
        )

    context = build_context(results)

    prompt = build_prompt(
        question,
        context
    )

    print("\nGenerating grounded answer...\n")

    start = time.perf_counter()

    answer = call_ollama(prompt)

    elapsed = time.perf_counter() - start

    print("Answer")
    print("------")
    print(answer)

    print("\nSources")
    print("-------")

    for i, result in enumerate(results, start=1):

        print(
            f"[{i}] "
            f"{result['doc_id']} - "
            f"{result['title']}"
        )

    print(
        f"\nGeneration time: "
        f"{elapsed:.2f} seconds"
    )

    save_run(
        question,
        results,
        answer,
        elapsed
    )

    print(
        f"Run logged to: "
        f"{RAG_LOG}"
    )


if __name__ == "__main__":
    main()