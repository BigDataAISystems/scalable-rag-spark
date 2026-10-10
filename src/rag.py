"""Minimal RAG: retrieve evidence, build context, and optionally call an LLM."""
import os
from dotenv import load_dotenv
from retrieve import Retriever

load_dotenv()


def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[Source {i} | doc_id={h['doc_id']}]\n{h['title']}\n{h['text']}")
    return "\n\n".join(blocks)


def answer(query: str, k: int = 5):
    hits = Retriever().search(query, k=k)
    context = build_context(hits)

    if not os.getenv("OPENAI_API_KEY"):
        return {"answer": None, "context": context, "sources": hits}

    from openai import OpenAI
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    prompt = f"""Answer the question using only the supplied sources.
If the sources do not support an answer, say that the evidence is insufficient.
Cite source numbers in square brackets.

Question: {query}

Sources:
{context}
"""
    response = client.responses.create(model=model, input=prompt)
    return {"answer": response.output_text, "context": context, "sources": hits}


if __name__ == "__main__":
    query = input("Question: ").strip()
    result = answer(query)
    if result["answer"]:
        print("\nANSWER\n", result["answer"])
    else:
        print("\nOPENAI_API_KEY is not set. Retrieval works; generated answer skipped.")
    print("\nSOURCES")
    for h in result["sources"]:
        print(f'- {h["doc_id"]}: {h["title"]} (score={h["score"]:.4f})')
