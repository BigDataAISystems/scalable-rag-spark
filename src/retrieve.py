"""Dense retrieval over SciFact using Sentence Transformers + FAISS."""
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, INDEX_DIR, TOP_K
from load_scifact import load_scifact


class Retriever:
    def __init__(self):
        self.corpus, _, _ = load_scifact()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(str(INDEX_DIR / "scifact.faiss"))
        self.doc_ids = json.loads((INDEX_DIR / "doc_ids.json").read_text(encoding="utf-8"))

    def search(self, query: str, k: int = TOP_K):
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, positions = self.index.search(np.asarray(q, dtype="float32"), k)
        results = []
        for score, pos in zip(scores[0], positions[0]):
            doc_id = self.doc_ids[pos]
            doc = self.corpus[doc_id]
            results.append({"doc_id": doc_id, "score": float(score), **doc})
        return results


if __name__ == "__main__":
    r = Retriever()
    query = input("Query: ").strip()
    for hit in r.search(query):
        print(f'\n[{hit["score"]:.4f}] {hit["doc_id"]} - {hit["title"]}')
        print(hit["text"][:500])
