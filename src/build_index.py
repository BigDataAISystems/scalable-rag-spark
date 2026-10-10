"""Create normalized SciFact embeddings and a FAISS cosine-similarity index."""
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, INDEX_DIR
from load_scifact import load_scifact


def main():
    corpus, _, _ = load_scifact()
    doc_ids = list(corpus.keys())
    texts = [f'{corpus[i]["title"]}\n{corpus[i]["text"]}'.strip() for i in doc_ids]

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / "scifact.faiss"))
    (INDEX_DIR / "doc_ids.json").write_text(json.dumps(doc_ids), encoding="utf-8")
    print(f"Indexed {index.ntotal:,} documents with dimension {vectors.shape[1]}.")


if __name__ == "__main__":
    main()
