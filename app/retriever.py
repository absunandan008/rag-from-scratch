# app/retriever.py
from sentence_transformers import SentenceTransformer
import chromadb
import os

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from app.config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K,
)

def retrieve(query: str) -> list[dict]:
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return chunks


if __name__ == "__main__":
    results = retrieve("what makes a movie great")
    for r in results:
        print(f"distance: {r['distance']:.4f} | {r['text'][:80]}")