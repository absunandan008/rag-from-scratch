import json
from sentence_transformers import SentenceTransformer
import chromadb

import os

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from app.config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    
    return chunks

def ingest():
    #load raw data
    print("loading datasets...")
    rows = []
    with open("data/raw/rotten_tomatoes.json1", "r") as f:
        for line in f:
            rows.append(json.loads(line))
    
    #set up embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL} ")
    model = SentenceTransformer(EMBEDDING_MODEL)

    #setup chroma db
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Chunking and embedding...")
    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadata = []

    for i,row in enumerate(rows):
        text = row["text"]
        label = row["label"]
        chunks = chunk_text(text)

        for j, chunk in enumerate(chunks):
            chunk_id = f"doc_{i}_chunk_{j}"
            embedding = model.encode(chunk).tolist()

            all_chunks.append(chunk)
            all_embeddings.append(embedding)
            all_ids.append(chunk_id)
            all_metadata.append({"label": label, "source_index": i})
        
    print(f"Storing {len(all_chunks)} chunks in ChromaDB...")
    BATCH_SIZE = 500
    for start in range(0, len(all_chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            documents=all_chunks[start:end],
            embeddings=all_embeddings[start:end],
            ids=all_ids[start:end],
            metadatas=all_metadata[start:end],
        )
        print(f"Upserted batch {start} to {end}")

    print(f"Done {len(all_chunks)} chunks stored in collection {COLLECTION_NAME}")

if __name__ == "__main__":
    ingest()
