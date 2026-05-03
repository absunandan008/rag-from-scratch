# rag-from-scratch

Local RAG app — no internet, no API keys.

## Stack
- ChromaDB       — vector store (local, persistent)
- sentence-transformers (all-MiniLM-L6-v2) — embeddings
- Ollama         — local LLM
- FastAPI        — query API

## Setup

1. Using uv as env manager
```
    uv init rag-from-scratch
    cd rag-from-scratch
    uv venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

2. Install deps
    pip install chromadb sentence-transformers fastapi uvicorn requests datasets