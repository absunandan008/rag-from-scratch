# rag-from-scratch

Local RAG app — no internet, no API keys.

## Stack
- ChromaDB       — vector store (local, persistent)
- sentence-transformers (all-MiniLM-L6-v2) — embeddings
- Ollama         — local LLM
- FastAPI        — query API

## Setup

1. Using uv as env manager and installing dependencies
```bash
    uv init rag-from-scratch
    cd rag-from-scratch
    uv venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    uv add chromadb sentence-transformers fastapi uvicorn requests datasets
```

2. Pull Ollama model (one-time)
```bash
   ollama pull llama3.2
```

3. Download + ingest dataset (one-time)
```bash
   python scripts/download_dataset.py
   python -m app.ingest
```

4. Start the API
```bash
   uvicorn app.api:app --reload
```

5. Query it
```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What makes a movie review positive?"}'
```

## Project structure
```
rag-from-scratch/
├── app/
│   ├── config.py       # model names, chunk size, paths
│   ├── ingest.py       # load → chunk → embed → store
│   ├── retriever.py    # embed query → top-k chunks
│   ├── prompt.py       # build prompt from chunks + question
│   ├── llm.py          # call Ollama via requests
│   └── api.py          # FastAPI /query endpoint
├── scripts/
│   └── download_dataset.py
├── data/raw/           # gitignored
├── vectorstore/        # gitignored
└── pyproject.toml
```

## Notes
- No LangChain — every step is explicit and readable
- Re-run ingest if you change CHUNK_SIZE or swap datasets
- `vectorstore/` and `data/raw/` are gitignored (too large to commit)