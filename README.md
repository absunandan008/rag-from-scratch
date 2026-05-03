# rag-from-scratch

A local RAG (Retrieval-Augmented Generation) pipeline built from scratch — no LangChain, no API keys, no internet required at query time.

Given a question, the app finds the most semantically similar chunks from a local vector store and uses a local LLM to generate a grounded answer. Every step of the pipeline is explicit and readable.

## How it works

```
User question → embed question → retrieve top-k chunks from ChromaDB → build prompt → Ollama LLM → answer
```

1. Documents are chunked and embedded using `sentence-transformers`
2. Embeddings are stored locally in ChromaDB
3. At query time, the question is embedded and compared against stored vectors using cosine similarity
4. The top-k most similar chunks are stuffed into a prompt
5. Ollama runs the LLM locally and returns a grounded answer

## Stack
- ChromaDB — vector store (local, persistent)
- sentence-transformers (all-MiniLM-L6-v2) — embeddings
- Ollama (llama3.2) — local LLM
- FastAPI + uvicorn — query API

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) — Python package manager
- [Ollama](https://ollama.com) — local LLM runner, must be installed and running

## Setup

1. Clone the repo and install dependencies
```bash
git clone https://github.com/absunandan008/rag-from-scratch.git
cd rag-from-scratch
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv add chromadb sentence-transformers fastapi uvicorn requests datasets
```

2. Pull the LLM (one-time, ~2GB)
```bash
ollama pull llama3.2
```

3. Download and ingest the dataset (one-time)
```bash
python scripts/download_dataset.py
python -m app.ingest
```
This downloads the `rotten_tomatoes` dataset from HuggingFace, chunks the reviews, embeds them using `all-MiniLM-L6-v2`, and stores everything in ChromaDB locally under `vectorstore/`.

4. Start Ollama (if not already running)
```bash
ollama serve
```

5. Start the API
```bash
uvicorn app.api:app --reload
```
API is now live at `http://localhost:8000`

## Testing the app

Check the API is up:
```bash
curl http://localhost:8000/health
```

Ask a question:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What makes a movie great?"}'
```

Example response:
```json
{
  "answer": "A great movie is marked by acute writing, splendid performances, and a solid refined piece of moviemaking imbued with passion and attitude.",
  "sources": [
    {
      "text": "a distinguished and thoughtful film , marked by acute writing and a host of splendid performances .",
      "metadata": { "label": 1, "source_index": 3143 },
      "distance": 0.6850
    }
  ]
}
```

The `sources` field shows exactly which chunks the answer was grounded in, along with their similarity distance (lower = more similar) and metadata (`label: 1` = positive review, `label: 0` = negative).

## Configuration

All tunable parameters are in `app/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `OLLAMA_MODEL` | `llama3.2` | Local LLM via Ollama |
| `CHUNK_SIZE` | `300` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `3` | Chunks retrieved per query |

## Project structure
```
rag-from-scratch/
├── app/
│   ├── config.py       # all constants and tunable parameters
│   ├── ingest.py       # load → chunk → embed → store in ChromaDB
│   ├── retriever.py    # embed query → cosine search → top-k chunks
│   ├── prompt.py       # build prompt from chunks + question
│   ├── llm.py          # call Ollama via HTTP, return response
│   └── api.py          # FastAPI /query and /health endpoints
├── scripts/
│   └── download_dataset.py  # one-time HuggingFace dataset download
├── data/raw/           # gitignored — raw dataset lives here
├── vectorstore/        # gitignored — ChromaDB persists here
└── pyproject.toml
```

## Notes
- No LangChain — every step of the pipeline is explicit and readable
- Re-run `python -m app.ingest` if you change `CHUNK_SIZE` or swap datasets
- `vectorstore/` and `data/raw/` are gitignored — too large to commit
- The embedding model (~80MB) is downloaded on first ingest run and cached automatically