# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.retriever import retrieve
from app.prompt import build_prompt
from app.llm import generate

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    chunks = retrieve(request.question)
    prompt = build_prompt(request.question, chunks)
    answer = generate(prompt)

    return QueryResponse(
        answer=answer,
        sources=chunks,
    )