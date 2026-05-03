# app/llm.py

import requests
from app.config import OLLAMA_MODEL, OLLAMA_URL


def generate(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )

    response.raise_for_status()
    return response.json()["response"]


if __name__ == "__main__":
    answer = generate("What makes a movie great in one sentence?")
    print(answer)