# app/prompt.py

def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join([chunk["text"] for chunk in chunks])

    prompt = f"""You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""

    return prompt


if __name__ == "__main__":
    test_chunks = [
        {"text": "the rock is destined to be the 21st century's new conan"},
        {"text": "a great idea becomes a not-great movie"},
    ]
    print(build_prompt("what makes a movie great", test_chunks))