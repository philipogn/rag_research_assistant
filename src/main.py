import sys
from pathlib import Path
import httpx
from fastapi import Body, FastAPI

import config
from rag import condense_query, generate_response, retrieve_relevant_chunks

sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(title="RAG Ollama Bridge")

NO_CONTEXT_MESSAGE = "I couldn't find relevant information in my knowledge base."


def extract_sources(chunks: list[dict]) -> list[str]:
    return sorted({chunk["paper"] for chunk in chunks if chunk.get("paper")})


def split_history(messages: list[dict]) -> tuple[list[dict], str]:
    """Splits messages into (previous chat, current query), taking the last user message as current"""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") in ("user", "human"):
            return messages[:i], messages[i].get("content", "") or ""
    return [], ""


def answer_query(messages: list[dict], model: str) -> tuple[str, list[str]]:
    history, query = split_history(messages)

    search_query = condense_query(query, history, model=model) if history else query
    chunks = retrieve_relevant_chunks(search_query)
    if not chunks:
        return NO_CONTEXT_MESSAGE, []

    result = generate_response(query, chunks, model=model, history=history)
    answer = result["answer"]
    sources = extract_sources(chunks)
    if sources:
        answer += f"\n\nSources: {', '.join(sources)}"
    return answer, sources


@app.post("/api/chat")
def chat(payload: dict = Body(default_factory=dict)):
    """Ollama-compatible /api/chat endpoint (with RAG)."""
    model = payload.get("model") or config.GENERATION_MODEL
    messages = payload.get("messages") or []
    if not messages and payload.get("prompt"):
        messages = [{"role": "user", "content": payload["prompt"]}]

    answer, _sources = answer_query(messages, model)

    return {
        "model": model,
        "message": {"role": "assistant", "content": answer},
        "done": True,
        "done_reason": "stop",
    }


@app.get("/api/tags")
def tags():
    """Lists available models for Open WebUI."""
    response = httpx.get(f"{config.OLLAMA_URL}/api/tags")
    response.raise_for_status()
    return response.json()


@app.get("/api/ps")
def ps():
    """Lists currently loaded/running models for Open WebUI."""
    response = httpx.get(f"{config.OLLAMA_URL}/api/ps")
    response.raise_for_status()
    return response.json()


@app.get("/api/version")
def version():
    """Proxy Ollama version endpoint for Open WebUI."""
    response = httpx.get(f"{config.OLLAMA_URL}/api/version")
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
