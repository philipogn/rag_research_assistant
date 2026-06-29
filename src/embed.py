import httpx
import chromadb
from pathlib import Path

EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://ollama:11434"
COLLECTION_NAME = "research_papers"


def get_collection() -> chromadb.Collection:
    client = chromadb.HttpClient(host="chromadb", port=8000)
    return client.get_or_create_collection(COLLECTION_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"])
    return embeddings
