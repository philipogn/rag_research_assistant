import httpx
import chromadb
import config


def get_collection() -> chromadb.Collection:
    client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
    return client.get_or_create_collection(config.COLLECTION_NAME)


def embed_texts(text: str) -> list[float]:
    resp = httpx.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={"model": config.EMBEDDING_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]
