import httpx
import chromadb
import config

_chroma_client = None
_collection = None
_http_client = httpx.Client(
    base_url=config.OLLAMA_URL,
    timeout=60,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=20),
)


def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
    return _chroma_client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = _get_chroma_client().get_or_create_collection(
            config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_texts(text: str) -> list[float]:
    resp = _http_client.post("/api/embeddings", json={"model": config.EMBEDDING_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]
