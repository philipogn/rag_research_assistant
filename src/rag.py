from functools import lru_cache
from embed import embed_texts, get_collection
import config
import httpx

_http_client = httpx.Client(base_url=config.OLLAMA_URL, timeout=120)

@lru_cache(maxsize=256)
def _cached_query_embedding(query: str) -> tuple[float, ...]:
    # keyed on exact query text, tuple keeps the cached vector immutable
    return tuple(embed_texts(query))


def retrieve_relevant_chunks(query: str, n_results: int=config.N_RESULTS) -> list[dict]:
    collection = get_collection()
    query_embedding = list(_cached_query_embedding(query))
    # retrieves relevant results based on similarity ranking
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=n_results,
        # include text from doc, metadata in results
        include=["documents", "metadatas"]
    )
    relevant_chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        relevant_chunks.append({
            "text": doc,
            "paper": meta.get("paper", 0),
            "page": meta.get("page", 0)
        })
    return relevant_chunks

SYSTEM_PROMPT = (
    "You are a research assistant that helps users understand academic papers. "
    "You will be given excerpts retrieved from one or more papers, each labeled with its source paper and page number, followed by a question.\n\n"
    "Rules:\n"
    "1. Answer using only the information in the provided excerpts. Do not rely on outside or prior knowledge about the topic.\n"
    "2. If the excerpts do not contain enough information to answer the question, say so explicitly (e.g. \"The provided excerpts don't contain enough information to answer this.\") instead of guessing.\n"
    "3. When you state a fact or finding from the excerpts, cite its source inline as (paper, page).\n"
    "4. If different excerpts disagree or come from different papers, point that out rather than silently merging them.\n"
    "5. Be concise and precise; prefer direct answers over restating the excerpts."
)

def generate_response(query: str, context: list[dict], model=config.GENERATION_MODEL):
    context_chunks = []
    for chunk in context:
        context_chunks.append(f"{chunk['text']} (paper:{chunk['paper']}, page:{chunk['page']})")
    full_context = "\n\n".join(context_chunks)
    prompt = f"Context:\n{full_context}\n\nQuestion:\n{query}"

    response = _http_client.post(
        "/api/generate",
        json={
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    query = "what are the results of paper Political Leaning and Politicalness Classification of Texts"
    context = retrieve_relevant_chunks(query)
    response = generate_response(query, context)
    print(response)