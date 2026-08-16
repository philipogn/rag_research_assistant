from functools import lru_cache
import json
from embed import embed_texts, get_collection
import config
import httpx

_http_client = httpx.Client(base_url=config.OLLAMA_URL, timeout=120)

SYSTEM_PROMPT = (
    "You are a research assistant that helps users understand academic papers. "
    "You will be given excerpts retrieved from one or more papers, each labeled with its source paper and page number "
    "in the form '(paper:ID, page:N)', followed by a question.\n\n"

    "Rules:\n"
    "1. Answer using only the information in the provided excerpts. Do not rely on outside or prior knowledge about the topic.\n"
    "2. If the excerpts do not contain enough information to answer the question, say so explicitly (e.g. \"The provided excerpts don't contain enough information to answer this.\") instead of guessing.\n"
    "3. Every sentence that states a fact or finding from the excerpts MUST end with a citation in the exact form "
    "(ID, N), using the literal ID and N values from that excerpt's '(paper:ID, page:N)' label - never invent, paraphrase, or drop the ID. "
    "Do not write a factual sentence without one. Sentences that ask a clarifying question or state that the excerpts are insufficient do not need a citation.\n"
    "4. If different excerpts disagree or come from different papers, point that out rather than silently merging them.\n"
    "5. Be concise and precise; prefer direct answers over restating the excerpts.\n\n"

    "Respond with ONLY a JSON object with exactly these fields:\n"
    "answer (the full text answer, written according to the rules above, with inline (ID, page) citations on every factual sentence),\n"
    "sources_used (a JSON array of {\"paper\": ID, \"page\": N} objects, one per excerpt you actually drew on "
    "to write the answer - omit excerpts you were given but did not use; empty array if you declined to answer).\n"
    "No text outside the JSON object.\n\n"

    "Example:\n"
    "Excerpts:\n"
    "BERT is pretrained on masked language modeling and next sentence prediction. (paper:smith2023, page:3)\n\n"
    "Fine-tuning uses a learning rate of 2e-5 for 3 epochs. (paper:smith2023, page:4)\n\n"
    "Question:\n"
    "How is the model trained?\n\n"
    "Output:\n"
    "{\"answer\": \"The model is first pretrained using masked language modeling and next sentence prediction "
    "(smith2023, 3). It is then fine-tuned with a learning rate of 2e-5 for 3 epochs (smith2023, 4).\", "
    "\"sources_used\": [{\"paper\": \"smith2023\", \"page\": 3}, {\"paper\": \"smith2023\", \"page\": 4}]}\n\n"

    "6. You may also be given prior conversation turns before the Context. Use them only to understand what the current question is referring to (e.g. pronouns, \"the paper\", \"it\"); "
    "never treat them as a source of facts, every factual claim must still come from, and be cited to, the excerpts."
)

CONDENSE_SYSTEM_PROMPT = (
    "Given a conversation history and a follow up question, rewrite the follow-up question as a standalone "
    "question that includes any necessary context from the history (e.g. paper names or topics being discussed). "
    "If the follow-up question is already standalone, return it unchanged. "
    "Respond with ONLY the rewritten question - no explanation, no quotes, no extra text."
)


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    turns = history[-config.HISTORY_TURNS:]
    lines = []
    for message in turns:
        role = "User" if message.get("role") in ("user", "human") else "Assistant"
        content = (message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)

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


def condense_query(query: str, history: list[dict] | None, model: str = config.GENERATION_MODEL) -> str:
    """Rewrites a follow-up question into a standalone one using prior chat turns, for retrieval."""
    history_text = _format_history(history)
    if not history_text:
        return query

    response = _http_client.post(
        "/api/generate",
        json={
            "model": model,
            "system": CONDENSE_SYSTEM_PROMPT,
            "prompt": f"Conversation history:\n{history_text}\n\nFollow-up question:\n{query}",
            "stream": False,
            "options": {"temperature": 0.0},
        },
    )
    response.raise_for_status()
    rewritten = response.json()["response"].strip().strip('"')
    return rewritten or query


def _build_prompt(query: str, context: list[dict], history: list[dict] | None = None) -> str:
    context_chunks = []
    for chunk in context:
        context_chunks.append(f"{chunk['text']} (paper:{chunk['paper']}, page:{chunk['page']})")
    full_context = "\n\n".join(context_chunks)

    parts = []
    history_text = _format_history(history)
    if history_text:
        parts.append(f"Conversation history:\n{history_text}")
    parts.append(f"Context:\n{full_context}")
    parts.append(f"Question:\n{query}")
    return "\n\n".join(parts)


def generate_response(query: str, context: list[dict], model: str = config.GENERATION_MODEL, history: list[dict] | None = None) -> dict:
    """
    Builds full system prompt and generates an answer
    Returns {"answer": str, "sources_used": list[dict]}. On JSON parse failure, falls back to the raw model output as the answer with an empty sources_used
    """
    prompt = _build_prompt(query, context, history)
    response = _http_client.post(
        "/api/generate",
        json={
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3}
        },
    )
    response.raise_for_status()
    raw = response.json()["response"]

    try:
        parsed = json.loads(raw)
        return {
            "answer": parsed.get("answer", ""),
            "sources_used": parsed.get("sources_used") or [],
        }
    except json.JSONDecodeError:
        return {"answer": raw, "sources_used": []}

if __name__ == "__main__":
    query = "what are the results of paper Political Leaning and Politicalness Classification of Texts"
    context = retrieve_relevant_chunks(query)
    result = generate_response(query, context)
    print(result["answer"])
    print("Sources used:", result["sources_used"])