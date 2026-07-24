import json
import httpx
import config

_http_client = httpx.Client(base_url=config.GPU_JUDGE_MODEL, timeout=120)

JUDGE_SYSTEM_PROMPT = (
    "You are grading an AI assistant's answer against retrieved source excerpts. "
    "Respond with ONLY a JSON object with exactly these fields, in this order so you reason before you score: "
    "reasoning (one short sentence weighing the answer against the excerpts, decided first), "
    "faithfulness_score (integer 1-5, is every claim in the answer supported by the excerpts), "
    "hallucination (true/false, does the answer state anything NOT supported by the excerpts), "
    "relevancy_score (integer 1-5, does the answer address the question), "
    "declined (true/false, does the answer state that the excerpts don't contain enough information "
    "to answer the question, rather than attempting a substantive answer). "
    "The score/boolean fields must agree with what you wrote in reasoning. "
    "No text outside the JSON object."
)


def _build_judge_prompt(query: str, context: list[dict], answer: str) -> str:
    excerpts = "\n\n".join(
        f"{chunk['text']} (paper:{chunk['paper']}, page:{chunk['page']})" for chunk in context
    )
    return (
        f"Excerpts:\n{excerpts}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer to grade:\n{answer}"
    )


def judge_response(query: str, context: list[dict], answer: str, model: str=config.OLLAMA_URL) -> dict:
    prompt = _build_judge_prompt(query, context, answer)
    response = _http_client.post(
        "/api/generate",
        json={
            "model": model,
            "system": JUDGE_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
    )
    raw = response.json()["response"]

    try:
        parsed = json.loads(raw)
        return {
            "reasoning": parsed.get("reasoning"),
            "faithfulness_score": parsed.get("faithfulness_score"),
            "hallucination": parsed.get("hallucination"),
            "relevancy_score": parsed.get("relevancy_score"),
            "declined": parsed.get("declined"),
        }
    except (json.JSONDecodeError, AttributeError):
        return {"error": "judge_parse_failed", "raw": raw}
