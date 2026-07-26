import json
import httpx
import config

_http_client = httpx.Client(base_url=config.GPU_JUDGE_URL, timeout=120)

JUDGE_SYSTEM_PROMPT = (
    "You are grading an AI assistant's answer against retrieved source excerpts. "
    "Respond with ONLY a JSON object with exactly these fields, in this order so you reason before you score: "
    "reasoning (one short sentence weighing the answer against the excerpts, decided first), "
    "declined (true/false, does the answer state that the excerpts don't contain enough information "
    "to answer the question, rather than attempting a substantive answer), "
    "faithfulness_score (integer 1-5, is every claim in the answer supported by the excerpts), "
    "hallucination (true/false, does the answer state anything NOT supported by the excerpts), "
    "relevancy_score (integer 1-5, does the answer address the question). "
    "faithfulness_score and relevancy_score must be integers between 1 and 5 inclusive - never 0. "
    "If declined is true, the answer is making no factual claims beyond the absence of information, "
    "so hallucination must be false and faithfulness_score must be 5, unless the answer ALSO asserts "
    "unsupported facts elsewhere alongside the decline. "
    "Example: excerpts about model architecture, question about training cost, answer 'The excerpts do not "
    "mention training cost' -> declined=true, hallucination=false, faithfulness_score=5, relevancy_score "
    "reflects that the question could not be answered from the excerpts. "
    "The score/boolean fields must agree with what you wrote in reasoning. "
    "No text outside the JSON object."
)


def _validate_judge_output(parsed: dict) -> dict:
    warnings = []

    for score_key in ("faithfulness_score", "relevancy_score"):
        score = parsed.get(score_key)
        if isinstance(score, (int, float)) and not (1 <= score <= 5):
            warnings.append(f"{score_key}={score} outside 1-5 range")
            parsed[score_key] = max(1, min(5, score))

    if parsed.get("declined") is True and parsed.get("hallucination") is True:
        warnings.append("declined=true but hallucination=true (contradictory)")

    if warnings:
        parsed["judge_warnings"] = warnings

    return parsed


def _build_judge_prompt(query: str, context: list[dict], answer: str) -> str:
    excerpts = "\n\n".join(
        f"{chunk['text']} (paper:{chunk['paper']}, page:{chunk['page']})" for chunk in context
    )
    return (
        f"Excerpts:\n{excerpts}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer to grade:\n{answer}"
    )


def judge_response(query: str, context: list[dict], answer: str, model: str=config.JUDGE_MODEL) -> dict:
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
    response.raise_for_status()
    raw = response.json()["response"]

    try:
        parsed = _validate_judge_output(json.loads(raw))
        result = {
            "reasoning": parsed.get("reasoning"),
            "faithfulness_score": parsed.get("faithfulness_score"),
            "hallucination": parsed.get("hallucination"),
            "relevancy_score": parsed.get("relevancy_score"),
            "declined": parsed.get("declined"),
        }
        if "judge_warnings" in parsed:
            result["judge_warnings"] = parsed["judge_warnings"]
        return result
    except (json.JSONDecodeError, AttributeError):
        return {"error": "judge_parse_failed", "raw": raw}
