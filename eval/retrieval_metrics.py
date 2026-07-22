# retrieve_relevant_chunks returns keys of {"text", "paper", "page"}
# golden dataset contains {..., "expected_sources": [{"paper", "page"}]}

# def _is_relevant(chunk: dict, expected: list[dict]) -> bool:
#     return any(chunk["paper"] == e["paper"] and chunk["page"] == e["page"] for e in expected)


def precision_at_k(retrieved: list[dict], expected: list[dict], k: int) -> float:
    top_k_retrieved = retrieved[:k]
    if not top_k_retrieved:
        return 0.0
    expected_set = {(e["paper"], e["page"]) for e in expected}
    relevant_count = sum(1 for chunk in top_k_retrieved if (chunk["paper"], chunk["page"]) in expected_set)
    return relevant_count / len(top_k_retrieved)


def recall_at_k(retrieved: list[dict], expected: list[dict], k: int) -> float:
    if not expected:
        return 0.0
    top_k_retrieved = retrieved[:k]
    retrieved_set = {(c["paper"], c["page"]) for c in top_k_retrieved}
    expected_set = {(e["paper"], e["page"]) for e in expected}

    return len(retrieved_set & expected_set) / len(expected_set)

# add mrr