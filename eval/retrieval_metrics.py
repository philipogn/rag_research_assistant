def precision_at_k(retrieved, expected, k) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for chunk in top_k if chunk in expected)
    return relevant_count / len(top_k)

# add recall, mrr