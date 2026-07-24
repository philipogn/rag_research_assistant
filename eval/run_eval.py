import json
from pathlib import Path

from rag import generate_response, retrieve_relevant_chunks
from judge import judge_response
from retrieval_metrics import evaluate_retrieval

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.jsonl"
K_VALUES = (3, 5, 10)

def load_golden_dataset(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def run_single_eval(entry: dict, k_vals=K_VALUES):
    retrieved = retrieve_relevant_chunks(entry["query"], n_results=k_vals[-1])
    answer = generate_response(entry["query"], retrieved)
    judge = judge_response(entry["query"], retrieved, answer)

    results = {
        "id": entry["id"],
        "query": entry["query"],
        "answerable": entry["answerable"],
        "answer": answer,
        "judge": judge,
        "retrieval": evaluate_retrieval(retrieved, entry["expectecd_sources"], k_vals)
    }
    return results

# include judge scores

def main():
    results = []
    count = 0
    data = load_golden_dataset(GOLDEN_DATASET_PATH)
    for entry in data:
        results.append(run_single_eval(entry))
        print(f"Completed {count}")
        count += 1
    return results


if __name__ == "__main__":
    main()