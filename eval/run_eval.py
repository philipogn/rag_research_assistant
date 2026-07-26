import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag import generate_response, retrieve_relevant_chunks
from judge import judge_response
from retrieval_metrics import evaluate_retrieval

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
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
        "judge": judge, # reasoning, faithfulness_score, hallucination, relevancy_score, declined
    }

    if entry["expected_sources"]:
        results["retrieval"] = evaluate_retrieval(retrieved, entry["expected_sources"], k_vals)
    else:
        results["declined"] = judge.get("declined")
    
    return results

def _mean(values: list[float]) -> float:
    return sum(values) / len(values)

def aggregate(results: list[dict]) -> dict:
    answerable = [r for r in results if r["answerable"]]
    judge_success = [r for r in results if "error" not in r["judge"]]

    agg = {"total_queries": len(results)}

    if answerable:
        keys = answerable[0]["retrieval"]
        for k in keys:
            agg[f"mean_{k}"] = _mean([r["retrieval"][k] for r in answerable])

    if judge_success:
        agg["mean_faithfulness_score"] = _mean(
            [r["judge"]["faithfulness_score"] for r in judge_success if r["judge"].get("faithfulness_score") is not None]
        ) / 5
        agg["mean_relevancy_score"] = _mean(
            [r["judge"]["relevancy_score"] for r in judge_success if r["judge"].get("relevancy_score") is not None]
        ) / 5
    
    return agg

def main():
    from tqdm import tqdm
    # results = []
    # count = 0
    # data = load_golden_dataset(GOLDEN_DATASET_PATH)
    # for entry in data:
    #     results.append(run_single_eval(entry))
    #     print(f"Completed {count}")
    #     count += 1

    results = [
        run_single_eval(entry)
        for entry in tqdm(load_golden_dataset(GOLDEN_DATASET_PATH), desc="Evaluating")
    ]
    agg = aggregate(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{datetime.now():%Y%m%dT%H%M%S}.json"
    out_path.write_text(json.dumps({"results": results, "aggregate": agg}, indent=2))

    return results


if __name__ == "__main__":
    main()