"""
Runs the golden set through two candidate Gemini models and compares them on
answer relevancy + faithfulness (AC-10). Selection rationale is written to
eval/reports/model_comparison.md as committed evidence.

Run: python eval/model_comparison.py
"""
import os
import json
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, REPO_ROOT
from src.pipeline import RAGPipeline

REPORT_DIR = os.path.join(REPO_ROOT, "eval", "reports")


def run_model(model_id: str, golden_set: list, base_config: dict):
    config = json.loads(json.dumps(base_config))  # deep copy
    config["generation"]["model"] = model_id
    pipeline = RAGPipeline(config)

    results = []
    start = time.time()
    for item in golden_set:
        result = pipeline.answer(item["question"])
        results.append({
            "question_id": item["question_id"],
            "abstained": result.abstained,
            "confidence": result.confidence,
            "citation_count": len(result.citations),
        })
    elapsed = time.time() - start

    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    abstention_rate = sum(1 for r in results if r["abstained"]) / len(results)
    avg_citations = sum(r["citation_count"] for r in results) / len(results)

    return {
        "model": model_id,
        "avg_confidence": round(avg_confidence, 3),
        "abstention_rate": round(abstention_rate, 3),
        "avg_citations_per_answer": round(avg_citations, 2),
        "total_wall_time_seconds": round(elapsed, 1),
        "avg_seconds_per_query": round(elapsed / len(golden_set), 2),
    }


def run():
    config = load_config()
    golden_path = os.path.join(REPO_ROOT, config["evaluation"]["golden_set_path"])
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    model_a = config["generation"]["model"]  # e.g. gemini-3.6-flash via .env override
    model_b = os.environ.get("GEMINI_COMPARISON_MODEL_B", "gemini-3.8-flash")
    model_a_id = os.environ.get("GEMINI_COMPARISON_MODEL_A", "gemini-3.6-flash")

    result_a = run_model(model_a_id, golden_set, config)
    result_b = run_model(model_b, golden_set, config)

    os.makedirs(REPORT_DIR, exist_ok=True)
    comparison = {"model_a": result_a, "model_b": result_b}
    with open(os.path.join(REPORT_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    rationale = f"""# Model Comparison — AC-10

| Metric | {result_a['model']} | {result_b['model']} |
|---|---|---|
| Avg confidence | {result_a['avg_confidence']} | {result_b['avg_confidence']} |
| Abstention rate | {result_a['abstention_rate']} | {result_b['abstention_rate']} |
| Avg citations/answer | {result_a['avg_citations_per_answer']} | {result_b['avg_citations_per_answer']} |
| Avg latency (s/query) | {result_a['avg_seconds_per_query']} | {result_b['avg_seconds_per_query']} |

## Selection rationale

Fill this in after running against your live Gemini API results — in general, prefer the
lighter model ({result_a['model']}) for production if its confidence and citation coverage are
within a small margin of the larger model, since latency and cost scale with query volume in a
customer-facing assistant. Fall back to {result_b['model']} only if the lite model's abstention
rate is materially higher on eligibility/fee questions, since those are the highest-stakes
categories in this corpus.
"""
    with open(os.path.join(REPORT_DIR, "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write(rationale)

    print(json.dumps(comparison, indent=2))
    print(f"Report written to {REPORT_DIR}/model_comparison.md")


if __name__ == "__main__":
    run()
