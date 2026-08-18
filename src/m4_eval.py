from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    empty_result = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }
    if not questions:
        return empty_result

    try:
        lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
        if len(lengths) != 1:
            raise ValueError("RAGAS inputs must have the same length")

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        dataframe = result.to_pandas()
        metric_names = (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )

        per_question = [
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=_as_string_list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            )
            for _, row in dataframe.iterrows()
        ]
        aggregates = {
            metric: _safe_float(dataframe[metric].mean()) if metric in dataframe else 0.0
            for metric in metric_names
        }
        return {**aggregates, "per_question": per_question}
    except Exception as exc:
        print(f"  ⚠️  RAGAS evaluation failed: {exc}")
        return empty_result


def _safe_float(value) -> float:
    """Convert metric values to finite floats so reports remain valid JSON."""
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _as_string_list(value) -> list[str]:
    """Normalize a contexts cell without relying on ambiguous array truth values."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating or making unsupported claims",
            "Tighten the grounded-answer prompt and lower temperature",
        ),
        "context_recall": (
            "Relevant information is missing from retrieved chunks",
            "Improve chunking and hybrid retrieval coverage",
        ),
        "context_precision": (
            "Retrieved context contains too many irrelevant chunks",
            "Add reranking or metadata filters and reduce top-k",
        ),
        "answer_relevancy": (
            "The answer does not directly address the question",
            "Improve the answer prompt and explicitly require a direct response",
        ),
    }

    analyzed: list[dict] = []
    for result in eval_results:
        metric_scores = {
            metric: _safe_float(_result_value(result, metric, 0.0))
            for metric in diagnostic_tree
        }
        worst_metric = min(metric_scores, key=metric_scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": str(_result_value(result, "question", "")),
            "worst_metric": worst_metric,
            "score": metric_scores[worst_metric],
            "average_score": sum(metric_scores.values()) / len(metric_scores),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda item: item["average_score"])
    return analyzed[:bottom_n]


def _result_value(result, key: str, default):
    """Read either an EvalResult or a dict produced by a serialized report."""
    return result.get(key, default) if isinstance(result, dict) else getattr(result, key, default)


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
