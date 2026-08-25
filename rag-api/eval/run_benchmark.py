import json
import os
import sys
import time
from collections import defaultdict
import numpy as np

# Force UTF-8 encoding on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure app is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.query_service import answer_question


def evaluate_response(response: dict, expected_keywords: list[str]) -> bool:
    """Check if answer contains relevant information or correctly refuses out-of-scope."""
    ans = response.get("answer", "").lower()
    if not ans:
        return False
    # If out-of-scope query, check if it gracefully refuses
    if any("could not find" in kw.lower() or "not found" in kw.lower() for kw in expected_keywords):
        return "could not find" in ans or "no direct answer" in ans or "not mentioned" in ans or "no relevant" in ans
    
    # Check keyword hit rate
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans)
    return hits >= max(1, len(expected_keywords) // 2)


def run_benchmark(questions_file: str = "eval/benchmark_questions.json"):
    filepath = os.path.join(os.path.dirname(__file__), "benchmark_questions.json")
    if not os.path.exists(filepath):
        filepath = questions_file

    with open(filepath, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n=======================================================")
    print(f"[BENCHMARK] Running Hybrid GraphRAG Evaluation ({len(questions)} queries)")
    print(f"=======================================================\n")

    results_by_cat = defaultdict(lambda: {"vector_correct": 0, "hybrid_correct": 0, "total": 0})
    vector_latencies = []
    hybrid_latencies = []

    for q in questions:
        qid = q["id"]
        cat = q["category"]
        question_text = q["question"]
        expected_keywords = q["expected_answer_keywords"]

        # 1. Run Vector-Only
        t0 = time.time()
        vec_res = answer_question(question=question_text, force_route="vector")
        vec_lat = round((time.time() - t0) * 1000, 2)
        vector_latencies.append(vec_lat)
        vec_passed = evaluate_response(vec_res, expected_keywords)

        # 2. Run Hybrid (Automatic Router + Graph + Vector)
        t1 = time.time()
        hyb_res = answer_question(question=question_text)
        hyb_lat = round((time.time() - t1) * 1000, 2)
        hybrid_latencies.append(hyb_lat)
        hyb_passed = evaluate_response(hyb_res, expected_keywords)

        results_by_cat[cat]["total"] += 1
        if vec_passed:
            results_by_cat[cat]["vector_correct"] += 1
        if hyb_passed:
            results_by_cat[cat]["hybrid_correct"] += 1

        v_mark = "[PASS]" if vec_passed else "[FAIL]"
        h_mark = "[PASS]" if hyb_passed else "[FAIL]"
        print(f"[{cat.upper():<12}] Q{qid:<2}: {question_text[:45]:<45} | Vec: {v_mark} ({vec_lat}ms) | Hyb ({hyb_res.get('route')}): {h_mark} ({hyb_lat}ms)")

    # Print summary table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY: Vector-Only Baseline vs. Hybrid GraphRAG")
    print("=" * 70)
    print(f"{'Category / Metric':<20} | {'Vector-Only':<12} | {'Hybrid GraphRAG':<16} | {'Delta':<10}")
    print("-" * 70)

    category_display_order = ["single-hop", "two-hop", "three-hop", "aggregation", "out-of-scope"]
    for cat in category_display_order:
        if cat in results_by_cat:
            data = results_by_cat[cat]
            total = data["total"]
            vec_acc = (data["vector_correct"] / total) * 100 if total else 0
            hyb_acc = (data["hybrid_correct"] / total) * 100 if total else 0
            delta = hyb_acc - vec_acc
            print(f"{cat.capitalize():<20} | {vec_acc:>10.1f}% | {hyb_acc:>14.1f}% | {f'+{delta:.1f}%' if delta >= 0 else f'{delta:.1f}%':>9}")

    p50_vec = np.percentile(vector_latencies, 50) if vector_latencies else 0
    p95_vec = np.percentile(vector_latencies, 95) if vector_latencies else 0
    p50_hyb = np.percentile(hybrid_latencies, 50) if hybrid_latencies else 0
    p95_hyb = np.percentile(hybrid_latencies, 95) if hybrid_latencies else 0

    print("-" * 70)
    print(f"{'Latency (p50)':<20} | {p50_vec:>10.1f}ms | {p50_hyb:>14.1f}ms | {f'+{p50_hyb - p50_vec:.1f}ms':>9}")
    print(f"{'Latency (p95)':<20} | {p95_vec:>10.1f}ms | {p95_hyb:>14.1f}ms | {f'+{p95_hyb - p95_vec:.1f}ms':>9}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
