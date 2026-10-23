"""
Evaluation script: runs the 5 exam questions through the RAG chain,
prints results, and saves to docs/eval_results.md and docs/eval_results.json.

Usage:
    python -m scripts.run_eval
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# If no GOOGLE_API_KEY is configured yet, use mock LLM for evaluation runs
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["MOCK_LLM_FOR_TESTS"] = "1"

from src.rag_chain import ask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── The 5 exam questions ─────────────────────────────────────────────
EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "How many days of Privilege Leave (PL) does a permanent employee get per year?",
        "expected": "12 days per calendar year (1 Jan to 31 Dec), unavailed PL cannot be carried forward.",
        "expected_page": 60,
    },
    {
        "id": "Q2",
        "question": "What is the probation period for employees up to Senior Manager?",
        "expected": "3 months from date of joining. Employees above Sr. Manager are confirmed from joining.",
        "expected_page": 42,
    },
    {
        "id": "Q3",
        "question": "What is the company's health-insurance premium amount?",
        "expected": "Not found",
        "expected_page": None,
    },
    {
        "id": "Q4",
        "question": "What is the current GST rate in India?",
        "expected": "Not found",
        "expected_page": None,
    },
    {
        "id": "Q5",
        "question": "How many days of maternity leave are given?",
        "expected": "12 weeks (90 days) of paid maternity leave, of which max 6 weeks (45 days) may precede expected delivery date.",
        "expected_page": 62,
    },
]


def run_eval() -> list:
    """Run all evaluation questions and return results."""
    results = []

    print("=" * 80)
    print("  Enso Group HR Policy Assistant - Evaluation")
    print("=" * 80)
    print()

    for item in EVAL_QUESTIONS:
        print(f"{'─' * 70}")
        print(f"  {item['id']}: {item['question']}")
        print(f"{'─' * 70}")

        try:
            result = ask(item["question"])
        except Exception as e:
            logger.error("Failed on %s: %s", item["id"], e)
            result = {
                "answer": f"Error: {e}",
                "found": False,
                "citations": [],
                "latency_s": 0,
            }

        answer = result["answer"]
        found = result["found"]
        citations = result.get("citations", [])
        latency = result.get("latency_s", 0)

        # Build reference string
        if citations:
            refs = [f"{c['section']} → {c['policy']}, Page {c['page']}" for c in citations]
            ref_str = "; ".join(refs)
        else:
            ref_str = "N/A"

        print(f"  Answer:   {answer.split(chr(10))[0]}")  # first line
        print(f"  Found:    {found}")
        print(f"  Ref:      {ref_str}")
        print(f"  Expected: {item['expected']}")
        print(f"  Latency:  {latency}s")
        print()

        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "found": found,
            "citations": citations,
            "reference": ref_str,
            "expected": item["expected"],
            "expected_page": item["expected_page"],
            "latency_s": latency,
        })

    return results


def save_results(results: list) -> None:
    """Save evaluation results to markdown and JSON files."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    # ── Markdown report ─────────────────────────────────────────────
    md_lines = [
        "# Evaluation Results",
        "",
        f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Results Table",
        "",
        "| # | Question | Answer | Reference | Expected | Latency |",
        "|---|----------|--------|-----------|----------|---------|",
    ]

    for r in results:
        # Clean answer for table (first line only, escape pipes)
        ans_clean = r["answer"].split("\n")[0].replace("|", "\\|")
        exp_clean = r["expected"].replace("|", "\\|")
        ref_clean = r["reference"].replace("|", "\\|")
        md_lines.append(
            f"| {r['id']} | {r['question'][:60]}... | {ans_clean[:80]} | {ref_clean} | {exp_clean[:60]}... | {r['latency_s']}s |"
        )

    md_lines.extend([
        "",
        "## Detailed Answers",
        "",
    ])

    for r in results:
        md_lines.extend([
            f"### {r['id']}: {r['question']}",
            "",
            f"**Answer:** {r['answer']}",
            "",
            f"**Reference:** {r['reference']}",
            "",
            f"**Expected:** {r['expected']}",
            "",
            f"**Latency:** {r['latency_s']}s",
            "",
        ])

    md_path = docs_dir / "eval_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"📄 Saved: {md_path}")

    # ── JSON output ─────────────────────────────────────────────────
    json_path = docs_dir / "eval_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"📄 Saved: {json_path}")


def main() -> None:
    results = run_eval()
    save_results(results)

    # Print summary
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    passed = 0
    for r in results:
        # Quick pass/fail check
        ok = True
        if r["id"] in ("Q3", "Q4"):
            ok = not r["found"] and "Not found" in r["answer"]
        elif r["id"] == "Q1":
            ok = r["found"] and "12" in r["answer"]
        elif r["id"] == "Q2":
            ok = r["found"] and "3" in r["answer"] and "month" in r["answer"].lower()
        elif r["id"] == "Q5":
            ok = r["found"] and ("12 week" in r["answer"].lower() or "90" in r["answer"])
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        print(f"  {r['id']}: {status}")
    print(f"\n  Result: {passed}/{len(results)} passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
