"""
Evaluation harness for the assessment agent.

Runs the agent against evals/eval_set.json — 20 records with known outcomes —
and reports how well the agent's recommendations align with ground truth.

Usage:
    uv run python evals/eval.py

Part 2b: implement score_recommendation() to define what a correct recommendation
looks like, then run this to see how your agent performs across the eval set.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# ── Load eval set ─────────────────────────────────────────────────────────────

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text())


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_recommendation(recommendation: str, ground_truth: dict) -> dict:
    """
    Score a single agent recommendation against the ground truth.

    TODO — implement this function.

    Args:
        recommendation: The agent's free-text recommendation string.
        ground_truth:   Dict with keys:
                            is_loss_making (bool)   — the actual outcome
                            loss_ratio     (float)  — the actual loss ratio

    Returns:
        A dict with at minimum:
            correct   (bool)   — did the recommendation align with the actual outcome?
            reasoning (str)    — brief explanation of why you scored it this way

    Approach:
        Use an LLM as judge. Prompt it to extract the agent's implied prediction and
        confidence from the recommendation text, then compare to ground truth. This
        is the right tool for scoring free text — a regex heuristic will miss nuance
        and give you false confidence in your eval. Explain your prompting approach
        in APPROACH.md.

    Scoring dimensions to consider (you do not have to use all of these, but address
    at least three and explain why you chose them):

        - Outcome alignment  — does the recommendation point in the right direction?
        - Calibration        — is the expressed certainty proportionate to the actual
                               loss ratio? A confident wrong answer on a borderline case
                               (loss_ratio near 1.0) is worse than an uncertain one.
        - Actionability      — would a non-technical reviewer know what to do next?
        - Evidence quality   — does the response cite relevant features or similar cases,
                               or does it make claims without grounding?
        - Safe deferral      — does the system appropriately hedge or escalate when
                               evidence is weak or conflicting?

    Records in the eval set are annotated with a "case_type" field for borderline
    and clear cases — use these to analyse whether your agent's calibration holds
    up under different conditions, not just its average accuracy.
    """
    raise NotImplementedError(
        "Implement score_recommendation() in evals/eval.py. "
        "See the docstring above for guidance on what to measure."
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def run_eval() -> None:
    """
    Run the agent against every record in the eval set and print a summary.

    TODO — wire this up to your agent once Part 2 / Part 3 is implemented.

    The agent call below is a placeholder. Replace it with a real call to your
    /assess endpoint or directly to your agent function.
    """
    import httpx  # pip install httpx, or use requests

    eval_set = load_eval_set()
    results = []

    for i, item in enumerate(eval_set):
        record = {k: v for k, v in item.items() if k != "ground_truth"}
        ground_truth = item["ground_truth"]

        print(f"[{i + 1}/{len(eval_set)}] Assessing {record['record_id']}...", end=" ")

        start = time.time()

        # TODO: replace with your actual agent call
        # Option A — call the /assess endpoint directly:
        # response = httpx.post("http://localhost:8000/assess", json={"record": record})
        # recommendation = response.json()["recommendation"]

        # Option B — call your agent function directly (faster, no HTTP overhead):
        # from app.main import run_agent
        # recommendation = run_agent(record)

        # Placeholder:
        recommendation = "[agent not yet wired up]"

        elapsed = time.time() - start
        score = score_recommendation(recommendation, ground_truth)
        results.append(
            {
                "record_id": record["record_id"],
                "ground_truth": ground_truth,
                "recommendation": recommendation,
                "score": score,
                "latency_s": round(elapsed, 2),
            }
        )
        status = "✓" if score.get("correct") else "✗"
        print(f"{status} ({elapsed:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    n_correct = sum(1 for r in results if r["score"].get("correct"))
    avg_latency = sum(r["latency_s"] for r in results) / n

    print(f"\n{'─' * 40}")
    print(f"Results: {n_correct}/{n} correct ({n_correct / n:.0%})")
    print(f"Average latency: {avg_latency:.1f}s")

    loss_making = [r for r in results if r["ground_truth"]["is_loss_making"]]
    not_loss_making = [r for r in results if not r["ground_truth"]["is_loss_making"]]
    if loss_making:
        tp = sum(1 for r in loss_making if r["score"].get("correct"))
        print(f"Loss-making records correct: {tp}/{len(loss_making)}")
    if not_loss_making:
        tn = sum(1 for r in not_loss_making if r["score"].get("correct"))
        print(f"Non-loss-making records correct: {tn}/{len(not_loss_making)}")

    # Write full results to file for inspection
    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out}")


if __name__ == "__main__":
    run_eval()
