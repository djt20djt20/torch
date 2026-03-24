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

    Hints:
        - Think carefully about what "correct" means here. A binary correct/incorrect
          may not capture everything useful — a confident wrong prediction is worse
          than an uncertain one.
        - Consider whether the agent flagged appropriate uncertainty when the loss
          ratio was close to 1.0 (borderline cases).
        - You might use the LLM itself as a judge — prompt it to extract the
          agent's implied prediction from the recommendation text and compare to
          ground truth. Or use a simpler heuristic. Both are valid; explain your choice.
        - What other dimensions matter beyond binary accuracy? Think about:
            - Does the recommendation mention relevant risk factors?
            - Is the uncertainty expressed proportionate to the actual loss ratio?
            - Would a non-technical reviewer find it useful?
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
