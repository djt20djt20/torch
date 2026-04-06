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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load eval set ─────────────────────────────────────────────────────────────

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text())


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_recommendation(recommendation: str, ground_truth: dict) -> dict:
    """
    Score a single agent recommendation against the ground truth (LLM-as-judge).

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
    from app.llm import MODEL, get_client

    is_loss_making: bool = ground_truth["is_loss_making"]
    loss_ratio: float = ground_truth["loss_ratio"]
    case_type: str = ground_truth.get("case_type", "clear")
    is_borderline = case_type == "borderline" or abs(loss_ratio - 1.0) < 0.05

    judge_prompt = f"""You are evaluating an AI insurance underwriting agent's recommendation.

## Ground truth
- Actual outcome: {"LOSS-MAKING" if is_loss_making else "NOT loss-making"} (loss_ratio={loss_ratio:.4f})
- Case type: {"borderline (loss_ratio close to 1.0)" if is_borderline else "clear-cut"}

## Agent recommendation
{recommendation}

## Your task
Score the recommendation on these three dimensions. Reply with a JSON object only — no prose.

1. **outcome_alignment** (0 or 1):
   - 1 if the recommendation correctly points toward the actual outcome (loss-making or not), OR if it expresses genuine uncertainty on a borderline case (loss_ratio within 0.05 of 1.0).
   - 0 if it confidently points in the wrong direction.

2. **calibration** (0 or 1):
   - 1 if expressed certainty is proportionate to how clear-cut the case is. A confident wrong answer on a borderline case scores 0. Expressed uncertainty on a borderline case scores 1.
   - 0 if the recommendation is confidently wrong, or confidently right on a borderline case with a loss_ratio very close to 1.0 (over-confidence even when correct is a calibration failure).

3. **safe_deferral** (0 or 1):
   - 1 if the recommendation appropriately recommends human review when evidence is weak (low model confidence, tool failure, borderline case).
   - 0 if it issues an approve/decline verdict without adequate evidence, or fails to escalate a genuinely uncertain case.

Return exactly this JSON:
{{
  "outcome_alignment": 0 or 1,
  "outcome_alignment_reason": "one sentence",
  "calibration": 0 or 1,
  "calibration_reason": "one sentence",
  "safe_deferral": 0 or 1,
  "safe_deferral_reason": "one sentence"
}}"""

    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    raw = response.content[0].text.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}

    dims = {
        "outcome_alignment": {"score": parsed.get("outcome_alignment", 0), "reason": parsed.get("outcome_alignment_reason", "")},
        "calibration":        {"score": parsed.get("calibration", 0),        "reason": parsed.get("calibration_reason", "")},
        "safe_deferral":      {"score": parsed.get("safe_deferral", 0),      "reason": parsed.get("safe_deferral_reason", "")},
    }
    total = sum(d["score"] for d in dims.values())
    correct = total >= 2

    return {
        "correct": correct,
        "total_score": total,
        "max_score": 3,
        "is_borderline": is_borderline,
        "dimensions": dims,
        "reasoning": " | ".join(f"{k}: {v['reason']}" for k, v in dims.items()),
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_eval() -> None:
    """
    Run the agent against every record in the eval set and print a summary.

    Calls `run_agent` directly (same code path as `POST /assess`).
    """
    print("Importing agent...", flush=True)
    t0 = time.time()
    from app.agent import run_agent
    print(f"  Agent imported in {time.time() - t0:.1f}s", flush=True)

    print("Pre-warming vectorstore...", flush=True)
    t0 = time.time()
    from app import vectorstore
    vectorstore.init()
    print(f"  Vectorstore ready in {time.time() - t0:.1f}s\n", flush=True)

    eval_set = load_eval_set()
    results = []

    for i, item in enumerate(eval_set):
        record = {k: v for k, v in item.items() if k != "ground_truth"}
        ground_truth = item["ground_truth"]

        print(f"[{i + 1}/{len(eval_set)}] Assessing {record['record_id']}...", flush=True)
        start = time.time()

        print(f"  [+{time.time()-start:.1f}s] Calling LLM agent (ML + retrieval + synthesis)...", flush=True)
        result = run_agent(record)
        recommendation = result["recommendation"]
        components_used = result.get("tools_used", [])
        truncated = result.get("truncated", False)
        iteration_limit_reached = result.get("iteration_limit_reached", False)

        warnings = []
        if truncated:
            warnings.append("WARN: max_tokens hit — response may be cut off")
        if iteration_limit_reached:
            warnings.append("WARN: MAX_ITERATIONS reached — agent loop did not finish naturally")
        for w in warnings:
            print(f"  {w}", flush=True)

        print(f"  [+{time.time()-start:.1f}s] Agent done. Components used: {components_used}", flush=True)

        print(f"  [+{time.time()-start:.1f}s] Scoring with judge LLM...", flush=True)
        score = score_recommendation(recommendation, ground_truth)
        elapsed = time.time() - start

        results.append(
            {
                "record_id": record["record_id"],
                "ground_truth": ground_truth,
                "recommendation": recommendation,
                "score": score,
                "latency_s": round(elapsed, 2),
                "truncated": truncated,
                "iteration_limit_reached": iteration_limit_reached,
            }
        )
        status = "OK" if score.get("correct") else "FAIL"
        print(f"  [+{time.time()-start:.1f}s] {status} (total: {elapsed:.1f}s)\n", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    n_correct = sum(1 for r in results if r["score"].get("correct"))
    avg_latency = sum(r["latency_s"] for r in results) / n

    print(f"\n{'-' * 40}")
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

    n_truncated = sum(1 for r in results if r.get("truncated"))
    n_iter_limit = sum(1 for r in results if r.get("iteration_limit_reached"))
    if n_truncated:
        truncated_ids = [r["record_id"] for r in results if r.get("truncated")]
        print(f"max_tokens hit: {n_truncated}/{n} ({', '.join(truncated_ids)})")
    else:
        print(f"max_tokens hit: 0/{n}")
    if n_iter_limit:
        iter_limit_ids = [r["record_id"] for r in results if r.get("iteration_limit_reached")]
        print(f"MAX_ITERATIONS reached: {n_iter_limit}/{n} ({', '.join(iter_limit_ids)})")
    else:
        print(f"MAX_ITERATIONS reached: 0/{n}")

    # Write full results to file for inspection
    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out}")


if __name__ == "__main__":
    run_eval()
