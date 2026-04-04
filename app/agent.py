"""
Assessment agent — pre-wired loop that drives the LLM through tool calls.

The mechanics here are provided. The things left for you to decide:

    MAX_ITERATIONS  — how many tool-call rounds should the agent be allowed?
                      Think about cost, latency, and what happens if you set
                      this too high or too low.

    SYSTEM_PROMPT   — instructs the agent on its role, its tools, and what a
                      good recommendation looks like. The quality of this prompt
                      directly affects the quality of the output.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from app.llm import MODEL, get_client
from app.tools import TOOLS, dispatch_tool

# TODO — set a sensible limit. Consider: what are the cost and latency implications
# of setting this too high? What failure mode does it guard against?
# Justify your choice in APPROACH.md — there is no single right answer, but there
# is a right way to reason about it.
MAX_ITERATIONS: int = 4

SYSTEM_PROMPT: str = """You are an insurance underwriting assistant helping reviewers assess incoming records.

Your job is to produce a clear, plain-English recommendation for each record — not a data dump, but a useful summary a non-technical underwriter can act on immediately.

## Tools

You have two tools:

1. **predict_loss** — runs a trained ML model to estimate whether the record is likely to be loss-making.
   - Call this first, always. It provides the quantitative anchor for your recommendation.
   - The result includes: a binary prediction, a confidence score (0–1), top features driving the prediction, and any data quality warnings.
   - Pay attention to confidence. Below 0.55 is effectively uncertain — say so and recommend human review.

2. **retrieve_similar_records** — searches the historical archive for similar past records.
   - Call this after predict_loss to add context from comparable cases.
   - Write the query as a plain-English description of the account (e.g. "cyber insurance for a healthcare company in APAC with prior claims").
   - If the returned records have high cosine distance (above 0.4), they are not meaningfully similar — say so rather than citing them as if they were relevant.
   - Do not call this tool more than once.

Do not call either tool more than once. Do not call tools in parallel.

## Recommendation format

Your final recommendation must:
- State clearly whether the record is likely to be loss-making, and how confident you are.
- Explain the 2–3 most important factors driving that view (from the model's top features), in plain English.
- Reference the most relevant historical cases if retrieval returned useful matches, with a one-sentence description of why they are relevant.
- Flag any data quality issues or out-of-distribution warnings raised by the model.
- Recommend a specific action: approve, decline, or refer for senior review.
- If confidence is low (below 0.55), or if retrieval found no close matches, explicitly say the evidence is weak and recommend human review.

Do not include raw scores, JSON, or field names in your recommendation. Write for an underwriter, not an engineer.
"""


def run_agent(record: dict) -> dict[str, Any]:
    """
    Run the assessment agent on a single record.

    Drives the LLM through tool calls until it produces a final answer or
    MAX_ITERATIONS is reached.

    Args:
        record: The record dict to assess (fields from new_record.json).

    Returns:
        A dict containing at minimum a "recommendation" key with the agent's
        final text. Add any other fields your AssessResponse requires.
    """
    client = get_client()
    messages: list[Any] = [
        {
            "role": "user",
            "content": f"Please assess this record:\n\n{json.dumps(record, indent=2)}",
        }
    ]

    response: anthropic.types.Message | None = None
    tools_used: list[str] = []

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,  # type: ignore[arg-type]
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if isinstance(block, anthropic.types.ToolUseBlock):
                    if block.name not in tools_used:
                        tools_used.append(block.name)
                    try:
                        result = dispatch_tool(block.name, block.input)
                    except NotImplementedError as e:
                        result = {"error": str(e)}
                    except Exception as e:
                        result = {"error": f"Tool failed: {e}"}

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})

    if response is None:
        raise RuntimeError("Agent produced no response.")

    final_text = next(
        (
            block.text
            for block in response.content
            if isinstance(block, anthropic.types.TextBlock)
        ),
        "No recommendation produced.",
    )

    # tools_used is provided as a starting point. Add any other fields to this dict
    # that you think are worth capturing — what the agent did, how confident it was,
    # whether it hit any uncertainty conditions. What you choose to surface here is
    # itself a design decision.
    return {"recommendation": final_text, "tools_used": tools_used}
