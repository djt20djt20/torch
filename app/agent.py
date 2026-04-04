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
from app.tools import RETRIEVAL_TOOLS, dispatch_tool, run_predict_loss

# The model is always called unconditionally before the agent loop — there is no
# decision to make about whether to run it. MAX_ITERATIONS therefore only needs to
# cover: (1) the LLM deciding to call retrieve_similar_records, (2) the LLM
# synthesising a final answer. Two iterations is sufficient; three provides a small
# buffer if the LLM emits a reasoning step before its tool call.
MAX_ITERATIONS: int = 3

SYSTEM_PROMPT: str = """You are an insurance underwriting assistant helping reviewers assess incoming records.

Your job is to produce a clear, plain-English recommendation for each record — not a data dump, but a useful summary a non-technical underwriter can act on immediately.

You will be given the record and the output of a loss-prediction model that has already been run. You do not need to run the model yourself.

## Tool

You have one tool available:

**retrieve_similar_records** — searches the historical archive for similar past records.
- Call this to add context from comparable historical cases.
- Write the query as a plain-English description of the account (e.g. "cyber insurance for a healthcare company in APAC with prior claims").
- If the returned records have high cosine distance (above 0.4), they are not meaningfully similar — say so rather than citing them as if they were relevant.
- You may skip this tool if the model prediction is very high confidence (above 0.85) and no data quality warnings were raised — in that case you already have sufficient evidence for a recommendation.
- Do not call this tool more than once.

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

    The loss-prediction model is called unconditionally before the LLM loop —
    there is no decision to be made about whether to run it. The LLM receives
    the prediction result in its initial prompt and decides whether to call
    retrieve_similar_records for additional context.

    Args:
        record: The record dict to assess (fields from new_record.json).

    Returns:
        A dict containing at minimum a "recommendation" key with the agent's
        final text. Add any other fields your AssessResponse requires.
    """
    # Always run the model first — deterministic, no LLM involvement.
    prediction = run_predict_loss(record)
    tools_used: list[str] = ["predict_loss"]

    client = get_client()
    messages: list[Any] = [
        {
            "role": "user",
            "content": (
                f"Please assess this record:\n\n{json.dumps(record, indent=2)}\n\n"
                f"The loss-prediction model has already been run. Here is its output:\n\n"
                f"{json.dumps(prediction, indent=2)}"
            ),
        }
    ]

    response: anthropic.types.Message | None = None

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=RETRIEVAL_TOOLS,  # type: ignore[arg-type]
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

    return {
        "recommendation": final_text,
        "tools_used": tools_used,
        "prediction": prediction,
    }
