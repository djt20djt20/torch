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
from app.model import EXTREME_RAW_WARNING_PREFIX
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
- If any warning begins with "Extreme raw value —", say clearly in your recommendation that an extreme or implausible input was detected, name the field(s), and tell the underwriter to confirm the data before acting — the numeric prediction still used the submitted values.
- Recommend a specific action: approve, decline, or refer for senior review.
- If confidence is low (below 0.55), or if retrieval found no close matches, explicitly say the evidence is weak and recommend human review.

Do not include raw scores, JSON, or field names in your recommendation. Write for an underwriter, not an engineer.
"""

# When the ML artifact is missing or predict() fails, we still run the LLM with
# retrieval only — never invent model scores or confidence.
SYSTEM_PROMPT_RETRIEVAL_ONLY: str = """You are an insurance underwriting assistant helping reviewers assess incoming records.

**The quantitative loss-prediction model is unavailable** (artifact missing or failed to run). You do not have ML confidence scores, SHAP features, or model warnings — do not invent them.

## Tool

You have one tool: **retrieve_similar_records** — search the historical archive for similar past records.
- You **must** call this tool exactly once to gather context, unless the tool returns a hard error in the result.
- Write the query as a plain-English description of the account (e.g. "cyber insurance for a healthcare company in APAC with prior claims").
- If cosine distance is above 0.4, say the matches are not close — do not cite them as strong precedents.

## Recommendation format

Your final recommendation must:
- **State up front** that the automated loss model could not be run, so there is no ML probability for this assessment — only qualitative comparison to history and professional judgement.
- Summarise what similar historical cases suggest, or say evidence from retrieval is weak if distances are high or matches are irrelevant.
- Give a sensible qualitative read of the record (risk type, limits, claims history, etc.) without claiming false precision.
- **Recommend refer for senior review** in almost all cases; only if retrieval returns very close, clearly relevant comparables may you suggest a slightly more specific next step — still with explicit human oversight.
- Never assign a numeric "model confidence" or say the model predicted loss-making / not — it did not run.

Do not include raw JSON or internal field names. Write for an underwriter, not an engineer.
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
    # Always attempt the model first — deterministic, no LLM involvement.
    prediction = run_predict_loss(record)
    model_available = "error" not in prediction
    tools_used: list[str] = ["predict_loss"] if model_available else []  # internal name kept for backward compat

    extreme_raw: list[str] = []
    extreme_note = ""
    if model_available:
        extreme_raw = [
            w
            for w in prediction.get("warnings", [])
            if isinstance(w, str) and w.startswith(EXTREME_RAW_WARNING_PREFIX)
        ]
        if extreme_raw:
            extreme_note = (
                "\n\nData quality: one or more raw input fields look extreme or implausible "
                "(see warnings in the model output that start with \"Extreme raw value\"). "
                "Call this out prominently in your recommendation and advise verifying "
                "those fields with the submitter; the model score was computed from the "
                "values exactly as shown in the record above.\n"
            )

    if model_available:
        user_intro = (
            f"Please assess this record:\n\n{json.dumps(record, indent=2)}\n\n"
            f"The loss-prediction model has already been run. Here is its output:\n\n"
            f"{json.dumps(prediction, indent=2)}"
            f"{extreme_note}"
        )
        system = SYSTEM_PROMPT
    else:
        user_intro = (
            f"Please assess this record:\n\n{json.dumps(record, indent=2)}\n\n"
            "The loss-prediction model could not be run. Reason:\n\n"
            f"{prediction.get('error', 'Unknown error')}\n\n"
            "Follow the retrieval-only instructions in your system prompt. "
            "Do not invent model scores or confidence."
        )
        system = SYSTEM_PROMPT_RETRIEVAL_ONLY

    client = get_client()
    messages: list[Any] = [{"role": "user", "content": user_intro}]

    response: anthropic.types.Message | None = None
    truncated = False       # True if max_tokens was hit on the final response
    iterations_used = 0

    for _ in range(MAX_ITERATIONS):
        iterations_used += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system,
            tools=RETRIEVAL_TOOLS,  # type: ignore[arg-type]
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # Anthropic stop_reason values:
        #   "end_turn"      — model finished naturally; normal exit
        #   "tool_use"      — model wants to call a tool; dispatch and continue
        #   "max_tokens"    — hit max_tokens limit mid-response; break to avoid
        #                     re-sending a truncated message as a new turn
        #   "stop_sequence" — hit a stop sequence; treat as done
        # Anything other than "tool_use" should exit the loop.
        if response.stop_reason == "max_tokens":
            truncated = True
            break

        if response.stop_reason != "tool_use":
            break  # covers "end_turn", "stop_sequence", and any future values

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

    # If we exhausted all iterations without an end_turn, the loop hit its cap.
    iteration_limit_reached = (
        iterations_used == MAX_ITERATIONS
        and response is not None
        and response.stop_reason == "tool_use"
    )

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
        "model_available": model_available,
        "truncated": truncated,                          # max_tokens hit on final response
        "iteration_limit_reached": iteration_limit_reached,  # MAX_ITERATIONS exhausted
    }
