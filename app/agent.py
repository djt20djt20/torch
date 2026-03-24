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
MAX_ITERATIONS: int = ...  # type: ignore[assignment]

# TODO — write the system prompt. The agent needs to know:
#   - what it is and who it's helping
#   - what each tool does and when to use it (vs when not to)
#   - what a useful recommendation looks like to a non-technical reviewer
#   - how to handle uncertainty (low confidence, poor retrieval, ambiguous record)
SYSTEM_PROMPT: str = ...  # type: ignore[assignment]


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

    return {"recommendation": final_text, "tools_used": tools_used}
