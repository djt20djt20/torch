"""
Tool definitions for the assessment agent.

Two tools are available to the agent:
    1. predict_loss             — runs the ML model on a record (TODO — implement)
    2. retrieve_similar_records — fetches similar historical record summaries (pre-built)

TOOLS is the list of tool schemas in Anthropic tool_use format.
Pass it directly to client.messages.create(tools=TOOLS, ...).

run_retrieve_similar_records is already implemented — it wraps the pre-built vectorstore.
run_predict_loss is a stub for you to implement as part of Part 2.
"""

from __future__ import annotations

# ── Tool schemas (Anthropic tool_use format) ─────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "predict_loss",
        "description": (
            "Run the trained loss-prediction model on a record to estimate whether "
            "it is likely to be loss-making. Returns the prediction, a confidence "
            "score, and the top features driving the prediction. Use this tool when "
            "you need a quantitative risk signal for the record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "record": {
                    "type": "object",
                    "description": (
                        "The record to assess. Must include: risk_type, territory, "
                        "industry, limit, premium, broker, prior_claims, years_trading."
                    ),
                    "properties": {
                        "record_id":     {"type": "string"},
                        "risk_type":     {"type": "string"},
                        "territory":     {"type": "string"},
                        "industry":      {"type": "string"},
                        "limit":         {"type": "number"},
                        "premium":       {"type": "number"},
                        "broker":        {"type": "string"},
                        "prior_claims":  {"type": "integer"},
                        "years_trading": {"type": "integer"},
                    },
                    "required": ["risk_type", "territory", "limit", "premium"],
                }
            },
            "required": ["record"],
        },
    },
    {
        "name": "retrieve_similar_records",
        "description": (
            "Search the historical record archive and return the most similar past "
            "records along with their document summaries. Use this tool to provide "
            "the reviewer with context from comparable historical cases. A good query "
            "describes the account in plain English (e.g. 'cyber technology company "
            "in the US with prior claims')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A natural-language description of the record to search for. "
                        "Include risk type, industry, territory, and any notable "
                        "characteristics such as prior claims or limit size."
                    ),
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of similar records to return (default 3, max 10).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────

def run_predict_loss(record: dict) -> dict:
    """
    Run the ML model on the given record and return a structured prediction.

    Steps:
        1. Call app.model.load_model() to load the trained artifact.
        2. Call app.model.predict(model, record) and return its output.
        3. Handle the case where the model artifact does not exist yet
           (load_model raises FileNotFoundError).

    Expected return format:
        {
            "is_loss_making_prediction": bool,
            "confidence": float,          # probability of loss-making, 0–1
            "top_features": list[str],    # most influential feature names
        }

    The agent will use `confidence` to decide how much weight to place on the
    model's output — low confidence should influence the recommendation.
    """
    from app import model as _model

    try:
        artifact = _model.load_model()
    except FileNotFoundError as e:
        return {"error": str(e)}

    return _model.predict(artifact, record)


def run_retrieve_similar_records(query: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve similar historical records from the vector store.

    This is pre-implemented — no changes needed here.

    Args:
        query:     Natural-language description of the record to search for.
        n_results: Number of results to return (capped at 10 to control token usage).

    Returns:
        [
            {
                "record_id": str,
                "document":  str,   # plain-English summary of the historical record
                "distance":  float, # cosine distance — lower means more similar
            },
            ...
        ]
    """
    from app.vectorstore import retrieve
    return retrieve(query, n_results=min(n_results, 10))


# ── Tool dispatch ─────────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict):
    """
    Route a tool call from the agent to the correct implementation.

    Args:
        tool_name:  The name field from the agent's tool_use block.
        tool_input: The input dict from the agent's tool_use block.

    Returns:
        The result of the tool function, ready to pass back as a tool_result.

    Raises:
        ValueError: if tool_name is not recognised.
    """
    if tool_name == "predict_loss":
        return run_predict_loss(tool_input["record"])
    if tool_name == "retrieve_similar_records":
        return run_retrieve_similar_records(
            tool_input["query"],
            n_results=tool_input.get("n_results", 3),
        )
    raise ValueError(f"Unknown tool: {tool_name!r}")
