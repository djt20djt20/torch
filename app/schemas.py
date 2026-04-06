"""
Request and response schemas for the assessment API.

AssessResponse is your most important design decision in Part 3. This is not
boilerplate — it is the contract between your system and the people who use it.

What does a non-technical reviewer actually need to see? Replace or extend the
fields below and justify your choices in APPROACH.md. We will read both together.

Things to consider:
- How do you surface model confidence in a way a reviewer can act on?
- When should the response tell the reviewer to seek a second opinion?
- What from the retrieved similar records is genuinely useful vs noise?
- What would an auditor need to see in this response six months later?
- What is the minimum a reviewer needs to make a decision — and what clutters it?

There is no single correct schema. We are looking for a clear point of view,
not a complete one.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AssessRequest(BaseModel):
    record: dict[str, Any]


class AssessResponse(BaseModel):
    record_id: str
    recommendation: str        # Full narrative for the reviewer — primary human-readable output
    confidence: float | None = None   # Calibrated model probability (0–1); None if model unavailable
    requires_review: bool = True      # True when confidence is low, OOD warnings present, or model unavailable
    model_warnings: list[str] = []    # Structured OOD / extreme-value flags from the model artifact
    components_used: list[str] = []   # Audit trail of what ran — e.g. ["predict_loss", "retrieve_similar_records"]
    model_available: bool = True      # False when ML artifact missing / predict failed — retrieval-only fallback
    truncated: bool = False           # True if the LLM hit max_tokens and the response may be cut off
    iteration_limit_reached: bool = False  # True if MAX_ITERATIONS was exhausted before end_turn
