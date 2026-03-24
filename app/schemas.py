"""
Request and response schemas for the assessment API.

TODO — AssessResponse is your design decision (Part 3).

What does a non-technical reviewer actually need to see?
Replace or extend the fields below to match the recommendation your agent produces,
then justify your choices in APPROACH.md.

Things to consider:
- How do you surface model confidence in a way a reviewer can act on?
- When should the response tell the reviewer to seek a second opinion?
- What from the retrieved similar records is genuinely useful vs noise?
- What would an auditor need to see in this response six months later?
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AssessRequest(BaseModel):
    record: dict[str, Any]


class AssessResponse(BaseModel):
    record_id: str
    recommendation: str  # Keep this — the core output for the reviewer
    tools_used: list[str]  # Which tools the agent called — e.g. ["predict_loss", "retrieve_similar_records"]
