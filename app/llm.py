"""
LLM client — pre-configured Anthropic client ready to use.

Usage:
    from app.llm import get_client, MODEL

    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"


def get_client() -> anthropic.Anthropic:
    """Return a configured Anthropic client using ANTHROPIC_API_KEY from the environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic(api_key=api_key)
