"""Utility functions for the LangGraph Librarian.

Provides token estimation and LLM response parsing helpers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using the char/4 heuristic.

    This is a rough approximation. For precise counting, use
    the tokenizer for your specific model.

    Args:
        text: Input text to estimate tokens for.

    Returns:
        Approximate token count.
    """
    return max(1, len(text) // 4)


@dataclass
class SelectionResult:
    """Parsed result from the selection LLM response."""

    ids: list[int]
    reasoning: str


def parse_selection_response(response: str) -> SelectionResult:
    """Parse the LLM's selection response to extract message IDs and reasoning.

    Handles multiple response formats:
    1. Structured: "Reasoning: ... IDs: [1, 2, 3]"
    2. Just array: "[1, 2, 3]"
    3. Malformed: returns empty IDs

    Args:
        response: Raw LLM output from the selection step.

    Returns:
        SelectionResult with parsed IDs and reasoning.
    """
    reasoning = ""
    ids: list[int] = []

    try:
        # Try to extract "Reasoning: ..." section
        reasoning_match = re.search(r"Reasoning:\s*([\s\S]*?)(?=IDs:|$)", response, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Try "IDs: [...]" pattern first
        ids_match = re.search(r"IDs:\s*(\[[\s\S]*?\])", response, re.IGNORECASE)
        if ids_match:
            parsed = json.loads(ids_match.group(1))
            ids = _parse_id_list(parsed)
        else:
            # Fallback: find any JSON array of numbers
            array_match = re.search(r"\[[\d,\s]+\]", response)
            if array_match:
                parsed = json.loads(array_match.group(0))
                ids = _parse_id_list(parsed)
    except (json.JSONDecodeError, ValueError, TypeError):
        # If parsing fails entirely, return empty
        ids = []

    return SelectionResult(ids=ids, reasoning=reasoning)


def _parse_id_list(raw: list) -> list[int]:
    """Safely parse a list of values into integer IDs."""
    result = []
    for x in raw:
        try:
            val = int(x)
            if val >= 0:
                result.append(val)
        except (ValueError, TypeError):
            continue
    return result


def extract_message_text(message) -> str:
    """Extract text content from a LangChain message.

    Handles both string content and list-of-blocks content format.

    Args:
        message: A LangChain BaseMessage or similar object.

    Returns:
        Extracted text content as a string.
    """
    content = message.content if hasattr(message, "content") else str(message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)

    return str(content)
