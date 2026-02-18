"""State schema for the LangGraph Librarian.

Defines the core state types used throughout the Librarian pipeline:
- IndexEntry: a single message summary in the index
- LibrarianState: the full graph state (TypedDict for LangGraph)
"""

from dataclasses import dataclass
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class IndexEntry:
    """A pre-computed summary of a single message in the conversation history.

    The summary is a ≤3-sentence distillation of the message content,
    used by the Librarian to reason about relevance without reading full content.
    """

    id: int
    """Zero-based position in the original message array."""

    role: str
    """Message role: 'human', 'ai', 'tool', etc."""

    summary: str
    """≤3-sentence summary of the message content."""

    original_tokens: int = 0
    """Estimated token count of the original full message."""

    indexing_latency_ms: float = 0.0
    """Time taken to generate this summary in milliseconds."""


def _replace_index(
    left: list[IndexEntry], right: list[IndexEntry]
) -> list[IndexEntry]:
    """Reducer for the index: always replace with the latest version."""
    return right if right else left


def _replace_metadata(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Reducer for metadata: merge dicts, with right taking precedence."""
    merged = {**left, **right} if left else right
    return merged


class LibrarianState(TypedDict):
    """State for the Librarian graph.

    Uses Annotated types with reducers for LangGraph channel management.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    """Full conversation history, managed by LangGraph's add_messages reducer."""

    query: str
    """The current user query to find relevant context for."""

    index: Annotated[list[IndexEntry], _replace_index]
    """Summary index of all messages. Persisted across turns via checkpointer."""

    selected_ids: list[int]
    """IDs of messages selected by the selector LLM."""

    curated_messages: Sequence[BaseMessage]
    """Curated context: selected + recent messages. Use this in your agent."""

    librarian_metadata: Annotated[dict[str, Any], _replace_metadata]
    """Metrics and debug info (latency, token reduction, reasoning, etc.)."""
