"""Configuration for the LangGraph Librarian.

Provides LibrarianConfig to control model selection, prompts,
and behavior of the Librarian pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from .store import BaseIndexStore


@dataclass
class LibrarianConfig:
    """Configuration for the Librarian pipeline.

    Args:
        indexer_model: LLM for summarization. Should be fast and cheap
            (e.g., Gemini Flash, GPT-4o-mini). Required.
        selector_model: LLM for reasoning-based selection. Should be
            reasoning-capable (e.g., Gemini Flash, GPT-4o). If not set,
            falls back to indexer_model.
        always_include_recent_turns: Number of recent user turns to always
            include regardless of selection. Ensures conversational coherence.
            Default: 2.
        summary_prompt: Custom prompt for the summarization step.
            If not set, uses the default prompt.
        selection_prompt: Custom prompt template for the selection step.
            Must contain {query} and {index} placeholders.
            If not set, uses the default prompt.
        short_message_threshold: Messages with fewer characters than this
            are used as-is without summarization. Default: 200.
        store: Optional external index store. If not set, the index is
            persisted via the LangGraph checkpointer (recommended).
    """

    indexer_model: BaseChatModel = field(default=None)  # type: ignore[assignment]
    selector_model: Optional[BaseChatModel] = None
    always_include_recent_turns: int = 2
    summary_prompt: Optional[str] = None
    selection_prompt: Optional[str] = None
    short_message_threshold: int = 200
    store: Optional[BaseIndexStore] = None

    def get_selector_model(self) -> BaseChatModel:
        """Return the selector model, falling back to the indexer model."""
        return self.selector_model or self.indexer_model

    def get_indexer_model(self) -> BaseChatModel:
        """Return the indexer model."""
        if self.indexer_model is None:
            raise ValueError(
                "LibrarianConfig.indexer_model is required. "
                "Pass a BaseChatModel instance (e.g., ChatGoogleGenerativeAI, ChatOpenAI)."
            )
        return self.indexer_model
