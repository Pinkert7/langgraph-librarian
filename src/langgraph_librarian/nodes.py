"""LangGraph nodes for the Librarian pipeline.

Three node functions that form the core Librarian pipeline:
- select_and_hydrate: reads index, selects relevant messages, hydrates context
- index_messages: summarizes new messages incrementally (runs post-response)

These are designed to be used with LangGraph's StateGraph.
"""

import time
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .config import LibrarianConfig
from .prompts import DEFAULT_SELECTION_PROMPT, DEFAULT_SUMMARY_PROMPT
from .state import IndexEntry, LibrarianState
from .utils import SelectionResult, estimate_tokens, extract_message_text, parse_selection_response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _summarize_message(
    message: BaseMessage,
    msg_id: int,
    config: LibrarianConfig,
) -> IndexEntry:
    """Generate a summary for a single message using the indexer LLM.

    For short messages (below threshold), uses the text directly.
    For longer messages, calls the indexer model.

    Args:
        message: The message to summarize.
        msg_id: Zero-based index in the conversation.
        config: Librarian configuration.

    Returns:
        An IndexEntry with the summary.
    """
    text = extract_message_text(message)
    role = message.type  # 'human', 'ai', 'tool', etc.
    original_tokens = estimate_tokens(text)

    # Short messages: use text directly (no LLM call needed)
    if len(text) < config.short_message_threshold:
        return IndexEntry(
            id=msg_id,
            role=role,
            summary=text or "(empty message)",
            original_tokens=original_tokens,
            indexing_latency_ms=0.0,
        )

    # Long messages: call the indexer LLM
    start_ms = time.time() * 1000
    prompt = config.summary_prompt or DEFAULT_SUMMARY_PROMPT
    summary_prompt = f"{prompt}\n\nMessage:\n{text}"

    try:
        llm = config.get_indexer_model()
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = extract_message_text(response)
    except Exception as e:
        # Fallback: use truncated text if LLM fails
        summary = text[:300]

    elapsed_ms = (time.time() * 1000) - start_ms

    return IndexEntry(
        id=msg_id,
        role=role,
        summary=summary,
        original_tokens=original_tokens,
        indexing_latency_ms=elapsed_ms,
    )


def _select_relevant_ids(
    index: list[IndexEntry],
    query: str,
    config: LibrarianConfig,
) -> SelectionResult:
    """Use LLM reasoning to select relevant message IDs from the summary index.

    This is the core "select" step of select-then-hydrate.

    Args:
        index: The summary index to reason over.
        query: The user's query.
        config: Librarian configuration.

    Returns:
        SelectionResult with selected IDs and reasoning.
    """
    # Build the index string
    index_str = "\n".join(
        f"ID {entry.id} [{entry.role}]: {entry.summary}" for entry in index
    )

    # Build the prompt
    prompt_template = config.selection_prompt or DEFAULT_SELECTION_PROMPT
    prompt = prompt_template.replace("{query}", query).replace("{index}", index_str)

    # Call the selector LLM
    llm = config.get_selector_model()
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = extract_message_text(response)

    # Parse the response
    result = parse_selection_response(response_text)

    # Filter to valid IDs
    valid_ids = [id for id in result.ids if 0 <= id < len(index)]
    return SelectionResult(ids=valid_ids, reasoning=result.reasoning)


def _hydrate_messages(
    messages: list[BaseMessage],
    selected_ids: list[int],
    always_include_recent_turns: int = 2,
) -> list[BaseMessage]:
    """Fetch full message content for selected IDs + recent turns.

    Always includes the most recent N user turns to maintain
    conversational coherence, even if not selected by the Librarian.

    Args:
        messages: Full conversation history.
        selected_ids: IDs selected by the Librarian.
        always_include_recent_turns: Number of recent user turns to always include.

    Returns:
        Curated list of messages in original order.
    """
    # Find indices of the last N user turns (and their associated messages)
    recent_indices: set[int] = set()
    user_count = 0
    for i in range(len(messages) - 1, -1, -1):
        if user_count >= always_include_recent_turns and messages[i].type == "human":
            break
        recent_indices.add(i)
        if messages[i].type == "human":
            user_count += 1

    # Merge selected IDs with recent indices, maintaining original order
    all_ids = set(selected_ids) | recent_indices
    sorted_ids = sorted(id for id in all_ids if 0 <= id < len(messages))

    return [messages[id] for id in sorted_ids]


# ---------------------------------------------------------------------------
# LangGraph Node Functions
# ---------------------------------------------------------------------------


def select_and_hydrate(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """LangGraph node: Select relevant messages and hydrate context.

    This is the ONLINE step — runs before the agent, reading the
    pre-built index to select and hydrate relevant context.

    On first turn (no index), passes through all messages.

    Args:
        state: Current graph state (LibrarianState-compatible dict).
        config: LangGraph RunnableConfig with 'librarian' in configurable.

    Returns:
        State update with curated_messages and librarian_metadata.
    """
    messages: list[BaseMessage] = state.get("messages", [])
    query: str = state.get("query", "")
    index: list[IndexEntry] = state.get("index", [])

    # Extract LibrarianConfig from the LangGraph configurable
    librarian_config: LibrarianConfig = _get_config(config)

    start_time = time.time()

    # First turn or no index: pass through all/recent messages
    if not index or not query:
        return {
            "curated_messages": messages,
            "librarian_metadata": {
                "selection_skipped": True,
                "reason": "no_index" if not index else "no_query",
                "curated_count": len(messages),
                "original_count": len(messages),
            },
        }

    # Select: reason over the index to pick relevant IDs
    selection = _select_relevant_ids(index, query, librarian_config)

    # Hydrate: fetch full content for selected + recent messages
    curated = _hydrate_messages(
        messages,
        selection.ids,
        librarian_config.always_include_recent_turns,
    )

    elapsed_ms = (time.time() - start_time) * 1000

    # Calculate metrics
    original_tokens = sum(estimate_tokens(extract_message_text(m)) for m in messages)
    curated_tokens = sum(estimate_tokens(extract_message_text(m)) for m in curated)

    return {
        "curated_messages": curated,
        "selected_ids": selection.ids,
        "librarian_metadata": {
            "selection_skipped": False,
            "selected_ids": selection.ids,
            "selection_reasoning": selection.reasoning,
            "selection_latency_ms": elapsed_ms,
            "original_count": len(messages),
            "curated_count": len(curated),
            "original_tokens": original_tokens,
            "curated_tokens": curated_tokens,
            "token_reduction_pct": (
                round((original_tokens - curated_tokens) / original_tokens * 100, 1)
                if original_tokens > 0
                else 0
            ),
        },
    }


def index_messages(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """LangGraph node: Index new messages incrementally.

    This is the OFFLINE step — runs AFTER the agent response,
    so indexing latency is invisible to the user (with streaming).

    Only processes messages that haven't been indexed yet (incremental).

    Args:
        state: Current graph state (LibrarianState-compatible dict).
        config: LangGraph RunnableConfig with 'librarian' in configurable.

    Returns:
        State update with the updated index.
    """
    messages: list[BaseMessage] = state.get("messages", [])
    existing_index: list[IndexEntry] = state.get("index", [])

    librarian_config: LibrarianConfig = _get_config(config)

    # Determine which messages are new
    start_from = len(existing_index)
    if start_from >= len(messages):
        # Index is already up-to-date
        return {"index": existing_index}

    # Index only new messages
    new_entries: list[IndexEntry] = []
    for i in range(start_from, len(messages)):
        entry = _summarize_message(messages[i], i, librarian_config)
        new_entries.append(entry)

    # Merge with existing index
    updated_index = list(existing_index) + new_entries

    return {
        "index": updated_index,
        "librarian_metadata": {
            "indexing_new_count": len(new_entries),
            "indexing_total_count": len(updated_index),
            "indexing_latency_ms": sum(e.indexing_latency_ms for e in new_entries),
        },
    }


def _get_config(config: Optional[RunnableConfig]) -> LibrarianConfig:
    """Extract LibrarianConfig from LangGraph's RunnableConfig.

    The config is expected at config["configurable"]["librarian"].
    Falls back to a default config if not found.
    """
    if config is None:
        return LibrarianConfig()

    configurable = config.get("configurable", {})

    # Support direct LibrarianConfig or nested in configurable
    if isinstance(configurable, LibrarianConfig):
        return configurable

    librarian = configurable.get("librarian")
    if isinstance(librarian, LibrarianConfig):
        return librarian

    return LibrarianConfig()
