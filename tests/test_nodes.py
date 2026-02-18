"""Unit tests for langgraph_librarian.nodes."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_librarian.config import LibrarianConfig
from langgraph_librarian.nodes import (
    _hydrate_messages,
    _summarize_message,
    index_messages,
    select_and_hydrate,
)
from langgraph_librarian.state import IndexEntry


def _make_mock_llm(response_text: str) -> MagicMock:
    """Create a mock BaseChatModel that returns the given text."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=response_text)
    return mock


class TestSummarizeMessage:
    def test_short_message_no_llm_call(self):
        msg = HumanMessage(content="Hello!")
        config = LibrarianConfig(short_message_threshold=200)
        entry = _summarize_message(msg, 0, config)

        assert entry.id == 0
        assert entry.role == "human"
        assert entry.summary == "Hello!"
        assert entry.indexing_latency_ms == 0.0

    def test_long_message_calls_llm(self):
        long_text = "A" * 300
        mock_llm = _make_mock_llm("Summary of the long message")
        config = LibrarianConfig(
            indexer_model=mock_llm,
            short_message_threshold=200,
        )

        entry = _summarize_message(HumanMessage(content=long_text), 5, config)

        assert entry.id == 5
        assert entry.summary == "Summary of the long message"
        assert mock_llm.invoke.called

    def test_empty_message(self):
        msg = HumanMessage(content="")
        config = LibrarianConfig(short_message_threshold=200)
        entry = _summarize_message(msg, 0, config)

        assert entry.summary == "(empty message)"


class TestHydrateMessages:
    def _make_messages(self):
        return [
            HumanMessage(content="Message 0"),
            AIMessage(content="Response 0"),
            HumanMessage(content="Message 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Message 2"),
            AIMessage(content="Response 2"),
            HumanMessage(content="Message 3"),
            AIMessage(content="Response 3"),
        ]

    def test_includes_recent_turns(self):
        messages = self._make_messages()
        result = _hydrate_messages(messages, selected_ids=[0], always_include_recent_turns=2)

        # Should include: ID 0 (selected) + recent turns
        contents = [m.content for m in result]
        assert "Message 0" in contents
        assert "Response 3" in contents  # most recent
        assert len(result) > 1

    def test_no_duplicates(self):
        messages = self._make_messages()
        result = _hydrate_messages(messages, selected_ids=[6, 7], always_include_recent_turns=2)

        # selected IDs overlap with recent — no duplicates
        content_list = [m.content for m in result]
        assert len(content_list) == len(set(content_list))

    def test_maintains_order(self):
        messages = self._make_messages()
        result = _hydrate_messages(messages, selected_ids=[4, 0, 2], always_include_recent_turns=1)

        # Verify chronological order
        indices = [messages.index(m) for m in result]
        assert indices == sorted(indices)

    def test_empty_selection(self):
        messages = self._make_messages()
        result = _hydrate_messages(messages, selected_ids=[], always_include_recent_turns=1)

        # Should only have recent turns
        assert len(result) > 0
        assert result[-1].content == "Response 3"


class TestSelectAndHydrate:
    def test_skips_when_no_index(self):
        state = {
            "messages": [HumanMessage(content="Hi")],
            "query": "test",
            "index": [],
        }
        result = select_and_hydrate(state)
        assert result["librarian_metadata"]["selection_skipped"] is True

    def test_skips_when_no_query(self):
        state = {
            "messages": [HumanMessage(content="Hi")],
            "query": "",
            "index": [IndexEntry(id=0, role="human", summary="Hi")],
        }
        result = select_and_hydrate(state)
        assert result["librarian_metadata"]["selection_skipped"] is True

    def test_selects_with_index(self):
        mock_llm = _make_mock_llm(
            "Reasoning: Message 0 has the answer.\nIDs: [0]"
        )
        config = LibrarianConfig(
            indexer_model=mock_llm,
            selector_model=mock_llm,
        )

        messages = [
            HumanMessage(content="What is the budget?"),
            AIMessage(content="The budget is $10,000"),
            HumanMessage(content="Tell me a joke"),
            AIMessage(content="Why did the chicken..."),
        ]
        index = [
            IndexEntry(id=0, role="human", summary="Asks about budget"),
            IndexEntry(id=1, role="ai", summary="Budget is $10,000"),
            IndexEntry(id=2, role="human", summary="Asks for a joke"),
            IndexEntry(id=3, role="ai", summary="Tells a joke"),
        ]

        state = {
            "messages": messages,
            "query": "What is the budget?",
            "index": index,
        }
        result = select_and_hydrate(
            state,
            {"configurable": {"librarian": config}},
        )

        assert result["librarian_metadata"]["selection_skipped"] is False
        assert len(result["curated_messages"]) > 0


class TestIndexMessages:
    def test_indexes_new_messages(self):
        mock_llm = _make_mock_llm("Summary")
        config = LibrarianConfig(indexer_model=mock_llm)

        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ],
            "index": [],
        }

        result = index_messages(
            state,
            {"configurable": {"librarian": config}},
        )

        assert len(result["index"]) == 2
        assert result["index"][0].id == 0
        assert result["index"][1].id == 1

    def test_incremental_indexing(self):
        mock_llm = _make_mock_llm("New summary")
        config = LibrarianConfig(indexer_model=mock_llm)

        existing = [
            IndexEntry(id=0, role="human", summary="Hello"),
            IndexEntry(id=1, role="ai", summary="Hi there"),
        ]

        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
                HumanMessage(content="New message"),  # Only this is new
            ],
            "index": existing,
        }

        result = index_messages(
            state,
            {"configurable": {"librarian": config}},
        )

        # Should have 3 entries total (2 existing + 1 new)
        assert len(result["index"]) == 3
        assert result["librarian_metadata"]["indexing_new_count"] == 1

    def test_already_up_to_date(self):
        config = LibrarianConfig()

        existing = [
            IndexEntry(id=0, role="human", summary="Hello"),
        ]

        state = {
            "messages": [HumanMessage(content="Hello")],
            "index": existing,
        }

        result = index_messages(
            state,
            {"configurable": {"librarian": config}},
        )

        # No new indexing needed
        assert result["index"] == existing
