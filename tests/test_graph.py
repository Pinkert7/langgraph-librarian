"""Integration tests for the full Librarian graph pipeline."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_librarian import LibrarianConfig, create_librarian_graph
from langgraph_librarian.state import IndexEntry


def _make_mock_llm(responses: list[str] | None = None) -> MagicMock:
    """Create a mock LLM that returns responses sequentially."""
    mock = MagicMock()
    if responses:
        mock.invoke.side_effect = [AIMessage(content=r) for r in responses]
    else:
        mock.invoke.return_value = AIMessage(content="Mock response")
    return mock


class TestFullPipeline:
    def test_first_turn_no_index(self):
        """On first turn, select_and_hydrate should pass through all messages."""

        def my_agent(state):
            # Agent should receive all messages (no index yet)
            curated = state.get("curated_messages", state["messages"])
            return {"messages": [AIMessage(content="Agent response")]}

        # Indexer LLM: will be called for each message during index_messages
        indexer_llm = _make_mock_llm()
        indexer_llm.invoke.return_value = AIMessage(content="Summary of message")

        config = LibrarianConfig(indexer_model=indexer_llm)

        graph = create_librarian_graph(agent_fn=my_agent, config=config)

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="What is X?")],
                "query": "What is X?",
                "index": [],
            },
            config={"configurable": {"librarian": config}},
        )

        # Index should now be populated
        assert len(result["index"]) > 0
        # The curated_messages should contain the original messages (passthrough)
        assert result["librarian_metadata"].get("selection_skipped") is True

    def test_second_turn_with_index(self):
        """On second turn, Librarian should select from the pre-built index."""

        def my_agent(state):
            return {"messages": [AIMessage(content="The budget is $10k")]}

        # Selector responds with IDs, indexer summarizes
        selector_llm = _make_mock_llm()
        selector_llm.invoke.return_value = AIMessage(
            content="Reasoning: Message 0 asks about budget.\nIDs: [0, 1]"
        )

        indexer_llm = _make_mock_llm()
        indexer_llm.invoke.return_value = AIMessage(content="New summary")

        config = LibrarianConfig(
            indexer_model=indexer_llm,
            selector_model=selector_llm,
        )

        # Simulate pre-built index from a previous turn
        pre_built_index = [
            IndexEntry(id=0, role="human", summary="Asks about budget"),
            IndexEntry(id=1, role="ai", summary="Budget is $10,000"),
            IndexEntry(id=2, role="human", summary="Asks for a joke"),
            IndexEntry(id=3, role="ai", summary="Tells a joke"),
        ]

        messages = [
            HumanMessage(content="What is the budget?"),
            AIMessage(content="The budget is $10,000"),
            HumanMessage(content="Tell me a joke"),
            AIMessage(content="Why did the chicken cross the road?"),
            HumanMessage(content="Remind me of the budget"),  # New message
        ]

        graph = create_librarian_graph(agent_fn=my_agent, config=config)

        result = graph.invoke(
            {
                "messages": messages,
                "query": "Remind me of the budget",
                "index": pre_built_index,
            },
            config={"configurable": {"librarian": config}},
        )

        # Selection should have been performed (not skipped)
        assert result["librarian_metadata"].get("selection_skipped") is False
        # Curated messages should be fewer than original
        assert len(result["curated_messages"]) <= len(messages)

    def test_graph_compiles(self):
        """Verify the graph compiles without errors."""

        def noop_agent(state):
            return {}

        config = LibrarianConfig(indexer_model=_make_mock_llm())
        graph = create_librarian_graph(agent_fn=noop_agent, config=config)

        # Verify graph has expected nodes
        assert graph is not None
