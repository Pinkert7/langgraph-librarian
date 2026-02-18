"""Graph builders for the LangGraph Librarian.

Provides factory functions to create LangGraph StateGraphs that
implement the Librarian select-then-hydrate architecture.

Main entry points:
- create_librarian_graph: wraps a user's agent node with Librarian nodes
- create_librarian_node: returns a standalone select+hydrate callable
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .config import LibrarianConfig
from .nodes import index_messages, select_and_hydrate
from .state import LibrarianState


def create_librarian_graph(
    agent_fn: Callable[[dict], dict],
    config: LibrarianConfig | None = None,
    *,
    checkpointer: Any = None,
) -> Any:
    """Create a compiled LangGraph that wraps an agent with the Librarian.

    The resulting graph has the flow:
        START → select_and_hydrate → agent → index_messages → END

    - select_and_hydrate: reads pre-built index, selects relevant context
    - agent: your agent function (receives curated_messages in state)
    - index_messages: indexes new messages for next turn (post-response)

    When using streaming, the user sees the agent response before
    indexing completes — achieving zero-latency indexing.

    Args:
        agent_fn: Your agent node function. It receives the full state dict
            and should return a state update dict. The curated context is
            available at state["curated_messages"].
        config: LibrarianConfig instance. Can also be passed at runtime
            via the LangGraph configurable.
        checkpointer: Optional LangGraph checkpointer for state persistence.
            The index is stored in state, so the checkpointer automatically
            persists it across turns.

    Returns:
        A compiled LangGraph StateGraph ready for invoke/stream.

    Example:
        >>> from langgraph_librarian import create_librarian_graph, LibrarianConfig
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>>
        >>> config = LibrarianConfig(
        ...     indexer_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
        ...     selector_model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
        ... )
        >>>
        >>> def my_agent(state):
        ...     # Use state["curated_messages"] instead of state["messages"]
        ...     context = state["curated_messages"]
        ...     # ... generate response using context ...
        ...     return {"messages": [response]}
        >>>
        >>> graph = create_librarian_graph(my_agent, config)
        >>> result = graph.invoke(
        ...     {"messages": conversation_history, "query": "What was decided?"},
        ...     config={"configurable": {"librarian": config}},
        ... )
    """
    builder = StateGraph(LibrarianState)

    # Add nodes
    builder.add_node("select_and_hydrate", select_and_hydrate)
    builder.add_node("agent", agent_fn)
    builder.add_node("index_messages", index_messages)

    # Wire the flow: START → select → agent → index → END
    builder.add_edge(START, "select_and_hydrate")
    builder.add_edge("select_and_hydrate", "agent")
    builder.add_edge("agent", "index_messages")
    builder.add_edge("index_messages", END)

    # Compile with optional checkpointer
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return builder.compile(**compile_kwargs)


def create_librarian_node(
    config: LibrarianConfig | None = None,
) -> Callable[[dict], dict]:
    """Create a standalone select+hydrate callable for custom graph integration.

    Use this when you want to add the Librarian as a single node
    in your own StateGraph, without the full wrapper.

    You'll need to add the index_messages node separately in your graph
    (after your agent node) for async indexing to work.

    Args:
        config: LibrarianConfig instance.

    Returns:
        A callable suitable for builder.add_node("librarian", ...).

    Example:
        >>> from langgraph_librarian import create_librarian_node, index_messages
        >>>
        >>> librarian_node = create_librarian_node(config)
        >>>
        >>> builder = StateGraph(MyState)
        >>> builder.add_node("librarian", librarian_node)
        >>> builder.add_node("agent", my_agent)
        >>> builder.add_node("indexer", index_messages)
        >>> builder.add_edge(START, "librarian")
        >>> builder.add_edge("librarian", "agent")
        >>> builder.add_edge("agent", "indexer")
        >>> builder.add_edge("indexer", END)
    """

    def _node(state: dict, run_config: dict | None = None) -> dict:
        # Inject config if provided at build time
        if config is not None and run_config is not None:
            configurable = run_config.get("configurable", {})
            if "librarian" not in configurable:
                if "configurable" not in run_config:
                    run_config["configurable"] = {}
                run_config["configurable"]["librarian"] = config
        elif config is not None:
            run_config = {"configurable": {"librarian": config}}

        return select_and_hydrate(state, run_config)

    return _node
