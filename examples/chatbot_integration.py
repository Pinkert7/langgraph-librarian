"""Integration example: Adding the Librarian to an existing LangGraph chatbot.

This example shows how to integrate the Librarian into a LangGraph chatbot
that already uses StateGraph, by adding it as nodes in your graph.

This gives you more control over the graph flow while still benefiting
from the Librarian's intelligent context management.

Requirements:
    pip install langgraph-librarian langchain-google-genai

Usage:
    export GOOGLE_API_KEY=your-key
    python chatbot_integration.py
"""

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langgraph_librarian import (
    LibrarianConfig,
    create_librarian_node,
    index_messages,
)
from langgraph_librarian.state import IndexEntry, _replace_index, _replace_metadata


# ─── 1. Define your state (extend with Librarian fields) ────────────────────

class ChatbotState:
    """Your chatbot state, extended with Librarian fields."""
    __annotations__ = {
        # Standard chatbot fields
        "messages": Annotated[Sequence[BaseMessage], add_messages],
        "query": str,
        # Librarian fields (add these to your existing state)
        "index": Annotated[list[IndexEntry], _replace_index],
        "selected_ids": list[int],
        "curated_messages": Sequence[BaseMessage],
        "librarian_metadata": Annotated[dict, _replace_metadata],
    }


# ─── 2. Configure the Librarian ─────────────────────────────────────────────

librarian_config = LibrarianConfig(
    indexer_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    selector_model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    always_include_recent_turns=2,
)


# ─── 3. Define your agent node (same as before) ─────────────────────────────

def chatbot_agent(state: dict) -> dict:
    """Your existing chatbot logic.

    The only change: use state['curated_messages'] for context
    instead of state['messages'].
    """
    # Curated messages are set by the Librarian node
    context = state.get("curated_messages", state["messages"])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke(list(context))

    return {"messages": [response]}


# ─── 4. Build the graph with Librarian nodes ─────────────────────────────────

def build_graph():
    builder = StateGraph(ChatbotState)

    # Add the Librarian select+hydrate node
    librarian_node = create_librarian_node(librarian_config)
    builder.add_node("librarian", librarian_node)

    # Add your agent
    builder.add_node("chatbot", chatbot_agent)

    # Add the Librarian indexer (runs after agent response)
    builder.add_node("indexer", index_messages)

    # Wire: START → librarian → chatbot → indexer → END
    builder.add_edge(START, "librarian")
    builder.add_edge("librarian", "chatbot")
    builder.add_edge("chatbot", "indexer")
    builder.add_edge("indexer", END)

    return builder.compile()


# ─── 5. Run the chatbot ─────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = build_graph()

    # Simulate multi-turn conversation
    messages = [
        HumanMessage(content="My name is Alice and I'm working on Project Phoenix."),
    ]

    # Turn 1
    result = graph.invoke(
        {
            "messages": messages,
            "query": messages[-1].content,
            "index": [],
        },
        config={"configurable": {"librarian": librarian_config}},
    )

    print("Turn 1 response:", result["messages"][-1].content[:100])
    print(f"Index entries: {len(result.get('index', []))}")

    # Turn 2: add more messages and query
    messages = list(result["messages"])
    messages.append(HumanMessage(content="What project am I working on?"))

    result = graph.invoke(
        {
            "messages": messages,
            "query": "What project am I working on?",
            "index": result.get("index", []),
        },
        config={"configurable": {"librarian": librarian_config}},
    )

    print("\nTurn 2 response:", result["messages"][-1].content[:100])
    metadata = result.get("librarian_metadata", {})
    print(f"Token reduction: {metadata.get('token_reduction_pct', 'N/A')}%")
