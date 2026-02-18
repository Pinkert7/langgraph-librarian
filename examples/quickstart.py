"""Quickstart example: Using the LangGraph Librarian standalone.

This example shows the simplest way to use the Librarian:
wrap your agent function and let the Librarian handle context management.

Requirements:
    pip install langgraph-librarian langchain-google-genai

Usage:
    export GOOGLE_API_KEY=your-key
    python quickstart.py
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph_librarian import LibrarianConfig, create_librarian_graph


# ─── 1. Configure your models ───────────────────────────────────────────────
config = LibrarianConfig(
    # Fast, cheap model for summarizing messages
    indexer_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    # Reasoning-capable model for selecting relevant context
    selector_model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    # Always include last 2 user turns for coherence
    always_include_recent_turns=2,
)


# ─── 2. Define your agent function ──────────────────────────────────────────
def my_agent(state: dict) -> dict:
    """Your agent logic. Receives curated context in state['curated_messages']."""
    # Use curated_messages (Librarian-selected) instead of full messages
    context = state.get("curated_messages", state["messages"])

    # Build a simple prompt from the curated context
    context_str = "\n".join(
        f"[{m.type}]: {m.content}" for m in context
    )

    # Use any LLM for the final response
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke(
        f"Based on this conversation context:\n{context_str}\n\n"
        f"Answer: {state['query']}"
    )

    return {"messages": [response]}


# ─── 3. Create the graph ────────────────────────────────────────────────────
graph = create_librarian_graph(agent_fn=my_agent, config=config)


# ─── 4. Run it! ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a conversation with several turns
    conversation = [
        HumanMessage(content="I want to plan a trip to Japan in March. Budget is $3000."),
        AIMessage(content="Great choice! March is cherry blossom season. With $3000, you can..."),
        HumanMessage(content="Actually, let's talk about something else. What's the best Python web framework?"),
        AIMessage(content="There are several great options: Django, FastAPI, Flask..."),
        HumanMessage(content="Interesting. By the way, change my trip budget to $5000 and add Korea."),
        AIMessage(content="Updated! $5000 budget for Japan and Korea in March..."),
        HumanMessage(content="Let's discuss machine learning for a bit. What's a transformer?"),
        AIMessage(content="A transformer is a neural network architecture..."),
    ]

    # First turn: builds the index (no selection yet)
    result = graph.invoke(
        {
            "messages": conversation,
            "query": "What is the current budget for my trip?",
            "index": [],
        },
        config={"configurable": {"librarian": config}},
    )

    print("=" * 60)
    print("LIBRARIAN METRICS:")
    metadata = result.get("librarian_metadata", {})
    print(f"  Original messages: {metadata.get('original_count', 'N/A')}")
    print(f"  Curated messages:  {metadata.get('curated_count', 'N/A')}")
    print(f"  Token reduction:   {metadata.get('token_reduction_pct', 'N/A')}%")
    print(f"  Selected IDs:      {metadata.get('selected_ids', 'N/A')}")
    print("=" * 60)
    print("\nCURATED CONTEXT:")
    for m in result.get("curated_messages", []):
        print(f"  [{m.type}]: {m.content[:80]}...")
    print("\nAGENT RESPONSE:")
    print(f"  {result['messages'][-1].content}")
