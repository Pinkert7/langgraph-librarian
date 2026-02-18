"""LangGraph Librarian — Intelligent context management for LangGraph chatbots.

The Librarian implements a "select-then-hydrate" architecture that uses
LLM reasoning to pick the most relevant messages from conversation history,
dramatically reducing token usage while maintaining answer quality.

Quick Start:
    >>> from langgraph_librarian import create_librarian_graph, LibrarianConfig
    >>> from langchain_google_genai import ChatGoogleGenerativeAI
    >>>
    >>> config = LibrarianConfig(
    ...     indexer_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    ...     selector_model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    ... )
    >>>
    >>> graph = create_librarian_graph(agent_fn=my_agent, config=config)
"""

from .config import LibrarianConfig
from .graph import create_librarian_graph, create_librarian_node
from .nodes import index_messages, select_and_hydrate
from .state import IndexEntry, LibrarianState
from .store import BaseIndexStore, FileIndexStore

__all__ = [
    # Graph builders
    "create_librarian_graph",
    "create_librarian_node",
    # Node functions
    "select_and_hydrate",
    "index_messages",
    # Configuration
    "LibrarianConfig",
    # State types
    "LibrarianState",
    "IndexEntry",
    # Storage
    "BaseIndexStore",
    "FileIndexStore",
]

__version__ = "0.1.0"
