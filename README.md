# 📚 LangGraph Librarian

**Intelligent context management for LangGraph chatbots.**

The Librarian uses LLM reasoning to select the most relevant messages from conversation history — dramatically reducing token usage while *improving* answer quality. Instead of feeding your agent the entire conversation (Brute Force) or relying on keyword similarity (RAG), the Librarian **reasons** about what's relevant.

## How It Works

```
┌────────────────────────────────────────────────────────────────────┐
│                    LangGraph Librarian Pipeline                    │
│                                                                    │
│   ┌──────────────────┐   ┌─────────┐   ┌──────────────────────┐    │
│   │ select & hydrate │──▶│  Agent  │──▶│ index new messages   │    │
│   │ (reads index,    │   │ (YOUR   │   │ (summarizes new msgs │    │
│   │  picks relevant  │   │  CODE)  │   │  for next turn)      │    │
│   │  messages)       │   │         │   │                      │    │
│   └──────────────────┘   └─────────┘   └──────────────────────┘    │
│        ONLINE ⚡              │              OFFLINE 🔄             │
│   User waits for this    User sees       Runs after response       │
│                          response here   (zero latency impact)     │
└────────────────────────────────────────────────────────────────────┘
```

### The Select-Then-Hydrate Architecture

1. **Index** (async, post-response): Each message is summarized into a ≤3-sentence entry (~104 tokens vs ~328 tokens original). This runs *after* the agent responds, so the user never waits for it.

2. **Select** (online): An LLM reasons over the summary index to pick which messages are relevant to the current query. This uses reasoning, not keyword matching — it can identify temporal dependencies like "Message 3 modifies Message 1".

3. **Hydrate** (online, no LLM): Fetches the full original content for selected messages + always includes the most recent turns for conversational coherence.

## Benchmarks

Tested on synthetic chat datasets with varying noise levels (150–1000 words per message):

| Metric | Librarian | Brute Force | Vector RAG |
|---|---|---|---|
| **Answer Success** | **82.2%** | 77.8% | 57.8% |
| **Context Success** | **80.0%** | 100% | 68.9% |
| **Context Tokens** | **~800** | ~2180 | ~1400 |
| **Cost (tokens)** | **~2800** | ~4520 | ~2950 |

> The Librarian outperforms Brute Force on answer quality (by removing distracting noise) and beats RAG by 11% on context retrieval accuracy.

## Installation

```bash
pip install langgraph-librarian
```

You'll also need a LangChain LLM provider:
```bash
# Pick one (or more):
pip install langchain-google-genai   # Google Gemini
pip install langchain-openai         # OpenAI
pip install langchain-anthropic      # Anthropic
```

## Quick Start

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_librarian import create_librarian_graph, LibrarianConfig

# 1. Configure models
config = LibrarianConfig(
    indexer_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),    # cheap & fast
    selector_model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),   # reasoning
)

# 2. Define your agent (only change: use state["curated_messages"])
def my_agent(state):
    context = state.get("curated_messages", state["messages"])
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke(list(context))
    return {"messages": [response]}

# 3. Create and run
graph = create_librarian_graph(agent_fn=my_agent, config=config)
result = graph.invoke(
    {"messages": conversation, "query": "What was the budget?", "index": []},
    config={"configurable": {"librarian": config}},
)
```

## Integration Patterns

### Pattern 1: Wrap Your Agent (Easiest)

```python
graph = create_librarian_graph(agent_fn=my_agent, config=config)
```

Creates a complete graph: `select_and_hydrate → your_agent → index_messages`

### Pattern 2: Add Nodes to Your Graph (Flexible)

```python
from langgraph_librarian import create_librarian_node, index_messages

builder = StateGraph(MyState)
builder.add_node("librarian", create_librarian_node(config))
builder.add_node("agent", my_agent)
builder.add_node("indexer", index_messages)
builder.add_edge(START, "librarian")
builder.add_edge("librarian", "agent")
builder.add_edge("agent", "indexer")
builder.add_edge("indexer", END)
```

### Pattern 3: With Checkpointer (Persistence)

```python
from langgraph.checkpoint.memory import MemorySaver

graph = create_librarian_graph(
    agent_fn=my_agent,
    config=config,
    checkpointer=MemorySaver(),  # Index persists automatically!
)

# Each thread maintains its own index
result = graph.invoke(
    {"messages": msgs, "query": q, "index": []},
    config={"configurable": {"thread_id": "user-123", "librarian": config}},
)
```

---

## API Reference

### Graph Builders

#### `create_librarian_graph(agent_fn, config, *, checkpointer=None)`

Creates a compiled LangGraph `StateGraph` that wraps your agent with Librarian nodes.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `agent_fn` | `Callable[[dict], dict]` | Your agent node function. Receives the full state dict and should return a state update. The curated context is at `state["curated_messages"]`. |
| `config` | `LibrarianConfig \| None` | Librarian configuration. Can also be passed at runtime via the configurable. |
| `checkpointer` | `Any` | Optional LangGraph checkpointer for state persistence. The index in state is automatically persisted across turns. |

**Returns:** A compiled `StateGraph` with the flow: `START → select_and_hydrate → agent → index_messages → END`

**Example:**
```python
from langgraph_librarian import create_librarian_graph, LibrarianConfig
from langchain_google_genai import ChatGoogleGenerativeAI

config = LibrarianConfig(
    indexer_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
)

def my_agent(state):
    context = state.get("curated_messages", state["messages"])
    # ... generate response using context ...
    return {"messages": [response]}

graph = create_librarian_graph(agent_fn=my_agent, config=config)
```

---

#### `create_librarian_node(config)`

Returns a standalone `select_and_hydrate` callable for use as a node in your own `StateGraph`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `LibrarianConfig \| None` | Librarian configuration. |

**Returns:** `Callable[[dict, RunnableConfig \| None], dict]` — a function suitable for `builder.add_node()`.

> **Note:** When using this, you must also add the `index_messages` node separately in your graph (after your agent node) for async indexing to work.

**Example:**
```python
from langgraph_librarian import create_librarian_node, index_messages

librarian_node = create_librarian_node(config)

builder = StateGraph(MyState)
builder.add_node("librarian", librarian_node)
builder.add_node("agent", my_agent)
builder.add_node("indexer", index_messages)
builder.add_edge(START, "librarian")
builder.add_edge("librarian", "agent")
builder.add_edge("agent", "indexer")
builder.add_edge("indexer", END)
```

---

### Node Functions

#### `select_and_hydrate(state, config=None)`

LangGraph node that performs the **online** selection and hydration step.

1. Reads the pre-built summary index from state
2. Calls the selector LLM to reason over the index and pick relevant message IDs
3. Hydrates selected messages with full content + always includes recent turns
4. On first turn (no index), passes through all messages unchanged

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `state` | `dict` | Current graph state. Reads `messages`, `query`, `index`. |
| `config` | `RunnableConfig \| None` | LangGraph config. Expects `LibrarianConfig` at `config["configurable"]["librarian"]`. |

**Returns:** State update dict with:

| Key | Type | Description |
|---|---|---|
| `curated_messages` | `list[BaseMessage]` | Selected + recent messages for the agent |
| `selected_ids` | `list[int]` | IDs selected by the Librarian |
| `librarian_metadata` | `dict` | Metrics (see [Metadata Fields](#metadata-fields)) |

---

#### `index_messages(state, config=None)`

LangGraph node that performs **offline** incremental indexing.

Summarizes only new messages (messages not yet in the index), making it safe and efficient to run after every turn. Short messages (< `short_message_threshold` chars) are used as-is without an LLM call.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `state` | `dict` | Current graph state. Reads `messages`, `index`. |
| `config` | `RunnableConfig \| None` | LangGraph config. Expects `LibrarianConfig` at `config["configurable"]["librarian"]`. |

**Returns:** State update dict with:

| Key | Type | Description |
|---|---|---|
| `index` | `list[IndexEntry]` | Updated index with new entries appended |
| `librarian_metadata` | `dict` | Indexing metrics (`indexing_new_count`, `indexing_total_count`, `indexing_latency_ms`) |

---

### Configuration

#### `LibrarianConfig`

Dataclass controlling the Librarian pipeline behavior.

| Field | Type | Default | Description |
|---|---|---|---|
| `indexer_model` | `BaseChatModel` | *Required* | LLM for summarizing messages. Should be fast and cheap (e.g., Gemini Flash, GPT-4o-mini). |
| `selector_model` | `BaseChatModel \| None` | `None` | LLM for selection reasoning. Should be reasoning-capable. Falls back to `indexer_model` if not set. |
| `always_include_recent_turns` | `int` | `2` | Number of recent user turns to always include regardless of selection. Ensures conversational coherence. |
| `short_message_threshold` | `int` | `200` | Messages shorter than this (in characters) skip summarization and are used as-is. |
| `summary_prompt` | `str \| None` | `None` | Custom prompt for the summarization step. Uses a battle-tested default if not set. |
| `selection_prompt` | `str \| None` | `None` | Custom prompt template for the selection step. Must contain `{query}` and `{index}` placeholders. |
| `store` | `BaseIndexStore \| None` | `None` | Optional external index store. If not set, index is persisted via the LangGraph checkpointer (recommended). |

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `get_indexer_model()` | `BaseChatModel` | Returns the indexer model. Raises `ValueError` if not configured. |
| `get_selector_model()` | `BaseChatModel` | Returns the selector model, falling back to the indexer model. |

**Example:**
```python
LibrarianConfig(
    indexer_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    selector_model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    always_include_recent_turns=3,
    short_message_threshold=150,
    summary_prompt="Summarize in exactly 2 sentences. Focus on decisions and constraints.",
    selection_prompt="Custom template with {query} and {index} placeholders...",
)
```

---

### State Types

#### `LibrarianState` (TypedDict)

The graph state schema. Extend your own state with these fields.

| Field | Type | Reducer | Description |
|---|---|---|---|
| `messages` | `Sequence[BaseMessage]` | `add_messages` | Full conversation history |
| `query` | `str` | — | Current user query |
| `index` | `list[IndexEntry]` | replace | Summary index (persisted across turns) |
| `selected_ids` | `list[int]` | — | IDs selected by the Librarian |
| `curated_messages` | `Sequence[BaseMessage]` | — | **← Use this in your agent** |
| `librarian_metadata` | `dict[str, Any]` | merge | Metrics and debug info |

---

#### `IndexEntry` (dataclass)

A single message summary in the index.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `int` | — | Zero-based position in the original message array |
| `role` | `str` | — | Message role (`"human"`, `"ai"`, `"tool"`, etc.) |
| `summary` | `str` | — | ≤3-sentence summary of the message content |
| `original_tokens` | `int` | `0` | Estimated token count of the original message |
| `indexing_latency_ms` | `float` | `0.0` | Time taken to generate this summary (ms) |

---

### Storage

#### `BaseIndexStore` (Protocol)

Protocol for custom index storage backends. Implement this to create your own store.

| Method | Signature | Description |
|---|---|---|
| `load` | `(session_id: str) → list[IndexEntry] \| None` | Load an existing index. Returns `None` if not found. |
| `save` | `(session_id: str, entries: list[IndexEntry]) → None` | Save an index. |
| `delete` | `(session_id: str) → None` | Delete an index. |

---

#### `FileIndexStore`

JSON file-based index persistence. Each session's index is stored as a separate `.librarian-index.json` file.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `directory` | `str` | Directory path to store index files |

**Example:**
```python
from langgraph_librarian import LibrarianConfig, FileIndexStore

config = LibrarianConfig(
    indexer_model=my_llm,
    store=FileIndexStore("/path/to/indexes"),
)
```

---

### Metadata Fields

The `librarian_metadata` dict in state contains these keys after each step:

**After `select_and_hydrate`:**

| Key | Type | Description |
|---|---|---|
| `selection_skipped` | `bool` | `True` if selection was skipped (no index or no query) |
| `reason` | `str` | Why selection was skipped (only when `selection_skipped=True`) |
| `selected_ids` | `list[int]` | IDs selected by the Librarian |
| `selection_reasoning` | `str` | Raw reasoning output from the selector LLM |
| `selection_latency_ms` | `float` | Time taken for selection + hydration (ms) |
| `original_count` | `int` | Number of messages in full history |
| `curated_count` | `int` | Number of curated messages |
| `original_tokens` | `int` | Estimated tokens in full history |
| `curated_tokens` | `int` | Estimated tokens in curated context |
| `token_reduction_pct` | `float` | Percentage of tokens saved |

**After `index_messages`:**

| Key | Type | Description |
|---|---|---|
| `indexing_new_count` | `int` | Number of newly indexed messages |
| `indexing_total_count` | `int` | Total entries in the index |
| `indexing_latency_ms` | `float` | Total time for indexing new messages (ms) |

---

## Development

```bash
git clone https://github.com/Pinkert7/langgraph-librarian.git
cd langgraph-librarian
pip install -e ".[test]"
pytest tests/ -v
```

## License

MIT
