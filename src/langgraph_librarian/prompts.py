"""Default prompts for the LangGraph Librarian.

These prompts are battle-tested from the OpenClaw Librarian implementation
and benchmark evaluation (80% context success vs 69% for Vector RAG).
"""

DEFAULT_SUMMARY_PROMPT = (
    "Summarize this message in 1-3 sentences. Capture: key decisions, constraints, "
    "specific values (names, numbers, paths), and action items. Be factual and dense."
)

DEFAULT_SELECTION_PROMPT = """You are a context librarian. Given a user query and a summary index of conversation messages, select which messages contain information needed to answer the query.

User Query: "{query}"

Summary Index:
{index}

Instructions:
1. Analyze the query to understand what specific information is needed.
2. Scan the summaries to find where this information is located.
3. Reason about which messages contain unique constraints or context.
4. Select a MINIMAL set of IDs that covers ALL required information.
5. When in doubt, include the message — false negatives are worse than false positives.

Output Format:
Reasoning: <your reasoning about which messages are relevant and why>
IDs: [id1, id2, ...]"""
