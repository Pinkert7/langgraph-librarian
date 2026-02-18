"""Index persistence for the LangGraph Librarian.

Provides optional external storage for the summary index.
By default, the index is persisted via the LangGraph checkpointer
(stored in graph state). These stores are for advanced use cases
where you want external/shared index storage.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Protocol, runtime_checkable

from .state import IndexEntry


@runtime_checkable
class BaseIndexStore(Protocol):
    """Protocol for index persistence backends."""

    def load(self, session_id: str) -> list[IndexEntry] | None:
        """Load an existing index for a session.

        Returns None if no index exists.
        """
        ...

    def save(self, session_id: str, entries: list[IndexEntry]) -> None:
        """Save an index for a session."""
        ...

    def delete(self, session_id: str) -> None:
        """Delete the index for a session."""
        ...


class FileIndexStore:
    """JSON file-based index persistence.

    Stores each session's index as a separate JSON file.
    Useful for sharing indexes across graphs or for debugging.

    Args:
        directory: Directory path to store index files.
    """

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, session_id: str) -> str:
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.directory, f"{safe_id}.librarian-index.json")

    def load(self, session_id: str) -> list[IndexEntry] | None:
        path = self._path(session_id)
        try:
            with open(path) as f:
                data = json.load(f)
            return [
                IndexEntry(
                    id=entry["id"],
                    role=entry["role"],
                    summary=entry["summary"],
                    original_tokens=entry.get("original_tokens", 0),
                    indexing_latency_ms=entry.get("indexing_latency_ms", 0.0),
                )
                for entry in data
            ]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def save(self, session_id: str, entries: list[IndexEntry]) -> None:
        path = self._path(session_id)
        with open(path, "w") as f:
            json.dump([asdict(entry) for entry in entries], f, indent=2)

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
