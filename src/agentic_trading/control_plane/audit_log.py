"""Append-only audit event journal.

Contract:
    - append() MUST succeed or raise (no silent drops)
    - If append() raises, the caller (ToolGateway) MUST NOT proceed
    - read() returns all entries for a correlation_id
    - Entries are immutable after append
    - No delete/update operations exist

Phase 1: In-memory with optional JSONL file persistence.
Phase 2: Postgres-backed with WAL (future).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from .action_types import AuditEntry

logger = logging.getLogger(__name__)


class AuditLog:
    """Append-only audit event journal.

    The AuditLog is a critical dependency of the ToolGateway.
    If it is unavailable, the ToolGateway MUST refuse all mutating calls.
    This is the fail-closed contract.
    """

    def __init__(
        self,
        persist_path: str | None = None,
        max_memory_entries: int = 100_000,
    ) -> None:
        self._entries: list[AuditEntry] = []
        self._by_correlation: dict[str, list[AuditEntry]] = defaultdict(list)
        self._by_action: dict[str, list[AuditEntry]] = defaultdict(list)
        self._persist_path = persist_path
        self._max = max_memory_entries
        self._available = True

    async def append(self, entry: AuditEntry) -> None:
        """Append an audit entry.

        Raises:
            RuntimeError: If the audit log is unavailable or persistence
                fails.  Callers MUST treat this as a hard stop.
        """
        if not self._available:
            raise RuntimeError("AuditLog is unavailable")

        self._entries.append(entry)
        self._by_correlation[entry.correlation_id].append(entry)
        if entry.causation_id:
            self._by_action[entry.causation_id].append(entry)

        # Persist if configured
        if self._persist_path:
            try:
                path = Path(self._persist_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write(entry.model_dump_json() + "\n")
            except Exception as exc:
                # Persistence failure makes the log unavailable
                self._available = False
                raise RuntimeError(
                    f"AuditLog persistence failed: {exc}"
                ) from exc

        # Memory cap: evict oldest entries
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max :]

    def read(self, correlation_id: str) -> list[AuditEntry]:
        """Read all entries for a correlation_id (the full action trace)."""
        return list(self._by_correlation.get(correlation_id, []))

    def read_by_action(self, action_id: str) -> list[AuditEntry]:
        """Read all entries causally linked to an action_id."""
        return list(self._by_action.get(action_id, []))

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        """Toggle availability. Primary use: testing."""
        self._available = available
