"""Memory Store — keyword-indexed store of past analyses.

Agents query for relevant historical context (HTF assessments, SMC reports,
CMT assessments, trade plans) by symbol, timeframe, strategy, tags, and
time range. Entries have time-decayed relevance scoring.

Two implementations:
- ``InMemoryMemoryStore``: for backtest and tests.
- ``JsonFileMemoryStore``: for paper/live with JSONL persistence.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_trading.core.enums import MemoryEntryType
from agentic_trading.core.file_io import safe_append_line
from agentic_trading.core.ids import new_id as _uuid
from agentic_trading.core.ids import utc_now as _now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory Entry
# ---------------------------------------------------------------------------


class MemoryEntry(BaseModel):
    """A single remembered analysis."""

    entry_id: str = Field(default_factory=_uuid)
    entry_type: MemoryEntryType
    timestamp: datetime = Field(default_factory=_now)
    symbol: str = ""
    timeframe: str = ""
    strategy_id: str = ""
    tags: list[str] = Field(default_factory=list)
    content: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    relevance_score: float = 1.0
    ttl_hours: float = 24.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IMemoryStore(Protocol):
    """Keyword-indexed memory of past analyses."""

    def store(self, entry: MemoryEntry) -> None: ...

    def query(
        self,
        *,
        symbol: str | None = None,
        entry_type: MemoryEntryType | None = None,
        strategy_id: str | None = None,
        timeframe: str | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 10,
        min_relevance: float = 0.1,
    ) -> list[MemoryEntry]: ...

    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_decayed_relevance(entry: MemoryEntry, now: datetime) -> float:
    """Compute time-decayed relevance score.

    Uses exponential decay: score = base * exp(-lambda * age_hours)
    where lambda = ln(10) / ttl_hours so score drops to 0.1 at TTL.
    """
    age = (now - entry.timestamp).total_seconds() / 3600.0
    if age <= 0:
        return entry.relevance_score
    if entry.ttl_hours <= 0:
        return 0.0
    decay_lambda = math.log(10) / entry.ttl_hours
    return entry.relevance_score * math.exp(-decay_lambda * age)


def _matches_query(
    entry: MemoryEntry,
    *,
    symbol: str | None = None,
    entry_type: MemoryEntryType | None = None,
    strategy_id: str | None = None,
    timeframe: str | None = None,
    tags: list[str] | None = None,
    since: datetime | None = None,
) -> bool:
    """Check whether an entry matches query filters."""
    if symbol is not None and entry.symbol != symbol:
        return False
    if entry_type is not None and entry.entry_type != entry_type:
        return False
    if strategy_id is not None and entry.strategy_id != strategy_id:
        return False
    if timeframe is not None and entry.timeframe != timeframe:
        return False
    if tags is not None and not set(tags).issubset(set(entry.tags)):
        return False
    if since is not None and entry.timestamp < since:
        return False
    return True


# ---------------------------------------------------------------------------
# In-Memory Implementation
# ---------------------------------------------------------------------------


class InMemoryMemoryStore:
    """List-backed memory store for backtest and tests."""

    def __init__(self, *, max_entries: int = 10_000) -> None:
        self._entries: list[MemoryEntry] = []
        self._max_entries = max_entries

    def store(self, entry: MemoryEntry) -> None:
        """Store an entry, evicting oldest if at capacity."""
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def query(
        self,
        *,
        symbol: str | None = None,
        entry_type: MemoryEntryType | None = None,
        strategy_id: str | None = None,
        timeframe: str | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 10,
        min_relevance: float = 0.1,
    ) -> list[MemoryEntry]:
        """Query entries with filters and time-decayed relevance scoring."""
        now = _now()
        results: list[tuple[float, MemoryEntry]] = []

        for entry in self._entries:
            if not _matches_query(
                entry,
                symbol=symbol,
                entry_type=entry_type,
                strategy_id=strategy_id,
                timeframe=timeframe,
                tags=tags,
                since=since,
            ):
                continue

            decayed = _compute_decayed_relevance(entry, now)
            if decayed < min_relevance:
                continue

            results.append((decayed, entry))

        # Sort by decayed relevance descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        """Number of stored entries."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# JSONL File-Backed Implementation
# ---------------------------------------------------------------------------


class JsonFileMemoryStore:
    """JSONL-backed memory store for paper/live persistence.

    Appends to a JSONL file on every ``store()`` call.
    Loads existing entries into memory on init.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_entries: int = 10_000,
    ) -> None:
        self._path = Path(path)
        self._max_entries = max_entries
        self._inner = InMemoryMemoryStore(max_entries=max_entries)
        self._load()

    def _load(self) -> None:
        """Load existing entries from JSONL file."""
        if not self._path.exists():
            return

        count = 0
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = MemoryEntry.model_validate(data)
                        self._inner.store(entry)
                        count += 1
                    except Exception:
                        logger.warning("Skipping malformed memory entry")
        except Exception:
            logger.exception("Failed to load memory store from %s", self._path)

        if count > 0:
            logger.info("Loaded %d memory entries from %s", count, self._path)

    def store(self, entry: MemoryEntry) -> None:
        """Store entry in memory and append to JSONL file."""
        self._inner.store(entry)

        try:
            safe_append_line(self._path, entry.model_dump_json())
        except Exception:
            logger.exception("Failed to persist memory entry to %s", self._path)

    def query(
        self,
        *,
        symbol: str | None = None,
        entry_type: MemoryEntryType | None = None,
        strategy_id: str | None = None,
        timeframe: str | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 10,
        min_relevance: float = 0.1,
    ) -> list[MemoryEntry]:
        """Query entries (delegates to in-memory store)."""
        return self._inner.query(
            symbol=symbol,
            entry_type=entry_type,
            strategy_id=strategy_id,
            timeframe=timeframe,
            tags=tags,
            since=since,
            limit=limit,
            min_relevance=min_relevance,
        )

    def clear(self) -> None:
        """Clear in-memory entries. Does not delete the file."""
        self._inner.clear()

    @property
    def entry_count(self) -> int:
        """Number of in-memory entries."""
        return self._inner.entry_count
