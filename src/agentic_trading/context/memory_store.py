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

    def search_similar(
        self,
        query_text: str,
        *,
        entry_type: MemoryEntryType | None = None,
        symbol: str | None = None,
        limit: int = 5,
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
# BM25 text scoring
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lower-case split, strip punctuation, remove 1-char tokens."""
    import re
    return [t for t in re.split(r"[\s/\-_:,.|()]+", text.lower()) if len(t) > 1]


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    df: dict[str, int],
    n_docs: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Okapi BM25 score for a single document against query tokens."""
    if not doc_tokens or not query_tokens or n_docs == 0:
        return 0.0
    dl = len(doc_tokens)
    score = 0.0
    tf_map: dict[str, int] = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1

    for qt in query_tokens:
        if qt not in tf_map:
            continue
        tf = tf_map[qt]
        d = df.get(qt, 0)
        idf = math.log((n_docs - d + 0.5) / (d + 0.5) + 1.0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        score += idf * tf_norm
    return score


def _entry_tokens(entry: MemoryEntry) -> list[str]:
    """Tokenize an entry's summary + tags for BM25 search."""
    text = entry.summary + " " + " ".join(entry.tags)
    if entry.symbol:
        text += " " + entry.symbol
    if entry.strategy_id:
        text += " " + entry.strategy_id
    return _tokenize(text)


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

    def search_similar(
        self,
        query_text: str,
        *,
        entry_type: MemoryEntryType | None = None,
        symbol: str | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """BM25 text-similarity search over entry summaries and tags.

        Returns entries ranked by combined BM25 score and time decay.
        """
        if not self._entries or not query_text.strip():
            return []

        now = _now()
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        # Pre-filter by type and symbol
        candidates = self._entries
        if entry_type is not None:
            candidates = [e for e in candidates if e.entry_type == entry_type]
        if symbol is not None:
            candidates = [e for e in candidates if e.symbol == symbol]

        if not candidates:
            return []

        # Build doc tokens and DF
        all_doc_tokens = [_entry_tokens(e) for e in candidates]
        df: dict[str, int] = {}
        total_len = 0
        for dt in all_doc_tokens:
            seen: set[str] = set()
            total_len += len(dt)
            for t in dt:
                if t not in seen:
                    df[t] = df.get(t, 0) + 1
                    seen.add(t)
        avg_dl = total_len / len(candidates) if candidates else 1.0

        # Score and rank
        scored: list[tuple[float, MemoryEntry]] = []
        for entry, doc_tokens in zip(candidates, all_doc_tokens):
            bm25 = _bm25_score(
                query_tokens, doc_tokens, avg_dl, df, len(candidates),
            )
            if bm25 <= 0:
                continue
            decay = _compute_decayed_relevance(entry, now)
            scored.append((bm25 * decay, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

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

    def search_similar(
        self,
        query_text: str,
        *,
        entry_type: MemoryEntryType | None = None,
        symbol: str | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """BM25 text-similarity search (delegates to in-memory store)."""
        return self._inner.search_similar(
            query_text, entry_type=entry_type, symbol=symbol, limit=limit,
        )

    def clear(self) -> None:
        """Clear in-memory entries. Does not delete the file."""
        self._inner.clear()

    @property
    def entry_count(self) -> int:
        """Number of in-memory entries."""
        return self._inner.entry_count
