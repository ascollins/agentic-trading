"""Conversation Store — persistence and query for Soteria conversations.

Two implementations following the dual-store pattern:
- ``InMemoryConversationStore`` — for backtest/tests (no I/O)
- ``ConversationStore`` — for paper/live (Postgres via SQLAlchemy async)

Query interface:
- ``explain(conversation_id)`` → rendered explanation
- ``find_by_symbol(symbol, limit)`` → recent conversations for a symbol
- ``find_vetoed(symbol, limit)`` → conversations with VETO messages
- ``find_disagreements(symbol, limit)`` → conversations with CHALLENGE/DISAGREEMENT
- ``replay(conversation_id)`` → full AgentConversation for replay
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from agentic_trading.core.file_io import safe_append_line

from .agent_conversation import AgentConversation, ConversationOutcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InMemoryConversationStore
# ---------------------------------------------------------------------------


class InMemoryConversationStore:
    """In-memory conversation store for backtest and tests.

    Parameters
    ----------
    max_entries:
        Maximum conversations to retain. Oldest evicted when exceeded.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._max_entries = max_entries
        self._conversations: dict[str, AgentConversation] = {}
        self._order: list[str] = []  # insertion order for eviction

    def save(self, conversation: AgentConversation) -> None:
        """Store a conversation."""
        cid = conversation.conversation_id
        if cid not in self._conversations:
            self._order.append(cid)
        self._conversations[cid] = conversation

        # Evict oldest
        while len(self._conversations) > self._max_entries:
            oldest = self._order.pop(0)
            self._conversations.pop(oldest, None)

    def load(self, conversation_id: str) -> AgentConversation | None:
        """Load a conversation by ID."""
        return self._conversations.get(conversation_id)

    def explain(self, conversation_id: str) -> str:
        """Get a narrative explanation of a conversation."""
        conv = self.load(conversation_id)
        if conv is None:
            return f"Conversation {conversation_id} not found."
        return conv.explain()

    def find_by_symbol(
        self,
        symbol: str,
        *,
        limit: int = 20,
        since: datetime | None = None,
    ) -> list[AgentConversation]:
        """Find recent conversations for a symbol."""
        results: list[AgentConversation] = []
        for cid in reversed(self._order):
            conv = self._conversations.get(cid)
            if conv is None:
                continue
            if conv.symbol != symbol:
                continue
            if since and conv.started_at < since:
                continue
            results.append(conv)
            if len(results) >= limit:
                break
        return results

    def find_vetoed(
        self,
        symbol: str | None = None,
        *,
        limit: int = 20,
    ) -> list[AgentConversation]:
        """Find conversations that were vetoed."""
        results: list[AgentConversation] = []
        for cid in reversed(self._order):
            conv = self._conversations.get(cid)
            if conv is None:
                continue
            if symbol and conv.symbol != symbol:
                continue
            if not conv.has_veto:
                continue
            results.append(conv)
            if len(results) >= limit:
                break
        return results

    def find_disagreements(
        self,
        symbol: str | None = None,
        *,
        limit: int = 20,
    ) -> list[AgentConversation]:
        """Find conversations where agents disagreed."""
        results: list[AgentConversation] = []
        for cid in reversed(self._order):
            conv = self._conversations.get(cid)
            if conv is None:
                continue
            if symbol and conv.symbol != symbol:
                continue
            if not conv.has_disagreement:
                continue
            results.append(conv)
            if len(results) >= limit:
                break
        return results

    def find_by_outcome(
        self,
        outcome: ConversationOutcome,
        *,
        symbol: str | None = None,
        limit: int = 20,
    ) -> list[AgentConversation]:
        """Find conversations by outcome."""
        results: list[AgentConversation] = []
        for cid in reversed(self._order):
            conv = self._conversations.get(cid)
            if conv is None:
                continue
            if conv.outcome != outcome:
                continue
            if symbol and conv.symbol != symbol:
                continue
            results.append(conv)
            if len(results) >= limit:
                break
        return results

    def replay(self, conversation_id: str) -> AgentConversation | None:
        """Load the full conversation for replay."""
        return self.load(conversation_id)

    def query(
        self,
        *,
        symbol: str | None = None,
        strategy_id: str | None = None,
        outcome: ConversationOutcome | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[AgentConversation]:
        """General-purpose query with multiple filters."""
        results: list[AgentConversation] = []
        for cid in reversed(self._order):
            conv = self._conversations.get(cid)
            if conv is None:
                continue
            if symbol and conv.symbol != symbol:
                continue
            if strategy_id and conv.strategy_id != strategy_id:
                continue
            if outcome and conv.outcome != outcome:
                continue
            if since and conv.started_at < since:
                continue
            results.append(conv)
            if len(results) >= limit:
                break
        return results

    @property
    def count(self) -> int:
        return len(self._conversations)

    def clear(self) -> None:
        """Clear all stored conversations."""
        self._conversations.clear()
        self._order.clear()


# ---------------------------------------------------------------------------
# JsonFileConversationStore
# ---------------------------------------------------------------------------


class JsonFileConversationStore:
    """JSONL-backed conversation store for paper/live mode.

    Follows the same JSONL persistence pattern as ``JsonFileMemoryStore``
    and ``JsonFileEventStore``.

    Parameters
    ----------
    path:
        Path to the JSONL file.
    max_entries:
        Maximum entries in memory. Oldest evicted from memory
        (but remain in the file for long-term storage).
    """

    def __init__(self, path: str, *, max_entries: int = 10_000) -> None:
        self._path = Path(path)
        self._inner = InMemoryConversationStore(max_entries=max_entries)

        # Load existing data
        self._load_existing()

    def _load_existing(self) -> None:
        """Load conversations from JSONL file on startup."""
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
                        conv = AgentConversation.model_validate(data)
                        self._inner.save(conv)
                        count += 1
                    except Exception:
                        logger.debug("Skipping invalid conversation line")
        except Exception:
            logger.warning("Could not load conversations from %s", self._path)

        if count > 0:
            logger.info(
                "Loaded %d conversations from %s", count, self._path
            )

    def save(self, conversation: AgentConversation) -> None:
        """Store a conversation (memory + append to file)."""
        self._inner.save(conversation)

        try:
            safe_append_line(self._path, conversation.model_dump_json())
        except Exception:
            logger.warning(
                "Failed to persist conversation %s",
                conversation.conversation_id[:8],
            )

    def load(self, conversation_id: str) -> AgentConversation | None:
        return self._inner.load(conversation_id)

    def explain(self, conversation_id: str) -> str:
        return self._inner.explain(conversation_id)

    def find_by_symbol(
        self, symbol: str, *, limit: int = 20, since: datetime | None = None
    ) -> list[AgentConversation]:
        return self._inner.find_by_symbol(symbol, limit=limit, since=since)

    def find_vetoed(
        self, symbol: str | None = None, *, limit: int = 20
    ) -> list[AgentConversation]:
        return self._inner.find_vetoed(symbol, limit=limit)

    def find_disagreements(
        self, symbol: str | None = None, *, limit: int = 20
    ) -> list[AgentConversation]:
        return self._inner.find_disagreements(symbol, limit=limit)

    def find_by_outcome(
        self,
        outcome: ConversationOutcome,
        *,
        symbol: str | None = None,
        limit: int = 20,
    ) -> list[AgentConversation]:
        return self._inner.find_by_outcome(outcome, symbol=symbol, limit=limit)

    def replay(self, conversation_id: str) -> AgentConversation | None:
        return self._inner.replay(conversation_id)

    def query(
        self,
        *,
        symbol: str | None = None,
        strategy_id: str | None = None,
        outcome: ConversationOutcome | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[AgentConversation]:
        return self._inner.query(
            symbol=symbol,
            strategy_id=strategy_id,
            outcome=outcome,
            since=since,
            limit=limit,
        )

    @property
    def count(self) -> int:
        return self._inner.count

    def clear(self) -> None:
        self._inner.clear()
