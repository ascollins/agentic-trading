"""LLM interaction persistence.

``IInteractionStore`` is the protocol.  Two implementations ship:

* ``MemoryInteractionStore`` -- for backtest mode and unit tests.
* ``JsonFileInteractionStore`` -- for paper / live modes (JSONL).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from agentic_trading.core.file_io import safe_append_line
from agentic_trading.llm.envelope import LLMInteraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IInteractionStore(Protocol):
    """Append-only store for LLM interactions."""

    async def store(self, interaction: LLMInteraction) -> None:
        """Persist an interaction."""
        ...

    async def get_by_envelope_id(
        self,
        envelope_id: str,
    ) -> LLMInteraction | None:
        """Retrieve by envelope_id."""
        ...

    async def get_by_trace_id(
        self,
        trace_id: str,
    ) -> list[LLMInteraction]:
        """Retrieve all interactions for a given trace."""
        ...

    async def recent(
        self,
        *,
        limit: int = 50,
    ) -> list[LLMInteraction]:
        """Retrieve the most recent interactions."""
        ...


# ---------------------------------------------------------------------------
# MemoryInteractionStore  (backtest + tests)
# ---------------------------------------------------------------------------


class MemoryInteractionStore:
    """In-memory implementation -- no persistence."""

    def __init__(self) -> None:
        self._items: list[LLMInteraction] = []
        self._by_envelope_id: dict[str, LLMInteraction] = {}

    async def store(self, interaction: LLMInteraction) -> None:
        self._items.append(interaction)
        self._by_envelope_id[interaction.envelope.envelope_id] = interaction

    async def get_by_envelope_id(
        self,
        envelope_id: str,
    ) -> LLMInteraction | None:
        return self._by_envelope_id.get(envelope_id)

    async def get_by_trace_id(
        self,
        trace_id: str,
    ) -> list[LLMInteraction]:
        return [
            i for i in self._items
            if i.envelope.trace_id == trace_id
        ]

    async def recent(
        self,
        *,
        limit: int = 50,
    ) -> list[LLMInteraction]:
        return list(reversed(self._items[-limit:]))

    # -- helpers for tests --------------------------------------------------

    @property
    def items(self) -> list[LLMInteraction]:
        return self._items

    def clear(self) -> None:
        self._items.clear()
        self._by_envelope_id.clear()


# ---------------------------------------------------------------------------
# JsonFileInteractionStore  (paper + live)
# ---------------------------------------------------------------------------


class JsonFileInteractionStore:
    """JSONL file-backed implementation.

    Follows the same append-on-write, load-on-init pattern as
    ``NarrationStore`` and ``JsonFileMemoryStore``.
    """

    def __init__(
        self,
        path: str | Path = "data/llm_interactions.jsonl",
    ) -> None:
        self._path = Path(path)
        self._items: list[LLMInteraction] = []
        self._by_envelope_id: dict[str, LLMInteraction] = {}
        self._load()

    # -- public API ---------------------------------------------------------

    async def store(self, interaction: LLMInteraction) -> None:
        self._items.append(interaction)
        self._by_envelope_id[interaction.envelope.envelope_id] = interaction
        self._persist(interaction)

    async def get_by_envelope_id(
        self,
        envelope_id: str,
    ) -> LLMInteraction | None:
        return self._by_envelope_id.get(envelope_id)

    async def get_by_trace_id(
        self,
        trace_id: str,
    ) -> list[LLMInteraction]:
        return [
            i for i in self._items
            if i.envelope.trace_id == trace_id
        ]

    async def recent(
        self,
        *,
        limit: int = 50,
    ) -> list[LLMInteraction]:
        return list(reversed(self._items[-limit:]))

    # -- internals ----------------------------------------------------------

    def _load(self) -> None:
        """Load existing interactions from JSONL file."""
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
                        item = LLMInteraction.model_validate(data)
                        self._items.append(item)
                        self._by_envelope_id[
                            item.envelope.envelope_id
                        ] = item
                        count += 1
                    except Exception:
                        logger.warning(
                            "Skipping malformed LLM interaction entry",
                        )
        except Exception:
            logger.exception(
                "Failed to load LLM interactions from %s",
                self._path,
            )

        if count:
            logger.info(
                "Loaded %d LLM interactions from %s", count, self._path,
            )

    def _persist(self, interaction: LLMInteraction) -> None:
        """Append a single interaction to the JSONL file."""
        try:
            safe_append_line(self._path, interaction.model_dump_json())
        except Exception:
            logger.exception(
                "Failed to persist LLM interaction to %s", self._path,
            )

    # -- helpers for tests --------------------------------------------------

    @property
    def items(self) -> list[LLMInteraction]:
        return self._items
