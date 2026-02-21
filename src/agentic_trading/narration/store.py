"""NarrationStore — ring-buffer storage for narration items.

Keeps the last N narration items in memory with optional JSONL persistence.
Both the text stream and the avatar channel read from the same store,
ensuring consistency.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

from agentic_trading.core.file_io import safe_append_line

from .schema import DecisionExplanation, NarrationItem

logger = logging.getLogger(__name__)


class NarrationStore:
    """In-memory ring buffer for narration items.

    Parameters
    ----------
    max_items:
        Maximum number of narration items retained.
    persistence_path:
        Optional path to a JSONL file for durable storage.  When set,
        every ``add()`` call appends the item to the file and existing
        items are loaded on init.
    """

    def __init__(
        self,
        max_items: int = 200,
        persistence_path: str | Path | None = None,
    ) -> None:
        self._max_items = max_items
        self._items: deque[NarrationItem] = deque(maxlen=max_items)
        self._by_id: dict[str, NarrationItem] = {}
        # Keep the original DecisionExplanation alongside each item
        # so the avatar briefing can rebuild presenter scripts from source data
        self._explanations: dict[str, DecisionExplanation] = {}
        self._persistence_path: Path | None = (
            Path(persistence_path) if persistence_path is not None else None
        )
        if self._persistence_path is not None:
            self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load existing items from JSONL file."""
        if self._persistence_path is None or not self._persistence_path.exists():
            return

        count = 0
        try:
            with open(self._persistence_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        item = NarrationItem.model_validate(data)
                        self._items.append(item)
                        self._by_id[item.script_id] = item
                        count += 1
                    except Exception:
                        logger.warning("Skipping malformed narration entry")
        except Exception:
            logger.exception(
                "Failed to load narration store from %s",
                self._persistence_path,
            )

        if count > 0:
            logger.info(
                "Loaded %d narration items from %s",
                count,
                self._persistence_path,
            )

    def _persist(self, item: NarrationItem) -> None:
        """Append a single item to the JSONL file."""
        if self._persistence_path is None:
            return
        try:
            safe_append_line(self._persistence_path, item.model_dump_json())
        except Exception:
            logger.exception(
                "Failed to persist narration item to %s",
                self._persistence_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        item: NarrationItem,
        explanation: DecisionExplanation | None = None,
    ) -> None:
        """Add a narration item to the store.

        Parameters
        ----------
        item:
            The narration item to store.
        explanation:
            Optional original DecisionExplanation. When provided, the
            avatar briefing can use the Bloomberg Presenter to generate
            a proper broadcast script instead of reading raw text.
        """
        # Evict oldest if at capacity
        if len(self._items) >= self._max_items and self._items:
            oldest = self._items[0]
            self._by_id.pop(oldest.script_id, None)
            self._explanations.pop(oldest.script_id, None)

        self._items.append(item)
        self._by_id[item.script_id] = item
        if explanation is not None:
            self._explanations[item.script_id] = explanation
        self._persist(item)
        logger.debug("Narration stored: id=%s", item.script_id)

    def get_explanation(self, script_id: str) -> DecisionExplanation | None:
        """Return the original DecisionExplanation for a stored item."""
        return self._explanations.get(script_id)

    def get(self, script_id: str) -> NarrationItem | None:
        """Get a narration item by script_id."""
        return self._by_id.get(script_id)

    def latest(self, limit: int = 50) -> list[NarrationItem]:
        """Return the most recent narration items (newest first)."""
        items = list(self._items)
        items.reverse()
        return items[:limit]

    def latest_one(self) -> NarrationItem | None:
        """Return the single most recent narration item."""
        if self._items:
            return self._items[-1]
        return None

    @property
    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        """Clear all stored items."""
        self._items.clear()
        self._by_id.clear()
        self._explanations.clear()

    def to_json_list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return latest items as JSON-serializable dicts (for HTTP endpoint)."""
        items = self.latest(limit)
        result = []
        for item in items:
            result.append(
                {
                    "script_id": item.script_id,
                    "timestamp": item.timestamp.isoformat(),
                    "script_text": item.script_text,
                    "verbosity": item.verbosity,
                    "action": item.metadata.get("action", ""),
                    "symbol": item.metadata.get("symbol", ""),
                    "regime": item.metadata.get("regime", ""),
                    "decision_ref": item.decision_ref,
                    "playback_url": item.playback_url,
                    "published_text": item.published_text,
                    "published_avatar": item.published_avatar,
                }
            )
        return result
