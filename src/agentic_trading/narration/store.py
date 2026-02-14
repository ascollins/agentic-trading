"""NarrationStore — ring-buffer storage for narration items.

Keeps the last N narration items in memory (and optionally in Redis).
Both the text stream and the avatar channel read from the same store,
ensuring consistency.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from .schema import NarrationItem

logger = logging.getLogger(__name__)


class NarrationStore:
    """In-memory ring buffer for narration items.

    Parameters
    ----------
    max_items:
        Maximum number of narration items retained.
    """

    def __init__(self, max_items: int = 200) -> None:
        self._max_items = max_items
        self._items: deque[NarrationItem] = deque(maxlen=max_items)
        self._by_id: dict[str, NarrationItem] = {}

    def add(self, item: NarrationItem) -> None:
        """Add a narration item to the store."""
        # Evict oldest if at capacity
        if len(self._items) >= self._max_items and self._items:
            oldest = self._items[0]
            self._by_id.pop(oldest.script_id, None)

        self._items.append(item)
        self._by_id[item.script_id] = item
        logger.debug("Narration stored: id=%s", item.script_id)

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

    def to_json_list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return latest items as JSON-serializable dicts (for HTTP endpoint)."""
        items = self.latest(limit)
        result = []
        for item in items:
            result.append({
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
            })
        return result
