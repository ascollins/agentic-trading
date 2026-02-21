"""Feature snapshot persistence for decision auditability.

At the moment a TradeIntent is created, a FeatureSnapshot is persisted
containing the exact feature vector, signal, and model version that
contributed to the trading decision.  This enables deterministic replay
and regulatory audit of any historical trade.

Snapshots are stored in an append-only JSONL file (paper/live) or an
in-memory ring buffer (backtest).  Each snapshot's ``snapshot_id`` is
referenced in the corresponding ``AuditEntry`` or ``DecisionAudit`` so
that the full decision chain can be reconstructed.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.ids import new_id, payload_hash, utc_now

logger = logging.getLogger(__name__)


class FeatureSnapshot(BaseModel):
    """Immutable record of the features and signal used for a trading decision.

    Persisted at intent-creation time so that any trade can be replayed
    from exactly the same inputs.
    """

    snapshot_id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # Instrument context
    symbol: str
    timeframe: str = ""

    # Feature provenance
    feature_vector: dict[str, float | None] = Field(default_factory=dict)
    feature_version: str = ""  # Hash of indicator config / model params

    # Signal provenance
    strategy_id: str = ""
    model_id: str = ""
    model_version: str = ""
    signal_direction: str = ""  # "long" / "short" / "flat"
    signal_confidence: float = 0.0
    signal_rationale: str = ""

    # Linkage
    trace_id: str = ""
    dedupe_key: str = ""  # Links to the OrderIntent

    # Integrity
    content_hash: str = ""

    def compute_hash(self) -> str:
        """Compute a deterministic hash of the snapshot contents."""
        hashable = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "feature_vector": self.feature_vector,
            "feature_version": self.feature_version,
            "strategy_id": self.strategy_id,
            "model_id": self.model_id,
            "signal_direction": self.signal_direction,
            "signal_confidence": self.signal_confidence,
        }
        return payload_hash(hashable)


class FeatureSnapshotStore:
    """Thread-safe, append-only store for feature snapshots.

    Parameters
    ----------
    persistence_path:
        Path to a JSONL file for durable persistence.
        If ``None``, snapshots are stored in-memory only (backtest mode).
    max_memory:
        Maximum number of snapshots retained in the in-memory buffer.
    """

    def __init__(
        self,
        persistence_path: str | Path | None = None,
        max_memory: int = 10_000,
    ) -> None:
        self._lock = threading.Lock()
        self._buffer: deque[FeatureSnapshot] = deque(maxlen=max_memory)
        self._persistence_path: Path | None = None
        self._index: dict[str, FeatureSnapshot] = {}  # snapshot_id -> snapshot

        if persistence_path is not None:
            self._persistence_path = Path(persistence_path)
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    def store(self, snapshot: FeatureSnapshot) -> str:
        """Persist a snapshot and return its snapshot_id.

        Computes the content hash before storing if not already set.
        """
        if not snapshot.content_hash:
            snapshot = snapshot.model_copy(
                update={"content_hash": snapshot.compute_hash()}
            )

        with self._lock:
            # Evict from index if deque is at capacity
            if len(self._buffer) == self._buffer.maxlen:
                evicted = self._buffer[0]
                self._index.pop(evicted.snapshot_id, None)
            self._buffer.append(snapshot)
            self._index[snapshot.snapshot_id] = snapshot

            if self._persistence_path is not None:
                try:
                    with open(self._persistence_path, "a") as f:
                        f.write(snapshot.model_dump_json() + "\n")
                except Exception:
                    logger.exception(
                        "Failed to persist feature snapshot %s",
                        snapshot.snapshot_id,
                    )

        return snapshot.snapshot_id

    def get(self, snapshot_id: str) -> FeatureSnapshot | None:
        """Retrieve a snapshot by ID."""
        with self._lock:
            return self._index.get(snapshot_id)

    def get_by_dedupe_key(self, dedupe_key: str) -> FeatureSnapshot | None:
        """Retrieve the most recent snapshot for a given dedupe_key."""
        with self._lock:
            for snapshot in reversed(self._buffer):
                if snapshot.dedupe_key == dedupe_key:
                    return snapshot
        return None

    def get_by_trace(self, trace_id: str) -> list[FeatureSnapshot]:
        """Retrieve all snapshots associated with a trace_id."""
        with self._lock:
            return [s for s in self._buffer if s.trace_id == trace_id]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def _load_existing(self) -> None:
        """Load snapshots from the JSONL file on startup."""
        if self._persistence_path is None or not self._persistence_path.exists():
            return
        loaded = 0
        try:
            with open(self._persistence_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        snapshot = FeatureSnapshot(**data)
                        self._buffer.append(snapshot)
                        self._index[snapshot.snapshot_id] = snapshot
                        loaded += 1
                    except Exception:
                        logger.debug("Skipping malformed snapshot line", exc_info=True)
        except Exception:
            logger.exception("Failed to load feature snapshots from %s", self._persistence_path)
        if loaded:
            logger.info("Loaded %d feature snapshots from %s", loaded, self._persistence_path)
