"""Pipeline Log — persistent storage for pipeline results.

Two implementations following the ``IEventStore`` dual pattern:

- ``InMemoryPipelineLog``: list-backed, for backtest and tests.
- ``PipelineLog``: JSONL file-backed, for paper/live persistence.

Both support ``save()``, ``load()``, and ``query()``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from agentic_trading.core.file_io import safe_append_line

from .pipeline_result import PipelineResult

logger = logging.getLogger(__name__)


class InMemoryPipelineLog:
    """In-memory pipeline log for backtest mode."""

    def __init__(self) -> None:
        self._results: list[PipelineResult] = []
        self._index: dict[str, int] = {}

    def save(self, result: PipelineResult) -> None:
        """Store a pipeline result."""
        idx = len(self._results)
        self._results.append(result)
        self._index[result.pipeline_id] = idx

    def load(self, pipeline_id: str) -> PipelineResult | None:
        """Load a specific pipeline result by ID."""
        idx = self._index.get(pipeline_id)
        if idx is None:
            return None
        return self._results[idx]

    def query(
        self,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
        since: datetime | None = None,
        limit: int = 20,
    ) -> list[PipelineResult]:
        """Query pipeline results with filters."""
        results: list[PipelineResult] = []

        for result in reversed(self._results):
            if symbol is not None and result.trigger_symbol != symbol:
                continue
            if since is not None and result.started_at < since:
                continue
            if strategy is not None:
                # Check if any signal trace mentions the strategy
                has_strategy = any(
                    s.get("strategy_id") == strategy
                    for s in result.signals
                )
                if not has_strategy:
                    continue

            results.append(result)
            if len(results) >= limit:
                break

        return results

    @property
    def count(self) -> int:
        """Number of stored results."""
        return len(self._results)

    def clear(self) -> None:
        """Clear all results."""
        self._results.clear()
        self._index.clear()


class PipelineLog:
    """JSONL file-backed pipeline log for paper/live.

    One line per pipeline result. Supports query by pipeline_id,
    symbol, strategy, and time range.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._inner = InMemoryPipelineLog()
        self._load()

    def _load(self) -> None:
        """Load existing results from JSONL file."""
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
                        result = PipelineResult.model_validate(data)
                        self._inner.save(result)
                        count += 1
                    except Exception:
                        logger.warning("Skipping malformed pipeline result")
        except Exception:
            logger.exception(
                "Failed to load pipeline log from %s", self._path
            )

        if count > 0:
            logger.info(
                "Loaded %d pipeline results from %s", count, self._path
            )

    def save(self, result: PipelineResult) -> None:
        """Store result in memory and append to JSONL file."""
        self._inner.save(result)

        try:
            safe_append_line(self._path, result.model_dump_json())
        except Exception:
            logger.exception(
                "Failed to persist pipeline result to %s", self._path
            )

    def load(self, pipeline_id: str) -> PipelineResult | None:
        """Load a specific pipeline result by ID."""
        return self._inner.load(pipeline_id)

    def query(
        self,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
        since: datetime | None = None,
        limit: int = 20,
    ) -> list[PipelineResult]:
        """Query pipeline results with filters."""
        return self._inner.query(
            symbol=symbol,
            strategy=strategy,
            since=since,
            limit=limit,
        )

    @property
    def count(self) -> int:
        """Number of stored results."""
        return self._inner.count

    def clear(self) -> None:
        """Clear in-memory results. Does not delete the file."""
        self._inner.clear()
