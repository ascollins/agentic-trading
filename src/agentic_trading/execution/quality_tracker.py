"""Execution quality measurement per fill.

Computes:
  - Slippage: abs(fill_price - signal_price) / signal_price in bps
  - Fill rate: filled_orders / submitted_orders (rolling window)
  - Adverse selection: price move against us within 1min and 5min post-fill
  - Venue latency: fill_timestamp - submit_timestamp

Metrics are emitted to Prometheus and stored per-trade in the journal.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


class FillQuality:
    """Per-fill quality measurement."""

    __slots__ = (
        "symbol",
        "strategy_id",
        "side",
        "fill_price",
        "signal_price",
        "fill_qty",
        "slippage_bps",
        "venue_latency_ms",
        "adverse_selection_1min_bps",
        "adverse_selection_5min_bps",
        "filled_at",
    )

    def __init__(
        self,
        symbol: str,
        strategy_id: str,
        side: str,
        fill_price: Decimal,
        signal_price: Decimal | None,
        fill_qty: Decimal,
        slippage_bps: float,
        venue_latency_ms: float | None,
        filled_at: datetime,
    ) -> None:
        self.symbol = symbol
        self.strategy_id = strategy_id
        self.side = side
        self.fill_price = fill_price
        self.signal_price = signal_price
        self.fill_qty = fill_qty
        self.slippage_bps = slippage_bps
        self.venue_latency_ms = venue_latency_ms
        self.adverse_selection_1min_bps: float | None = None
        self.adverse_selection_5min_bps: float | None = None
        self.filled_at = filled_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for journal/API consumption."""
        return {
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "side": self.side,
            "fill_price": float(self.fill_price),
            "signal_price": float(self.signal_price) if self.signal_price else None,
            "fill_qty": float(self.fill_qty),
            "slippage_bps": self.slippage_bps,
            "venue_latency_ms": self.venue_latency_ms,
            "adverse_selection_1min_bps": self.adverse_selection_1min_bps,
            "adverse_selection_5min_bps": self.adverse_selection_5min_bps,
            "filled_at": self.filled_at.isoformat(),
        }


class ExecutionQualityTracker:
    """Tracks execution quality metrics per fill.

    Parameters
    ----------
    window_size:
        Number of recent fills to keep for rolling metrics.
    """

    def __init__(self, window_size: int = 500) -> None:
        self._window_size = window_size
        self._fills: deque[FillQuality] = deque(maxlen=window_size)
        self._submitted_count: int = 0
        self._filled_count: int = 0

    def record_submission(self) -> None:
        """Called when an order is submitted (for fill rate calculation)."""
        self._submitted_count += 1

    def record_fill(
        self,
        fill_price: Decimal,
        fill_qty: Decimal,
        signal_price: Decimal | None,
        side: str,
        symbol: str,
        strategy_id: str,
        submit_timestamp: datetime | None = None,
        fill_timestamp: datetime | None = None,
    ) -> FillQuality:
        """Record a fill and compute quality metrics.

        Returns FillQuality with computed slippage, venue latency.
        Adverse selection is computed later via update_post_fill_price().
        """
        self._filled_count += 1
        now = datetime.now(timezone.utc)

        # Slippage (bps)
        slippage_bps = 0.0
        if signal_price and signal_price > 0:
            slippage_bps = (
                float(abs(fill_price - signal_price))
                / float(signal_price)
                * 10_000
            )

        # Venue latency
        venue_latency_ms = None
        if submit_timestamp and fill_timestamp:
            venue_latency_ms = (
                fill_timestamp - submit_timestamp
            ).total_seconds() * 1000

        quality = FillQuality(
            symbol=symbol,
            strategy_id=strategy_id,
            side=side,
            fill_price=fill_price,
            signal_price=signal_price,
            fill_qty=fill_qty,
            slippage_bps=round(slippage_bps, 2),
            venue_latency_ms=venue_latency_ms,
            filled_at=fill_timestamp or now,
        )
        self._fills.append(quality)

        # Emit Prometheus metrics
        self._emit_metrics(quality)

        logger.debug(
            "Fill quality: %s %s slip=%.1fbps latency=%s",
            symbol,
            side,
            slippage_bps,
            f"{venue_latency_ms:.0f}ms" if venue_latency_ms else "N/A",
        )

        return quality

    def update_post_fill_price(
        self,
        symbol: str,
        filled_at: datetime,
        price_1min: float | None = None,
        price_5min: float | None = None,
    ) -> None:
        """Update adverse selection metrics for a recent fill.

        Called when post-fill candle data becomes available.
        """
        for fq in reversed(self._fills):
            if fq.symbol == symbol and fq.filled_at == filled_at:
                if price_1min is not None and fq.fill_price:
                    direction = 1.0 if fq.side == "buy" else -1.0
                    fq.adverse_selection_1min_bps = round(
                        (price_1min - float(fq.fill_price))
                        * direction
                        / float(fq.fill_price)
                        * -10_000,  # Negative = adverse
                        2,
                    )
                if price_5min is not None and fq.fill_price:
                    direction = 1.0 if fq.side == "buy" else -1.0
                    fq.adverse_selection_5min_bps = round(
                        (price_5min - float(fq.fill_price))
                        * direction
                        / float(fq.fill_price)
                        * -10_000,
                        2,
                    )
                break

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    @property
    def fill_rate(self) -> float:
        """Rolling fill rate: filled / submitted."""
        if self._submitted_count == 0:
            return 1.0
        return self._filled_count / self._submitted_count

    @property
    def avg_slippage_bps(self) -> float:
        """Average slippage in bps across recent fills."""
        if not self._fills:
            return 0.0
        return sum(f.slippage_bps for f in self._fills) / len(self._fills)

    @property
    def total_fills(self) -> int:
        """Total number of fills recorded."""
        return self._filled_count

    @property
    def total_submissions(self) -> int:
        """Total number of order submissions recorded."""
        return self._submitted_count

    def get_strategy_metrics(self, strategy_id: str) -> dict[str, float]:
        """Get per-strategy execution quality metrics."""
        fills = [f for f in self._fills if f.strategy_id == strategy_id]
        if not fills:
            return {"slippage_bps": 0.0, "fill_count": 0}
        return {
            "slippage_bps": sum(f.slippage_bps for f in fills) / len(fills),
            "fill_count": len(fills),
            "avg_venue_latency_ms": (
                sum(f.venue_latency_ms or 0 for f in fills) / len(fills)
            ),
        }

    def get_recent_fills(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent fills as serializable dicts."""
        fills = list(self._fills)[-limit:]
        return [f.to_dict() for f in reversed(fills)]

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate summary for UI/API."""
        return {
            "fill_rate": round(self.fill_rate, 4),
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
            "total_fills": self._filled_count,
            "total_submissions": self._submitted_count,
            "window_size": self._window_size,
            "fills_in_window": len(self._fills),
        }

    # ------------------------------------------------------------------
    # Prometheus emission
    # ------------------------------------------------------------------

    def _emit_metrics(self, quality: FillQuality) -> None:
        try:
            from agentic_trading.observability.metrics import (
                record_execution_slippage,
            )

            record_execution_slippage(
                quality.symbol,
                quality.strategy_id,
                quality.slippage_bps,
            )
        except Exception:
            pass
