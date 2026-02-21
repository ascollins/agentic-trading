"""Execution quality measurement (spec §5.2).

Tracks per-order and daily aggregate execution metrics:
- Slippage vs benchmark (mid price at intent time)
- Participation rate (fill volume / market volume in window)
- Latency (intent → submission → ack → fill)
- Opportunity cost (intended price vs last price if unfilled)

Fed by :class:`ExecutionEngine` on each fill.  Read by
:class:`DailyEffectivenessScorecard` for the execution quality score.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class OrderMetrics:
    """Metrics for a single order lifecycle."""

    dedupe_key: str
    symbol: str
    strategy_id: str = ""

    # Prices
    reference_price: float = 0.0  # Mid/mark at intent time
    fill_price: float = 0.0
    intended_price: float = 0.0  # Limit price or reference

    # Slippage
    slippage_bps: float = 0.0

    # Timestamps (monotonic seconds)
    intent_time: float = 0.0
    submission_time: float = 0.0
    ack_time: float = 0.0
    fill_time: float = 0.0

    # Latencies (milliseconds)
    intent_to_submission_ms: float = 0.0
    submission_to_ack_ms: float = 0.0
    ack_to_fill_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Volume
    fill_qty: float = 0.0
    market_volume: float = 0.0  # Market volume during execution window
    participation_rate: float = 0.0

    # Outcome
    filled: bool = False


class ExecutionQualityTracker:
    """Tracks execution quality metrics per order and in aggregate.

    Thread-safe.  Maintains a sliding window of the last N orders
    for aggregate statistics.

    Parameters
    ----------
    window_size:
        Number of recent orders to retain for aggregate stats.
    target_slippage_bps:
        Target slippage for the execution quality score.
    target_latency_ms:
        Target latency (intent → fill) for the latency score.
    target_participation:
        Target participation rate for the participation score.
    """

    def __init__(
        self,
        window_size: int = 500,
        target_slippage_bps: float = 10.0,
        target_latency_ms: float = 1000.0,
        target_participation: float = 0.05,
    ) -> None:
        self._lock = threading.Lock()
        self._orders: deque[OrderMetrics] = deque(maxlen=window_size)
        self._pending: dict[str, OrderMetrics] = {}  # dedupe_key → metrics

        self.target_slippage_bps = target_slippage_bps
        self.target_latency_ms = target_latency_ms
        self.target_participation = target_participation

        # Counters
        self._total_orders: int = 0
        self._total_fills: int = 0

    # ------------------------------------------------------------------
    # Recording events
    # ------------------------------------------------------------------

    def record_intent(
        self,
        dedupe_key: str,
        symbol: str,
        strategy_id: str = "",
        reference_price: float = 0.0,
        intended_price: float = 0.0,
    ) -> None:
        """Record when an intent is created (start of lifecycle)."""
        now = time.monotonic()
        metrics = OrderMetrics(
            dedupe_key=dedupe_key,
            symbol=symbol,
            strategy_id=strategy_id,
            reference_price=reference_price,
            intended_price=intended_price or reference_price,
            intent_time=now,
        )
        with self._lock:
            self._pending[dedupe_key] = metrics
            self._total_orders += 1

    def record_submission(self, dedupe_key: str) -> None:
        """Record when an order is submitted to the venue."""
        now = time.monotonic()
        with self._lock:
            m = self._pending.get(dedupe_key)
            if m is not None:
                m.submission_time = now
                m.intent_to_submission_ms = (now - m.intent_time) * 1000

    def record_ack(self, dedupe_key: str) -> None:
        """Record when the venue acknowledges the order."""
        now = time.monotonic()
        with self._lock:
            m = self._pending.get(dedupe_key)
            if m is not None:
                m.ack_time = now
                if m.submission_time > 0:
                    m.submission_to_ack_ms = (now - m.submission_time) * 1000

    def record_fill(
        self,
        dedupe_key: str,
        fill_price: float,
        fill_qty: float,
        market_volume: float = 0.0,
    ) -> OrderMetrics | None:
        """Record when a fill arrives (end of lifecycle).

        Returns the completed :class:`OrderMetrics`, or ``None`` if
        the dedupe_key was not tracked.
        """
        now = time.monotonic()
        with self._lock:
            m = self._pending.pop(dedupe_key, None)
            if m is None:
                return None

            m.fill_time = now
            m.fill_price = fill_price
            m.fill_qty = fill_qty
            m.filled = True

            # Latencies
            if m.ack_time > 0:
                m.ack_to_fill_ms = (now - m.ack_time) * 1000
            if m.intent_time > 0:
                m.total_latency_ms = (now - m.intent_time) * 1000

            # Slippage vs reference
            if m.reference_price > 0:
                m.slippage_bps = (
                    abs(fill_price - m.reference_price) / m.reference_price * 10_000
                )

            # Participation rate
            m.market_volume = market_volume
            if market_volume > 0:
                m.participation_rate = fill_qty / market_volume

            self._orders.append(m)
            self._total_fills += 1

            return m

    def record_unfilled(self, dedupe_key: str) -> None:
        """Remove a pending order that was never filled (cancel/reject)."""
        with self._lock:
            self._pending.pop(dedupe_key, None)

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    @property
    def avg_slippage_bps(self) -> float:
        """Average slippage in basis points over the sliding window."""
        with self._lock:
            filled = [m for m in self._orders if m.filled]
            if not filled:
                return 0.0
            return sum(m.slippage_bps for m in filled) / len(filled)

    @property
    def avg_latency_ms(self) -> float:
        """Average total latency (intent → fill) in milliseconds."""
        with self._lock:
            filled = [m for m in self._orders if m.filled and m.total_latency_ms > 0]
            if not filled:
                return 0.0
            return sum(m.total_latency_ms for m in filled) / len(filled)

    @property
    def avg_participation_rate(self) -> float:
        """Average participation rate over filled orders with volume data."""
        with self._lock:
            with_vol = [
                m for m in self._orders
                if m.filled and m.market_volume > 0
            ]
            if not with_vol:
                return 0.0
            return sum(m.participation_rate for m in with_vol) / len(with_vol)

    @property
    def fill_rate(self) -> float:
        """Fraction of tracked orders that were filled."""
        if self._total_orders == 0:
            return 1.0
        return self._total_fills / self._total_orders

    @property
    def pct_orders_within_target_latency(self) -> float:
        """Fraction of filled orders whose total latency is within target."""
        with self._lock:
            filled = [m for m in self._orders if m.filled and m.total_latency_ms > 0]
            if not filled:
                return 1.0
            within = sum(
                1 for m in filled
                if m.total_latency_ms <= self.target_latency_ms
            )
            return within / len(filled)

    # ------------------------------------------------------------------
    # Score components (0-10 scale, per design spec §10)
    # ------------------------------------------------------------------

    @property
    def slippage_score(self) -> float:
        """Score: 10 - 10 * min(1, avg_slippage / target_slippage)."""
        if self.target_slippage_bps <= 0:
            return 10.0
        ratio = min(1.0, self.avg_slippage_bps / self.target_slippage_bps)
        return 10.0 * (1.0 - ratio)

    @property
    def participation_score(self) -> float:
        """Score: 10 * (1 - |actual - target| / target).

        Returns 10.0 when no participation data is available (conservative).
        """
        if self.target_participation <= 0:
            return 10.0
        avg = self.avg_participation_rate
        if avg == 0.0:
            return 10.0  # No data — neutral
        deviation = abs(avg - self.target_participation) / self.target_participation
        return max(0.0, 10.0 * (1.0 - deviation))

    @property
    def latency_score(self) -> float:
        """Score: 10 * pct_orders_within_target_latency."""
        return 10.0 * self.pct_orders_within_target_latency

    @property
    def composite_execution_score(self) -> float:
        """Combined execution quality score (0-10).

        Per spec §10: (slippage_score + participation_score + latency_score) / 3
        """
        return (self.slippage_score + self.participation_score + self.latency_score) / 3

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def total_orders(self) -> int:
        return self._total_orders

    @property
    def total_fills(self) -> int:
        return self._total_fills

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def window_size(self) -> int:
        with self._lock:
            return len(self._orders)

    def get_recent_metrics(self, n: int = 10) -> list[OrderMetrics]:
        """Return the N most recent completed order metrics."""
        with self._lock:
            items = list(self._orders)
            return items[-n:]
