"""Orderbook depth features and institutional wall detection.

Computes features from order book snapshots, essential for:

* **Bid/ask wall detection** -- large resting orders that act as
  support / resistance.
* **Orderbook imbalance** -- bid-heavy vs ask-heavy depth ratio.
* **Spread analysis** -- micro-structure health.
* **Wall persistence** -- how long a wall has been present across
  successive snapshots (institutional conviction).

The engine subscribes to ``OrderBookSnapshot`` events and publishes
depth features into the ``FeatureVector`` pipeline.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from agentic_trading.core.events import BaseEvent, FeatureVector, OrderBookSnapshot
from agentic_trading.core.enums import Timeframe
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)

# Maximum snapshots kept for persistence tracking.
_DEFAULT_HISTORY_SIZE = 100

# How far from mid-price (as fraction) to scan for walls.
_DEFAULT_WALL_SCAN_PCT = 0.05  # 5%


class OrderbookEngine:
    """Computes orderbook depth features from snapshots.

    Usage (event-bus driven)::

        engine = OrderbookEngine(event_bus=bus)
        await engine.start()

    Or direct (backtesting)::

        engine = OrderbookEngine()
        features = engine.compute_orderbook_features(
            symbol="BTC/USDT",
            bids=[[67000, 5.0], [66990, 3.0], ...],
            asks=[[67010, 4.0], [67020, 2.0], ...],
        )
    """

    def __init__(
        self,
        event_bus: IEventBus | None = None,
        wall_scan_pct: float = _DEFAULT_WALL_SCAN_PCT,
        history_size: int = _DEFAULT_HISTORY_SIZE,
        depth_bps_levels: list[int] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._wall_scan_pct = wall_scan_pct
        self._history_size = history_size
        self._depth_bps_levels = depth_bps_levels or [10, 20, 50, 100]

        # Wall persistence tracking: price level -> number of snapshots
        # where a wall was observed at that level.
        # Key: symbol -> dict[rounded_price, appearance_count]
        self._bid_wall_history: dict[str, dict[float, int]] = defaultdict(dict)
        self._ask_wall_history: dict[str, dict[float, int]] = defaultdict(dict)
        self._snapshot_count: dict[str, int] = defaultdict(int)

        # Trade intensity tracking (for MBO microstructure features)
        # Key: symbol -> deque of (timestamp, trade_count) tuples
        self._trade_timestamps: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        # Snapshot timing for inter-arrival tracking
        self._last_snapshot_time: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Event bus lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to orderbook snapshot events."""
        if self._event_bus is None:
            logger.warning(
                "OrderbookEngine started without event bus - "
                "call compute_orderbook_features() directly"
            )
            return

        await self._event_bus.subscribe(
            topic="market.orderbook",
            group="orderbook_engine",
            handler=self._handle_snapshot,
        )
        logger.info("OrderbookEngine subscribed to market.orderbook")

    async def stop(self) -> None:
        """Clean up (currently a no-op)."""
        logger.info("OrderbookEngine stopped")

    async def _handle_snapshot(self, event: BaseEvent) -> None:
        """Handle incoming orderbook snapshot events."""
        if not isinstance(event, OrderBookSnapshot):
            return

        features = self.compute_orderbook_features(
            symbol=event.symbol,
            bids=event.bids,
            asks=event.asks,
        )

        if self._event_bus is not None:
            fv = FeatureVector(
                symbol=event.symbol,
                timeframe=Timeframe.M1,
                features=features,
                source_module="features.orderbook",
            )
            await self._event_bus.publish("feature.vector", fv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_orderbook_features(
        self,
        symbol: str,
        bids: list[list[float]],
        asks: list[list[float]],
    ) -> dict[str, float]:
        """Compute orderbook depth features.

        Args:
            symbol: Trading pair.
            bids: List of ``[price, quantity]`` sorted descending by price.
            asks: List of ``[price, quantity]`` sorted ascending by price.

        Returns:
            Dict of feature name to float value.
        """
        features: dict[str, float] = {}

        if not bids or not asks:
            return features

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2.0

        if mid_price <= 0:
            return features

        # --- Spread ---
        spread = best_ask - best_bid
        features["ob_spread"] = spread
        features["ob_spread_pct"] = (spread / mid_price) * 100.0

        # --- Depth within scan range ---
        scan_high = mid_price * (1.0 + self._wall_scan_pct)
        scan_low = mid_price * (1.0 - self._wall_scan_pct)

        bid_depth = sum(
            qty for price, qty in bids
            if price >= scan_low
        )
        ask_depth = sum(
            qty for price, qty in asks
            if price <= scan_high
        )

        features["ob_bid_depth_5pct"] = bid_depth
        features["ob_ask_depth_5pct"] = ask_depth

        # --- Imbalance ---
        if ask_depth > 0:
            features["ob_imbalance"] = bid_depth / ask_depth
        else:
            features["ob_imbalance"] = 1.0

        # --- Bid wall detection ---
        bid_wall_price, bid_wall_size = self._find_largest_level(
            bids, scan_low, mid_price,
        )
        features["ob_bid_wall_price"] = bid_wall_price
        features["ob_bid_wall_size"] = bid_wall_size

        # --- Ask wall detection ---
        ask_wall_price, ask_wall_size = self._find_largest_level(
            asks, mid_price, scan_high,
        )
        features["ob_ask_wall_price"] = ask_wall_price
        features["ob_ask_wall_size"] = ask_wall_size

        # --- Persistence tracking ---
        self._snapshot_count[symbol] += 1
        snap_count = self._snapshot_count[symbol]

        # Update bid wall history
        if bid_wall_size > 0:
            rounded = self._round_price(bid_wall_price, mid_price)
            self._bid_wall_history[symbol][rounded] = (
                self._bid_wall_history[symbol].get(rounded, 0) + 1
            )

        if ask_wall_size > 0:
            rounded = self._round_price(ask_wall_price, mid_price)
            self._ask_wall_history[symbol][rounded] = (
                self._ask_wall_history[symbol].get(rounded, 0) + 1
            )

        # Compute persistence for current walls
        if bid_wall_size > 0 and snap_count > 1:
            rounded = self._round_price(bid_wall_price, mid_price)
            appearances = self._bid_wall_history[symbol].get(rounded, 1)
            features["ob_bid_wall_persistence"] = min(
                1.0, appearances / snap_count,
            )
        else:
            features["ob_bid_wall_persistence"] = 0.0

        if ask_wall_size > 0 and snap_count > 1:
            rounded = self._round_price(ask_wall_price, mid_price)
            appearances = self._ask_wall_history[symbol].get(rounded, 1)
            features["ob_ask_wall_persistence"] = min(
                1.0, appearances / snap_count,
            )
        else:
            features["ob_ask_wall_persistence"] = 0.0

        # --- Depth at N bps levels ---
        for bps in self._depth_bps_levels:
            bps_frac = bps / 10_000.0
            bid_limit = mid_price * (1.0 - bps_frac)
            ask_limit = mid_price * (1.0 + bps_frac)

            bid_depth_bps = sum(
                qty for price, qty in bids if price >= bid_limit
            )
            ask_depth_bps = sum(
                qty for price, qty in asks if price <= ask_limit
            )
            features[f"ob_bid_depth_{bps}bps"] = bid_depth_bps
            features[f"ob_ask_depth_{bps}bps"] = ask_depth_bps

            # Depth imbalance at this level
            total = bid_depth_bps + ask_depth_bps
            if total > 0:
                features[f"ob_imbalance_{bps}bps"] = (
                    (bid_depth_bps - ask_depth_bps) / total
                )
            else:
                features[f"ob_imbalance_{bps}bps"] = 0.0

        # --- Bid/ask level count within scan range ---
        bid_levels = sum(1 for price, _ in bids if price >= scan_low)
        ask_levels = sum(1 for price, _ in asks if price <= scan_high)
        features["ob_bid_level_count"] = float(bid_levels)
        features["ob_ask_level_count"] = float(ask_levels)

        # --- Weighted mid price (size-weighted) ---
        if bids and asks:
            best_bid_qty = bids[0][1] if bids[0][1] > 0 else 1.0
            best_ask_qty = asks[0][1] if asks[0][1] > 0 else 1.0
            total_qty = best_bid_qty + best_ask_qty
            weighted_mid = (
                best_bid * best_ask_qty + best_ask * best_bid_qty
            ) / total_qty
            features["ob_weighted_mid"] = weighted_mid
            features["ob_microprice_offset_bps"] = (
                (weighted_mid - mid_price) / mid_price * 10_000
            )

        # --- Snapshot inter-arrival time ---
        now = time.monotonic()
        prev_time = self._last_snapshot_time.get(symbol)
        if prev_time is not None:
            interval_ms = (now - prev_time) * 1000.0
            features["ob_snapshot_interval_ms"] = interval_ms
        self._last_snapshot_time[symbol] = now

        # Prune old levels from history (keep at most 50 tracked levels)
        for hist in (
            self._bid_wall_history[symbol],
            self._ask_wall_history[symbol],
        ):
            if len(hist) > 50:
                # Keep highest-count entries
                sorted_levels = sorted(hist.items(), key=lambda x: x[1], reverse=True)
                hist.clear()
                for k, v in sorted_levels[:50]:
                    hist[k] = v

        return features

    def record_trade(
        self,
        symbol: str,
        timestamp: float | None = None,
    ) -> None:
        """Record a trade event for trade intensity computation.

        Call this from the event handler when a TradeEvent arrives for
        a symbol.  The :meth:`compute_orderbook_features` method will
        use the accumulated trade timestamps to compute trade intensity.

        Args:
            symbol: Trading pair.
            timestamp: Monotonic timestamp (defaults to ``time.monotonic()``).
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        self._trade_timestamps[symbol].append(ts)

    def get_trade_intensity(
        self, symbol: str, window_seconds: float = 60.0,
    ) -> dict[str, float]:
        """Compute trade intensity features from recorded trades.

        Args:
            symbol: Trading pair.
            window_seconds: Lookback window for trades/sec calculation.

        Returns:
            Dict with keys:
                ``ob_trade_intensity``   — trades per second over window
                ``ob_trade_count_window`` — total trades in the window
                ``ob_avg_inter_trade_ms`` — average time between trades (ms)
        """
        features: dict[str, float] = {}
        trades = self._trade_timestamps.get(symbol)
        if not trades or len(trades) < 2:
            features["ob_trade_intensity"] = 0.0
            features["ob_trade_count_window"] = float(len(trades) if trades else 0)
            features["ob_avg_inter_trade_ms"] = 0.0
            return features

        now = time.monotonic()
        cutoff = now - window_seconds

        recent = [t for t in trades if t >= cutoff]
        count = len(recent)
        features["ob_trade_count_window"] = float(count)

        if count > 0:
            elapsed = now - cutoff
            features["ob_trade_intensity"] = count / elapsed if elapsed > 0 else 0.0
        else:
            features["ob_trade_intensity"] = 0.0

        if count >= 2:
            intervals = [
                (recent[i] - recent[i - 1]) * 1000.0
                for i in range(1, len(recent))
            ]
            features["ob_avg_inter_trade_ms"] = sum(intervals) / len(intervals)
        else:
            features["ob_avg_inter_trade_ms"] = 0.0

        return features

    def clear(self, symbol: str | None = None) -> None:
        """Reset state.  If *symbol* is ``None``, clears all symbols."""
        if symbol is None:
            self._bid_wall_history.clear()
            self._ask_wall_history.clear()
            self._snapshot_count.clear()
        else:
            self._bid_wall_history.pop(symbol, None)
            self._ask_wall_history.pop(symbol, None)
            self._snapshot_count.pop(symbol, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_largest_level(
        levels: list[list[float]],
        price_low: float,
        price_high: float,
    ) -> tuple[float, float]:
        """Find the price level with the largest quantity within range.

        Returns:
            Tuple of (price, quantity).  ``(0.0, 0.0)`` if no levels
            in range.
        """
        best_price = 0.0
        best_qty = 0.0

        for price, qty in levels:
            if price_low <= price <= price_high:
                if qty > best_qty:
                    best_qty = qty
                    best_price = price

        return best_price, best_qty

    @staticmethod
    def _round_price(price: float, mid_price: float) -> float:
        """Round price to a coarse bucket for persistence tracking.

        Uses 0.1% of mid-price as the bucket width so that small
        price movements don't create separate tracking entries.
        """
        if mid_price <= 0:
            return round(price, 2)
        bucket = mid_price * 0.001  # 0.1%
        if bucket <= 0:
            return round(price, 2)
        return round(price / bucket) * bucket
