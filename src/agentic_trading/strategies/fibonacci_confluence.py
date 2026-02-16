"""Fibonacci Confluence Zone System (CMT Strategy 13).

Identifies zones where 3+ Fibonacci levels from different swing
moves cluster within a 0.5% band. Entry requires candlestick
confirmation at the confluence zone.

Signal logic:
  LONG:  Price enters a bullish Fibonacci confluence zone (support)
         + candlestick confirmation (bullish close above zone)
         + RSI < 40 (not overbought)
  SHORT: Price enters a bearish Fibonacci confluence zone (resistance)
         + candlestick confirmation (bearish close below zone)
         + RSI > 60 (not oversold)
  EXIT:  Fibonacci extension targets or ATR-based stop
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import numpy as np

from agentic_trading.core.enums import SignalDirection
from agentic_trading.core.events import FeatureVector, RegimeState, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle

from .base import BaseStrategy
from .registry import register_strategy

logger = logging.getLogger(__name__)

# Standard Fibonacci retracement ratios
_FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]


@register_strategy("fibonacci_confluence")
class FibonacciConfluenceStrategy(BaseStrategy):
    """CMT Strategy 13: Fibonacci Confluence Zone System."""

    def __init__(
        self,
        strategy_id: str = "fibonacci_confluence",
        params: dict[str, Any] | None = None,
    ):
        defaults = {
            "swing_lookback": 50,
            "min_swing_pct": 0.03,  # Min 3% move to qualify as a swing
            "confluence_band_pct": 0.005,  # 0.5% clustering tolerance
            "min_confluence_levels": 3,
            "rsi_long_max": 40,
            "rsi_short_min": 60,
            "atr_multiplier": 2.0,
            "atr_target_multiplier": 3.0,
            "min_confidence": 0.4,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)
        self._current_regime: RegimeState | None = None
        self._price_history: dict[str, list[float]] = {}
        self._high_history: dict[str, list[float]] = {}
        self._low_history: dict[str, list[float]] = {}
        self._max_history = 200

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features
        symbol = candle.symbol

        rsi = f.get("rsi")
        atr = f.get("atr")

        # Update price history
        self._update_history(symbol, candle.close, candle.high, candle.low)

        price = candle.close

        # ---- EXIT CHECK (before lookback gate) ----
        current_pos = self._position_direction(symbol)
        if current_pos is not None:
            should_exit = False
            exit_reasons = []

            # Exit on RSI reversal
            if rsi is not None:
                if current_pos == "long" and rsi > 70:
                    should_exit = True
                    exit_reasons.append(f"RSI={rsi:.1f} overbought (target zone)")
                elif current_pos == "short" and rsi < 30:
                    should_exit = True
                    exit_reasons.append(f"RSI={rsi:.1f} oversold (target zone)")

            if should_exit:
                self._record_exit(symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={"rsi": rsi, "close": price} if rsi else {"close": price},
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            return None

        # ---- ENTRY CHECK ----
        prices = self._price_history.get(symbol, [])
        highs = self._high_history.get(symbol, [])
        lows = self._low_history.get(symbol, [])

        lookback = self._get_param("swing_lookback")
        if len(prices) < lookback:
            return None

        if self._on_cooldown(symbol, candle.timestamp):
            return None

        # Find swing points
        swings = self._find_swing_points(
            highs[-lookback:], lows[-lookback:], prices[-lookback:]
        )

        if len(swings) < 2:
            return None

        # Compute Fibonacci levels from multiple swing pairs
        fib_levels = self._compute_all_fib_levels(swings, prices[-lookback:])

        if not fib_levels:
            return None

        # Find confluence zones
        confluence_zones = self._find_confluence_zones(
            fib_levels, price,
            self._get_param("confluence_band_pct"),
            self._get_param("min_confluence_levels"),
        )

        if not confluence_zones:
            return None

        # Find the nearest confluence zone
        best_zone = min(confluence_zones, key=lambda z: abs(z["center"] - price))
        zone_center = best_zone["center"]
        zone_count = best_zone["count"]
        zone_distance_pct = abs(price - zone_center) / price

        # Must be near the zone (within 1.5x the band tolerance)
        band_pct = self._get_param("confluence_band_pct")
        if zone_distance_pct > band_pct * 1.5:
            return None

        # Determine direction based on zone position relative to price
        direction = SignalDirection.FLAT
        rationale_parts = [f"Fibonacci confluence zone at {zone_center:.2f} ({zone_count} levels)"]

        rsi_long_max = self._get_param("rsi_long_max")
        rsi_short_min = self._get_param("rsi_short_min")

        if zone_center <= price:
            # Zone is below = support = bullish
            if rsi is not None and rsi > rsi_long_max:
                return None
            # Candlestick confirmation: bullish close (close > open)
            if candle.close <= candle.open:
                return None
            direction = SignalDirection.LONG
            rationale_parts.append("Support zone + bullish candle close")
            if rsi is not None:
                rationale_parts.append(f"RSI={rsi:.1f} < {rsi_long_max}")
        else:
            # Zone is above = resistance = bearish
            if rsi is not None and rsi < rsi_short_min:
                return None
            # Candlestick confirmation: bearish close (close < open)
            if candle.close >= candle.open:
                return None
            direction = SignalDirection.SHORT
            rationale_parts.append("Resistance zone + bearish candle close")
            if rsi is not None:
                rationale_parts.append(f"RSI={rsi:.1f} > {rsi_short_min}")

        # Confidence: based on confluence count and zone quality
        count_conf = min(1.0, (zone_count - 2) / 4.0)
        proximity_conf = max(0.0, 1.0 - zone_distance_pct / band_pct)
        confidence = min(1.0, count_conf * 0.5 + proximity_conf * 0.3 + 0.2)

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        # Risk constraints
        risk_constraints = {
            "sizing_method": "fixed_fractional",
            "price": price,
            "fib_zone_center": zone_center,
        }
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            target_mult = self._get_param("atr_target_multiplier")
            risk_constraints["stop_distance_atr"] = atr * atr_mult
            risk_constraints["atr"] = atr
            if direction == SignalDirection.LONG:
                risk_constraints["target_price"] = price + atr * target_mult
            else:
                risk_constraints["target_price"] = price - atr * target_mult

        features_used = {
            "close": price,
            "fib_zone_center": zone_center,
            "fib_zone_count": zone_count,
        }
        if rsi is not None:
            features_used["rsi"] = rsi
        if atr is not None:
            features_used["atr"] = atr

        self._record_entry(symbol, direction.value)
        self._record_signal_time(symbol, candle.timestamp)

        # Compute explicit TP/SL prices
        _take_profit: Decimal | None = None
        _stop_loss: Decimal | None = None
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            sl_distance = atr * atr_mult
            target_price = risk_constraints.get("target_price")
            if direction == SignalDirection.LONG:
                _stop_loss = Decimal(str(price - sl_distance))
                _take_profit = Decimal(str(target_price)) if target_price is not None else Decimal(str(price + sl_distance * 2))
            else:
                _stop_loss = Decimal(str(price + sl_distance))
                _take_profit = Decimal(str(target_price)) if target_price is not None else Decimal(str(price - sl_distance * 2))

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            rationale=" | ".join(rationale_parts),
            features_used=features_used,
            timeframe=candle.timeframe,
            risk_constraints=risk_constraints,
            take_profit=_take_profit,
            stop_loss=_stop_loss,
        )

    def _update_history(self, symbol: str, close: float, high: float, low: float) -> None:
        """Maintain rolling price history."""
        for store, val in [
            (self._price_history, close),
            (self._high_history, high),
            (self._low_history, low),
        ]:
            if symbol not in store:
                store[symbol] = []
            store[symbol].append(val)
            if len(store[symbol]) > self._max_history:
                store[symbol] = store[symbol][-self._max_history:]

    def _find_swing_points(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[dict[str, Any]]:
        """Find significant swing highs and lows."""
        swings = []
        min_pct = self._get_param("min_swing_pct")
        n = len(closes)

        # Find swing highs
        for i in range(2, n - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] >= highs[i + 1]:
                swings.append({"type": "high", "price": highs[i], "index": i})

        # Find swing lows
        for i in range(2, n - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] <= lows[i + 1]:
                swings.append({"type": "low", "price": lows[i], "index": i})

        # Filter by minimum swing size
        filtered = []
        for s in swings:
            # Check against nearest opposite swing
            for other in swings:
                if other["type"] != s["type"]:
                    diff = abs(s["price"] - other["price"]) / max(s["price"], other["price"])
                    if diff >= min_pct:
                        if s not in filtered:
                            filtered.append(s)
                        break

        return sorted(filtered, key=lambda x: x["index"])

    def _compute_all_fib_levels(
        self,
        swings: list[dict[str, Any]],
        prices: list[float],
    ) -> list[float]:
        """Compute Fibonacci retracement levels from all valid swing pairs."""
        levels = []
        n_swings = len(swings)

        for i in range(n_swings):
            for j in range(i + 1, min(i + 4, n_swings)):
                s1 = swings[i]
                s2 = swings[j]

                # Need a high-low or low-high pair
                if s1["type"] == s2["type"]:
                    continue

                high_price = s1["price"] if s1["type"] == "high" else s2["price"]
                low_price = s1["price"] if s1["type"] == "low" else s2["price"]

                if high_price <= low_price:
                    continue

                diff = high_price - low_price
                for ratio in _FIB_LEVELS:
                    level = high_price - ratio * diff
                    levels.append(level)

        return levels

    def _find_confluence_zones(
        self,
        fib_levels: list[float],
        current_price: float,
        band_pct: float,
        min_count: int,
    ) -> list[dict[str, Any]]:
        """Find zones where multiple Fibonacci levels cluster."""
        if not fib_levels:
            return []

        sorted_levels = sorted(fib_levels)
        zones = []
        used = set()

        for i, level in enumerate(sorted_levels):
            if i in used:
                continue

            # Find all levels within band_pct of this level
            band = level * band_pct
            cluster = [level]
            cluster_indices = [i]

            for j in range(i + 1, len(sorted_levels)):
                if j in used:
                    continue
                if abs(sorted_levels[j] - level) <= band:
                    cluster.append(sorted_levels[j])
                    cluster_indices.append(j)

            if len(cluster) >= min_count:
                center = float(np.mean(cluster))
                # Only consider zones near the current price (within 5%)
                if abs(center - current_price) / current_price <= 0.05:
                    zones.append({
                        "center": center,
                        "count": len(cluster),
                        "levels": cluster,
                    })
                    used.update(cluster_indices)

        return zones
