"""Funding rate arbitrage strategy.

Monitors the perpetual funding rate and opens positions when the rate
exceeds a threshold, capturing the funding payment.

Signal logic:
  SHORT: funding_rate > threshold  (earn positive funding by being short)
  LONG:  funding_rate < -threshold (earn negative funding by being long)
  Confidence scaled by abs(funding_rate) magnitude.

This is a classic crypto-native delta-neutral strategy. In production
it would pair a perp position with a spot hedge, but the signal logic
is the same: go against the crowd when funding is extreme.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import SignalDirection
from agentic_trading.core.events import FeatureVector, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle

from .base import BaseStrategy
from .registry import register_strategy


@register_strategy("funding_arb")
class FundingArbStrategy(BaseStrategy):
    """Funding rate arbitrage: trade against extreme funding rates."""

    def __init__(self, strategy_id: str = "funding_arb", params: dict[str, Any] | None = None):
        defaults = {
            "funding_threshold": 0.0001,   # 0.01% (1 bps) — typical 8h rate
            "high_funding_threshold": 0.0005,  # 0.05% — very elevated funding
            "position_size_pct": 0.05,      # 5% of portfolio per position
            "min_confidence": 0.3,
            "atr_stop_multiplier": 3.0,     # Wide stop for funding trades
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features

        # Funding rate can be passed via features or market data
        funding_rate = f.get("funding_rate")
        if funding_rate is None:
            return None

        threshold = self._get_param("funding_threshold")
        high_threshold = self._get_param("high_funding_threshold")
        atr = f.get("atr") or f.get("atr_14")

        direction = SignalDirection.FLAT
        rationale_parts = []

        if funding_rate > threshold:
            # Positive funding: longs pay shorts → go short to earn funding
            direction = SignalDirection.SHORT
            rationale_parts.append(f"Funding rate {funding_rate:.4%} > {threshold:.4%}")
            rationale_parts.append("Shorts earn funding")
        elif funding_rate < -threshold:
            # Negative funding: shorts pay longs → go long to earn funding
            direction = SignalDirection.LONG
            rationale_parts.append(f"Funding rate {funding_rate:.4%} < -{threshold:.4%}")
            rationale_parts.append("Longs earn funding")
        else:
            return None

        # Confidence: scaled by how far funding exceeds threshold
        abs_funding = abs(funding_rate)
        if abs_funding >= high_threshold:
            confidence = 0.9  # Very strong funding signal
            rationale_parts.append("HIGH funding")
        elif abs_funding >= threshold * 3:
            confidence = 0.7
            rationale_parts.append("Elevated funding")
        else:
            confidence = 0.4
            rationale_parts.append("Moderate funding")

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        # Risk constraints
        risk_constraints = {
            "sizing_method": "fixed_fractional",
            "position_size_pct": self._get_param("position_size_pct"),
            "funding_rate": funding_rate,
        }
        if atr is not None:
            risk_constraints["atr"] = atr
            risk_constraints["stop_distance_atr"] = atr * self._get_param("atr_stop_multiplier")

        features_used = {"funding_rate": funding_rate}
        if atr is not None:
            features_used["atr"] = atr

        # Compute explicit TP/SL prices (2:1 R:R)
        _take_profit: Decimal | None = None
        _stop_loss: Decimal | None = None
        price = candle.close
        if atr is not None:
            sl_distance = atr * self._get_param("atr_stop_multiplier")
            if direction == SignalDirection.LONG:
                _stop_loss = Decimal(str(price - sl_distance))
                _take_profit = Decimal(str(price + sl_distance * 2))
            else:
                _stop_loss = Decimal(str(price + sl_distance))
                _take_profit = Decimal(str(price - sl_distance * 2))

        return Signal(
            strategy_id=self.strategy_id,
            symbol=candle.symbol,
            direction=direction,
            confidence=round(confidence, 3),
            rationale=" | ".join(rationale_parts),
            features_used=features_used,
            timeframe=candle.timeframe,
            risk_constraints=risk_constraints,
            take_profit=_take_profit,
            stop_loss=_stop_loss,
        )
