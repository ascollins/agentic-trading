"""Supply/Demand Zone Systematic Trading (CMT Strategy 8).

Uses Smart Money Concepts (SMC) zone identification to trade:
  - Demand zones (bullish order blocks) for long entries
  - Supply zones (bearish order blocks) for short entries

Zone quality scoring based on:
  - Freshness (unmitigated zones score higher)
  - Distance from current price (proximity)
  - Volume confirmation at zone creation
  - Structure alignment (BOS/CHoCH confirmation)

Signal logic:
  LONG:  Price enters demand zone + zone is unmitigated + BOS bullish
         + volume confirmation
  SHORT: Price enters supply zone + zone is unmitigated + BOS bearish
         + volume confirmation
  EXIT:  Partial at 1:1 R:R, remainder at opposite zone or ATR target
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import SignalDirection
from agentic_trading.core.events import FeatureVector, RegimeState, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle

from .base import BaseStrategy
from .registry import register_strategy


@register_strategy("supply_demand")
class SupplyDemandStrategy(BaseStrategy):
    """CMT Strategy 8: Supply/Demand Zone Systematic Trading."""

    def __init__(
        self,
        strategy_id: str = "supply_demand",
        params: dict[str, Any] | None = None,
    ):
        defaults = {
            "min_demand_distance": 0.0,    # Min distance to demand zone (0 = at zone)
            "max_demand_distance": 0.02,   # Max 2% from demand zone
            "min_supply_distance": 0.0,
            "max_supply_distance": 0.02,
            "require_bos": True,           # Require Break of Structure confirmation
            "volume_gate": 1.0,            # Min volume ratio
            "atr_multiplier": 1.5,
            "target_rr_ratio": 2.0,        # Risk-reward target
            "min_confidence": 0.4,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)
        self._current_regime: RegimeState | None = None

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features
        symbol = candle.symbol

        # SMC features required
        demand_dist = f.get("smc_nearest_demand_distance")
        supply_dist = f.get("smc_nearest_supply_distance")
        ob_bullish = f.get("smc_ob_unmitigated_bullish", 0)
        ob_bearish = f.get("smc_ob_unmitigated_bearish", 0)
        bos_bullish = f.get("smc_bos_bullish", 0)
        bos_bearish = f.get("smc_bos_bearish", 0)
        choch_bullish = f.get("smc_choch_bullish", 0)
        choch_bearish = f.get("smc_choch_bearish", 0)
        swing_bias = f.get("smc_swing_bias")
        volume_ratio = f.get("volume_ratio", 1.0)
        atr = f.get("atr")

        # If SMC features are not available, cannot trade this strategy
        if demand_dist is None and supply_dist is None:
            return None

        price = candle.close

        # ---- EXIT CHECK ----
        current_pos = self._position_direction(symbol)
        if current_pos is not None:
            should_exit = False
            exit_reasons = []

            # Exit long if price reaches supply zone
            if current_pos == "long" and supply_dist is not None:
                max_dist = self._get_param("max_supply_distance")
                if supply_dist <= max_dist:
                    should_exit = True
                    exit_reasons.append("Price reached supply zone (resistance)")

            # Exit short if price reaches demand zone
            elif current_pos == "short" and demand_dist is not None:
                max_dist = self._get_param("max_demand_distance")
                if demand_dist <= max_dist:
                    should_exit = True
                    exit_reasons.append("Price reached demand zone (support)")

            # Exit on structure break against position
            if current_pos == "long" and bos_bearish > 0:
                should_exit = True
                exit_reasons.append("Bearish Break of Structure")
            elif current_pos == "short" and bos_bullish > 0:
                should_exit = True
                exit_reasons.append("Bullish Break of Structure")

            if should_exit:
                self._record_exit(symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={"close": price},
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            return None

        # ---- ENTRY CHECK ----
        if self._on_cooldown(symbol, candle.timestamp):
            return None

        # Volume gate
        vol_gate = self._get_param("volume_gate")
        if volume_ratio < vol_gate:
            return None

        direction = SignalDirection.FLAT
        rationale_parts = []
        zone_quality = 0.0

        require_bos = self._get_param("require_bos")
        min_demand = self._get_param("min_demand_distance")
        max_demand = self._get_param("max_demand_distance")
        min_supply = self._get_param("min_supply_distance")
        max_supply = self._get_param("max_supply_distance")

        # Check demand zone entry (bullish)
        if (demand_dist is not None
                and min_demand <= demand_dist <= max_demand
                and ob_bullish > 0):
            # BOS/CHoCH confirmation
            if require_bos and bos_bullish <= 0 and choch_bullish <= 0:
                pass  # No structure confirmation
            else:
                direction = SignalDirection.LONG
                zone_quality = self._score_zone(
                    demand_dist, ob_bullish, bos_bullish + choch_bullish, volume_ratio
                )
                rationale_parts.append(f"Demand zone entry (dist={demand_dist:.4f})")
                rationale_parts.append(f"{ob_bullish} unmitigated bullish OBs")
                if bos_bullish > 0:
                    rationale_parts.append("BOS bullish confirmed")
                elif choch_bullish > 0:
                    rationale_parts.append("CHoCH bullish confirmed")

        # Check supply zone entry (bearish)
        if (direction == SignalDirection.FLAT
                and supply_dist is not None
                and min_supply <= supply_dist <= max_supply
                and ob_bearish > 0):
            if require_bos and bos_bearish <= 0 and choch_bearish <= 0:
                pass
            else:
                direction = SignalDirection.SHORT
                zone_quality = self._score_zone(
                    supply_dist, ob_bearish, bos_bearish + choch_bearish, volume_ratio
                )
                rationale_parts.append(f"Supply zone entry (dist={supply_dist:.4f})")
                rationale_parts.append(f"{ob_bearish} unmitigated bearish OBs")
                if bos_bearish > 0:
                    rationale_parts.append("BOS bearish confirmed")
                elif choch_bearish > 0:
                    rationale_parts.append("CHoCH bearish confirmed")

        if direction == SignalDirection.FLAT:
            return None

        # Swing bias alignment bonus
        if swing_bias is not None:
            if direction == SignalDirection.LONG and swing_bias > 0:
                zone_quality = min(1.0, zone_quality + 0.1)
                rationale_parts.append("Swing bias bullish")
            elif direction == SignalDirection.SHORT and swing_bias < 0:
                zone_quality = min(1.0, zone_quality + 0.1)
                rationale_parts.append("Swing bias bearish")

        # Confidence
        confidence = min(1.0, zone_quality * 0.7 + 0.3)

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        rationale_parts.append(f"Volume {volume_ratio:.1f}x avg")

        # Risk constraints
        rr_target = self._get_param("target_rr_ratio")
        risk_constraints = {
            "sizing_method": "fixed_fractional",
            "price": price,
            "risk_reward_target": rr_target,
        }
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            risk_constraints["stop_distance_atr"] = atr * atr_mult
            risk_constraints["atr"] = atr
            if direction == SignalDirection.LONG:
                risk_constraints["target_price"] = price + atr * atr_mult * rr_target
            else:
                risk_constraints["target_price"] = price - atr * atr_mult * rr_target

        features_used = {
            "close": price,
            "volume_ratio": volume_ratio,
            "zone_quality": zone_quality,
        }
        if demand_dist is not None:
            features_used["smc_nearest_demand_distance"] = demand_dist
        if supply_dist is not None:
            features_used["smc_nearest_supply_distance"] = supply_dist
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

    @staticmethod
    def _score_zone(
        distance: float,
        unmitigated_count: float,
        structure_count: float,
        volume_ratio: float,
    ) -> float:
        """Score zone quality from 0 to 1.

        Factors:
          - Proximity (closer = higher score)
          - Freshness (more unmitigated OBs = higher)
          - Structure confirmation (BOS/CHoCH presence)
          - Volume strength
        """
        # Proximity: 0.02 distance = 0, 0 distance = 1
        proximity = max(0.0, 1.0 - distance / 0.02)

        # Freshness: more unmitigated is better
        freshness = min(1.0, unmitigated_count / 3.0)

        # Structure
        structure = min(1.0, structure_count / 2.0)

        # Volume
        vol_score = min(1.0, (volume_ratio - 0.5) / 1.5) if volume_ratio > 0.5 else 0.0

        return proximity * 0.3 + freshness * 0.3 + structure * 0.25 + vol_score * 0.15
