"""Strategy/timeframe selection policy.

Determines which strategies and timeframes are active
based on current regime state. Implements hysteresis to
prevent excessive switching (churn).
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.core.enums import RegimeType, Timeframe, VolatilityRegime
from agentic_trading.core.events import RegimeState

logger = logging.getLogger(__name__)


class StrategyPolicy:
    """Policy that maps regime states to strategy/timeframe selections.

    Rules:
    - TREND regime: enable trend_following + breakout, disable mean_reversion
    - RANGE regime: enable mean_reversion, disable trend_following, breakout cautious
    - HIGH vol: prefer shorter timeframes, reduce position sizes
    - LOW vol: prefer longer timeframes, normal sizing
    """

    def __init__(self, strategy_configs: list[dict[str, Any]] | None = None) -> None:
        self._configs = strategy_configs or []
        self._current_policy: dict[str, bool] = {}
        self._timeframe_preferences: dict[str, list[Timeframe]] = {}
        self._sizing_multiplier = 1.0

        # Initialize all strategies as enabled
        for cfg in self._configs:
            sid = cfg.get("strategy_id", "")
            self._current_policy[sid] = cfg.get("enabled", True)

    def update(self, regime: RegimeState) -> dict[str, Any]:
        """Update policy based on regime change.

        Returns dict with:
        - strategy_enabled: {strategy_id: bool}
        - timeframe_preferences: {strategy_id: [Timeframe]}
        - sizing_multiplier: float (1.0 = normal)
        """
        policy = {}

        # Strategy enablement based on regime
        for cfg in self._configs:
            sid = cfg.get("strategy_id", "")
            base_enabled = cfg.get("enabled", True)

            if not base_enabled:
                policy[sid] = False
                continue

            if regime.regime == RegimeType.TREND:
                if sid == "mean_reversion":
                    policy[sid] = False
                else:
                    policy[sid] = True
            elif regime.regime == RegimeType.RANGE:
                if sid == "trend_following":
                    policy[sid] = False
                elif sid == "breakout":
                    policy[sid] = False  # Breakouts fail in range
                else:
                    policy[sid] = True
            else:
                policy[sid] = base_enabled

        # Timeframe preferences based on volatility
        tf_prefs: dict[str, list[Timeframe]] = {}
        for cfg in self._configs:
            sid = cfg.get("strategy_id", "")
            base_tfs = cfg.get("timeframes", [Timeframe.M5, Timeframe.H1])
            configured_tfs = [
                Timeframe(tf) if isinstance(tf, str) else tf for tf in base_tfs
            ]

            if regime.volatility == VolatilityRegime.HIGH:
                # Prefer shorter timeframes in high vol
                tf_prefs[sid] = [
                    tf for tf in configured_tfs if tf.minutes <= 60
                ] or configured_tfs
            elif regime.volatility == VolatilityRegime.LOW:
                # Prefer longer timeframes in low vol
                tf_prefs[sid] = [
                    tf for tf in configured_tfs if tf.minutes >= 15
                ] or configured_tfs
            else:
                tf_prefs[sid] = configured_tfs

        # Sizing adjustment
        if regime.volatility == VolatilityRegime.HIGH:
            sizing_mult = 0.5  # Half size in high vol
        elif regime.confidence < 0.5:
            sizing_mult = 0.7  # Reduced when uncertain
        else:
            sizing_mult = 1.0

        self._current_policy = policy
        self._timeframe_preferences = tf_prefs
        self._sizing_multiplier = sizing_mult

        result = {
            "strategy_enabled": dict(policy),
            "timeframe_preferences": {
                k: [tf.value for tf in v] for k, v in tf_prefs.items()
            },
            "sizing_multiplier": sizing_mult,
            "regime": regime.regime.value,
            "volatility": regime.volatility.value,
        }

        logger.info("Policy updated: %s", result)
        return result

    def is_strategy_enabled(self, strategy_id: str) -> bool:
        return self._current_policy.get(strategy_id, True)

    def get_timeframes(self, strategy_id: str) -> list[Timeframe]:
        return self._timeframe_preferences.get(strategy_id, [Timeframe.M5])

    @property
    def sizing_multiplier(self) -> float:
        return self._sizing_multiplier
