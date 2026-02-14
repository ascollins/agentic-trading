"""Regime detector facade.

Combines HMM-based and rule-based detection with hysteresis.
Outputs: trend/range, vol high/low, confidence.
Enforces max switches/day and cooldown between switches.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from agentic_trading.core.enums import (
    LiquidityRegime,
    RegimeType,
    VolatilityRegime,
)
from agentic_trading.core.events import RegimeState

from .hmm_model import HMMRegimeModel
from .rule_based import RuleBasedRegimeDetector

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Facade that combines HMM + rule-based regime detection with hysteresis."""

    def __init__(
        self,
        hysteresis_count: int = 3,
        max_switches_per_day: int = 4,
        cooldown_minutes: int = 60,
        use_hmm: bool = True,
    ) -> None:
        self._hysteresis_count = hysteresis_count
        self._max_switches_per_day = max_switches_per_day
        self._cooldown_minutes = cooldown_minutes
        self._use_hmm = use_hmm

        self._hmm = HMMRegimeModel() if use_hmm else None
        self._rule_based = RuleBasedRegimeDetector()

        # State
        self._current_regime = RegimeType.UNKNOWN
        self._current_vol = VolatilityRegime.UNKNOWN
        self._consecutive_count = 0
        self._pending_regime = RegimeType.UNKNOWN
        self._switches_today: list[datetime] = []
        self._last_switch_time: datetime | None = None

    def update(
        self,
        symbol: str,
        returns: list[float],
        volatilities: list[float],
        adx: float | None = None,
        bb_width: float | None = None,
        volume_ratio: float | None = None,
        timestamp: datetime | None = None,
    ) -> RegimeState:
        """Update regime detection with latest data.

        Args:
            symbol: Instrument symbol.
            returns: Recent price returns (log or simple).
            volatilities: Recent volatility values.
            adx: Current ADX value (optional, for rule-based).
            bb_width: Current Bollinger Band width (optional).
            volume_ratio: Current volume ratio (optional).
            timestamp: Current time (for cooldown/switch tracking).

        Returns:
            RegimeState event.
        """
        now = timestamp or datetime.now(timezone.utc)
        self._prune_switches(now)

        # Get raw regime from HMM or rule-based
        if self._hmm and len(returns) >= 30:
            raw_regime, raw_vol, confidence = self._hmm.predict(
                returns, volatilities
            )
        else:
            raw_regime, raw_vol, confidence = self._rule_based.detect(
                returns=returns,
                adx=adx,
                bb_width=bb_width,
                volume_ratio=volume_ratio,
            )

        # Liquidity regime from volume
        if volume_ratio is not None:
            liquidity = (
                LiquidityRegime.HIGH if volume_ratio > 1.2
                else LiquidityRegime.LOW if volume_ratio < 0.5
                else LiquidityRegime.UNKNOWN
            )
        else:
            liquidity = LiquidityRegime.UNKNOWN

        # Hysteresis: require N consecutive signals before switching
        if raw_regime != self._current_regime:
            if raw_regime == self._pending_regime:
                self._consecutive_count += 1
            else:
                self._pending_regime = raw_regime
                self._consecutive_count = 1

            if self._consecutive_count >= self._hysteresis_count:
                # Check cooldown and max switches
                if self._can_switch(now):
                    self._current_regime = raw_regime
                    self._current_vol = raw_vol
                    self._switches_today.append(now)
                    self._last_switch_time = now
                    self._consecutive_count = 0
                    logger.info(
                        "Regime switch: %s (vol=%s, confidence=%.2f, switches_today=%d)",
                        raw_regime.value,
                        raw_vol.value,
                        confidence,
                        len(self._switches_today),
                    )
        else:
            self._consecutive_count = 0
            self._pending_regime = raw_regime
            # Update vol regime even without regime switch
            self._current_vol = raw_vol

        return RegimeState(
            symbol=symbol,
            regime=self._current_regime,
            volatility=self._current_vol,
            liquidity=liquidity,
            confidence=confidence,
            consecutive_count=self._consecutive_count,
            switches_today=len(self._switches_today),
        )

    def _can_switch(self, now: datetime) -> bool:
        """Check if a regime switch is allowed (cooldown + max switches)."""
        if len(self._switches_today) >= self._max_switches_per_day:
            return False
        if self._last_switch_time is not None:
            elapsed = (now - self._last_switch_time).total_seconds()
            if elapsed < self._cooldown_minutes * 60:
                return False
        return True

    def _prune_switches(self, now: datetime) -> None:
        """Remove switches from previous days."""
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self._switches_today = [
            t for t in self._switches_today if t >= today_start
        ]
