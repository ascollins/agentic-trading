"""Base strategy class.

All strategies inherit from BaseStrategy and implement on_candle().
Strategy code MUST NOT import mode-specific modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from agentic_trading.core.events import FeatureVector, RegimeState, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle


class BaseStrategy(ABC):
    """Abstract base for all trading strategies.

    Subclasses implement on_candle() to produce signals.
    They interact only through TradingContext (never know the mode).
    """

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        self._strategy_id = strategy_id
        self._params = params or {}
        # Position tracking: symbol → direction ("long" or "short")
        self._open_positions: dict[str, str] = {}
        # Signal cooldown: symbol → last signal timestamp
        self._last_signal_time: dict[str, datetime] = {}
        self._signal_cooldown_seconds: int = int(
            self._params.get("signal_cooldown_minutes", 15) * 60
        )

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @abstractmethod
    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        """Process a new candle with computed features.

        Returns a Signal if the strategy wants to trade, None otherwise.
        The Signal must include: direction, confidence, rationale, features_used.
        """
        ...

    def on_regime_change(self, regime: RegimeState) -> None:
        """Called when regime detector reports a state change.

        Override to adjust strategy behavior based on regime.
        Default: store the regime for reference.
        """
        self._current_regime = regime

    def get_parameters(self) -> dict[str, Any]:
        """Return current parameters for audit/logging."""
        return dict(self._params)

    def _get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with optional default."""
        return self._params.get(key, default)

    # ------------------------------------------------------------------
    # Position tracking helpers
    # ------------------------------------------------------------------

    def _has_position(self, symbol: str) -> bool:
        """Check if we currently track an open position for *symbol*."""
        return symbol in self._open_positions

    def _position_direction(self, symbol: str) -> str | None:
        """Return the direction of the current position, or ``None``."""
        return self._open_positions.get(symbol)

    def _record_entry(self, symbol: str, direction: str) -> None:
        """Record that we opened a position."""
        self._open_positions[symbol] = direction

    def _record_exit(self, symbol: str) -> None:
        """Record that we closed a position."""
        self._open_positions.pop(symbol, None)

    # ------------------------------------------------------------------
    # Signal cooldown helpers
    # ------------------------------------------------------------------

    def _on_cooldown(self, symbol: str, timestamp: datetime) -> bool:
        """Return ``True`` if we should suppress a new entry signal."""
        last = self._last_signal_time.get(symbol)
        if last is None:
            return False
        elapsed = (timestamp - last).total_seconds()
        return elapsed < self._signal_cooldown_seconds

    def _record_signal_time(self, symbol: str, timestamp: datetime) -> None:
        """Record the timestamp of the last entry signal."""
        self._last_signal_time[symbol] = timestamp
