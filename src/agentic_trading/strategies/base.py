"""Base strategy class.

All strategies inherit from BaseStrategy and implement on_candle().
Strategy code MUST NOT import mode-specific modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
