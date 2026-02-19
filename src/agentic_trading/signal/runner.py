"""Strategy runner — dispatches feature vectors to strategies.

``StrategyRunner`` encapsulates the core signal-generation loop:

1.  Receive a ``FeatureVector`` event.
2.  Retrieve the candle buffer from the feature engine.
3.  Alias common indicator keys for backward compatibility.
4.  Call ``strategy.on_candle()`` for each registered strategy.
5.  Publish resulting ``Signal`` events to the event bus.

This module extracts the strategy dispatch logic that previously lived
inside ``main.py._run_live_or_paper.on_feature_vector``.  The full
wiring (portfolio manager, execution, narration) remains in ``main.py``
for now and will be migrated in subsequent PRs.

Usage (in bootstrap)::

    runner = StrategyRunner(
        strategies=strategies,
        feature_engine=feature_engine,
        event_bus=ctx.event_bus,
        clock=ctx.clock,
    )
    await runner.start(ctx)
"""

from __future__ import annotations

import logging
import time as _time_mod
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agentic_trading.core.events import FeatureVector, Signal
from agentic_trading.core.interfaces import IEventBus, TradingContext
from agentic_trading.signal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicator aliasing
# ---------------------------------------------------------------------------

#: Map of canonical indicator names → aliases required by some strategies.
_INDICATOR_ALIASES: dict[str, str] = {
    "adx_14": "adx",
    "atr_14": "atr",
    "rsi_14": "rsi",
    "donchian_upper_20": "donchian_upper",
    "donchian_lower_20": "donchian_lower",
}


def alias_features(features: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *features* with standard short aliases added.

    For example, ``adx_14`` is aliased to ``adx`` if ``adx`` is not
    already present.  This ensures backward-compat with strategies that
    reference the short name.
    """
    out = dict(features)
    for canonical, alias in _INDICATOR_ALIASES.items():
        if canonical in out and alias not in out:
            out[alias] = out[canonical]
    return out


# ---------------------------------------------------------------------------
# Feature-engine protocol (duck-typed to avoid circular import)
# ---------------------------------------------------------------------------

@runtime_checkable
class _FeatureEngineProto(Protocol):
    """Minimal interface the runner needs from the feature engine."""

    def get_buffer(self, symbol: str, timeframe: Any) -> list | None:
        """Return the candle buffer for (symbol, timeframe), or ``None``."""
        ...


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

#: Called after a signal is produced.  Signature:
#:  ``(signal: Signal, elapsed_seconds: float) -> None``
SignalCallback = Callable[[Signal, float], Any]


# ---------------------------------------------------------------------------
# StrategyRunner
# ---------------------------------------------------------------------------

class StrategyRunner:
    """Runs registered strategies against incoming feature vectors.

    Parameters
    ----------
    strategies
        Strategy instances to dispatch feature vectors to.
    feature_engine
        Provides candle buffers via ``get_buffer(symbol, tf)``.
    event_bus
        Used to subscribe to ``feature.vector`` and publish
        ``strategy.signal``.
    on_signal
        Optional callback invoked after each signal is produced.
        Receives ``(signal, elapsed_seconds)``.  Useful for metrics,
        caching, or downstream wiring (portfolio, execution).
    """

    def __init__(
        self,
        *,
        strategies: list[BaseStrategy],
        feature_engine: _FeatureEngineProto,
        event_bus: IEventBus,
        on_signal: SignalCallback | None = None,
    ) -> None:
        self._strategies = list(strategies)
        self._feature_engine = feature_engine
        self._event_bus = event_bus
        self._on_signal = on_signal
        self._signal_count = 0

    # -- Public API --------------------------------------------------------

    async def start(self, ctx: TradingContext) -> None:
        """Subscribe to ``feature.vector`` and begin dispatching.

        Must be called after the event bus is started.
        """

        async def _on_feature_vector(event: Any) -> None:
            if not isinstance(event, FeatureVector):
                return
            await self._dispatch(ctx, event)

        await self._event_bus.subscribe(
            "feature.vector",
            "strategy_runner",
            _on_feature_vector,
        )
        logger.info(
            "StrategyRunner started with %d strategies",
            len(self._strategies),
        )

    @property
    def signal_count(self) -> int:
        """Total signals produced since start."""
        return self._signal_count

    @property
    def strategies(self) -> list[BaseStrategy]:
        """Registered strategies (read-only copy)."""
        return list(self._strategies)

    # -- Internal ----------------------------------------------------------

    async def _dispatch(
        self,
        ctx: TradingContext,
        event: FeatureVector,
    ) -> None:
        """Run all strategies against *event* and publish signals."""
        buffer = self._feature_engine.get_buffer(event.symbol, event.timeframe)
        if not buffer:
            return

        aliased = alias_features(event.features)
        patched_fv = FeatureVector(
            symbol=event.symbol,
            timeframe=event.timeframe,
            features=aliased,
            source_module=event.source_module,
        )

        latest_candle = buffer[-1]

        for strategy in self._strategies:
            t0 = _time_mod.monotonic()
            sig = strategy.on_candle(ctx, latest_candle, patched_fv)
            if sig is not None:
                elapsed = _time_mod.monotonic() - t0
                self._signal_count += 1

                logger.info(
                    "Signal: %s %s conf=%.2f | %s",
                    sig.direction.value,
                    sig.symbol,
                    sig.confidence,
                    sig.rationale,
                )
                await self._event_bus.publish("strategy.signal", sig)

                if self._on_signal is not None:
                    self._on_signal(sig, elapsed)
