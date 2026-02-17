"""Signal Agent — canonical event-driven wrapper for the signal layer.

Consumes ``FeatureVector`` events (via the legacy topic bus) and emits
canonical ``SignalCreated`` and ``DecisionProposed`` domain events onto
the domain bus.  Legacy ``Signal`` / ``OrderIntent`` events continue to
flow on the topic bus for backward compatibility with ``main.py``
wiring.

Write ownership
~~~~~~~~~~~~~~~
This agent is the *sole* producer of:

*  ``SignalCreated``  (domain event)
*  ``DecisionProposed``  (domain event)

It does **not** produce ``OrderPlanned``, ``OrderSubmitted``, or any
execution-layer events — those belong to downstream agents.

Usage::

    agent = SignalAgent(
        strategies=strategies,
        feature_engine=feature_engine,
        event_bus=legacy_event_bus,
        domain_bus=domain_event_bus,  # optional
    )
    await agent.start()
"""

from __future__ import annotations

import logging
import time as _time_mod
from decimal import Decimal
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentCapabilities, FeatureVector, Signal
from agentic_trading.core.interfaces import IEventBus, TradingContext
from agentic_trading.domain.events import DecisionProposed, SignalCreated
from agentic_trading.signal.runner import StrategyRunner, alias_features

logger = logging.getLogger(__name__)

#: Source identifier for write-ownership enforcement.
_SOURCE = "signal"


class SignalAgent(BaseAgent):
    """Event-driven agent that runs strategies and emits signal domain events.

    Orchestrates:
    - ``StrategyRunner`` — dispatches feature vectors to strategies.
    - Portfolio sizing (via ``SignalManager``) — converts signals into
      sized ``DecisionProposed`` events.

    Parameters
    ----------
    strategies:
        Strategy instances to dispatch feature vectors to.
    feature_engine:
        Provides candle buffers via ``get_buffer(symbol, tf)``.
    event_bus:
        Legacy topic-routed event bus (``IEventBus``).
    domain_bus:
        Canonical type-routed domain bus (``INewEventBus``).  When
        ``None``, domain events are still created but not published.
    signal_manager:
        Optional ``SignalManager`` for sizing / intent generation.
        When provided, the agent also emits ``DecisionProposed``
        for every signal that produces order intents.
    ctx:
        ``TradingContext`` — required for strategy dispatch and sizing.
    on_signal:
        Optional callback ``(Signal, float) -> None`` invoked after
        each signal, for metrics/narration wiring in main.py.
    """

    def __init__(
        self,
        *,
        strategies: list[Any],
        feature_engine: Any,
        event_bus: IEventBus,
        domain_bus: Any | None = None,
        signal_manager: Any | None = None,
        ctx: TradingContext | None = None,
        on_signal: Any | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, interval=0)
        self._strategies = list(strategies)
        self._feature_engine = feature_engine
        self._event_bus = event_bus
        self._domain_bus = domain_bus
        self._signal_manager = signal_manager
        self._ctx = ctx
        self._on_signal = on_signal

        # Internal runner — wired in _on_start
        self._runner: StrategyRunner | None = None

        # Counters
        self._signal_created_count = 0
        self._decision_proposed_count = 0

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.STRATEGY

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["feature.vector"],
            publishes_to=["strategy.signal"],
            description=(
                "Runs strategies against feature vectors, emits "
                "SignalCreated and DecisionProposed domain events"
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        self._runner = StrategyRunner(
            strategies=self._strategies,
            feature_engine=self._feature_engine,
            event_bus=self._event_bus,
            on_signal=self._on_signal,
        )

        # Subscribe to feature.vector and dispatch through our handler
        async def _on_feature_vector(event: Any) -> None:
            if not isinstance(event, FeatureVector):
                return
            await self._dispatch(event)

        await self._event_bus.subscribe(
            "feature.vector",
            "signal_agent",
            _on_feature_vector,
        )

        logger.info(
            "SignalAgent started with %d strategies (domain_bus=%s)",
            len(self._strategies),
            self._domain_bus is not None,
        )

    async def _on_stop(self) -> None:
        logger.info(
            "SignalAgent stopped (signals=%d, decisions=%d)",
            self._signal_created_count,
            self._decision_proposed_count,
        )

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, event: FeatureVector) -> None:
        """Run strategies against *event* and emit domain events."""
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
            sig = strategy.on_candle(self._ctx, latest_candle, patched_fv)
            if sig is None:
                continue

            elapsed = _time_mod.monotonic() - t0

            # 1. Publish legacy Signal on topic bus (backward compat)
            logger.info(
                "Signal: %s %s conf=%.2f | %s",
                sig.direction.value,
                sig.symbol,
                sig.confidence,
                sig.rationale,
            )
            await self._event_bus.publish("strategy.signal", sig)

            # 2. Emit canonical SignalCreated domain event
            signal_created = _to_signal_created(sig)
            self._signal_created_count += 1
            await self._publish_domain(signal_created)

            # 3. Fire callback (for metrics / narration wiring)
            if self._on_signal is not None:
                self._on_signal(sig, elapsed)

            # 4. If signal_manager is available, generate DecisionProposed
            if self._signal_manager is not None:
                await self._propose_decision(sig, signal_created)

    async def _propose_decision(
        self,
        sig: Signal,
        signal_created: SignalCreated,
    ) -> None:
        """Convert a signal into a ``DecisionProposed`` domain event.

        Uses the portfolio manager inside SignalManager to size the
        position and generate an intent-like proposal.  The proposal is
        emitted as a domain event for the PolicyGateAgent to consume.
        """
        pm = self._signal_manager.portfolio_manager
        if pm is None:
            return

        pm.on_signal(sig)

        if self._ctx is None:
            return

        capital = 100_000.0  # Default; overridable via set_capital()
        if hasattr(self, "_capital"):
            capital = self._capital

        targets = pm.generate_targets(self._ctx, capital)
        if not targets:
            return

        from agentic_trading.signal.portfolio.intent_converter import (
            build_order_intents,
        )
        from agentic_trading.core.enums import Exchange

        exchange = Exchange.BYBIT
        intents = build_order_intents(
            targets, exchange, self._ctx.clock.now(),
        )

        for intent in intents:
            decision = DecisionProposed(
                source=_SOURCE,
                correlation_id=signal_created.correlation_id,
                causation_id=signal_created.event_id,
                strategy_id=intent.strategy_id,
                symbol=intent.symbol,
                side=intent.side.value,
                qty=intent.qty,
                order_type=intent.order_type.value,
                dedupe_key=intent.dedupe_key,
                signal_event_id=signal_created.event_id,
            )
            self._decision_proposed_count += 1
            await self._publish_domain(decision)

    # ------------------------------------------------------------------
    # Domain bus helper
    # ------------------------------------------------------------------

    async def _publish_domain(self, event: Any) -> None:
        """Publish a domain event if the domain bus is available."""
        if self._domain_bus is not None:
            await self._domain_bus.publish(event)
        else:
            logger.debug(
                "Domain event created (no bus): %s", type(event).__name__,
            )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_capital(self, capital: float) -> None:
        """Update the capital used for DecisionProposed sizing."""
        self._capital = capital

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def signal_created_count(self) -> int:
        """Total ``SignalCreated`` events emitted since start."""
        return self._signal_created_count

    @property
    def decision_proposed_count(self) -> int:
        """Total ``DecisionProposed`` events emitted since start."""
        return self._decision_proposed_count

    @property
    def strategies(self) -> list[Any]:
        """Registered strategies (read-only copy)."""
        return list(self._strategies)

    @property
    def runner(self) -> StrategyRunner | None:
        """The internal StrategyRunner (available after start)."""
        return self._runner


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _to_signal_created(sig: Signal) -> SignalCreated:
    """Map a legacy ``Signal`` (Pydantic) to a canonical ``SignalCreated``."""
    return SignalCreated(
        source=_SOURCE,
        correlation_id=sig.trace_id,
        strategy_id=sig.strategy_id,
        symbol=sig.symbol,
        direction=sig.direction.value,
        confidence=sig.confidence,
        rationale=sig.rationale,
        take_profit=sig.take_profit,
        stop_loss=sig.stop_loss,
        trailing_stop=sig.trailing_stop,
        features_used=tuple(
            (k, float(v)) for k, v in sig.features_used.items()
            if v is not None
        ),
        timeframe=sig.timeframe.value if hasattr(sig.timeframe, "value") else str(sig.timeframe),
    )
