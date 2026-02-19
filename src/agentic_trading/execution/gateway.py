"""Execution gateway — unified facade for order execution and risk management.

:class:`ExecutionGateway` composes the execution engine and risk manager
into a single high-level API.  It owns construction of the risk subsystem
and wires it into the execution engine, providing a single entry point for:

- Order submission and lifecycle management
- Pre-trade and post-trade risk checks
- Circuit breaker management
- Kill switch control
- Execution quality tracking

Usage::

    from agentic_trading.execution.gateway import ExecutionGateway

    gateway = ExecutionGateway.from_config(
        adapter=paper_adapter,
        event_bus=event_bus,
        risk_config=risk_config,
    )
    await gateway.start()

    # Submit an order — engine handles risk checks internally
    await gateway.engine.handle_intent(order_intent)

    # Direct risk access
    await gateway.risk_manager.activate_kill_switch("emergency")
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.core.config import RiskConfig
from agentic_trading.core.events import (
    CircuitBreakerEvent,
    KillSwitchEvent,
)
from agentic_trading.core.interfaces import (
    IEventBus,
    IExchangeAdapter,
    PortfolioState,
)
from agentic_trading.core.models import Instrument

from .engine import ExecutionEngine
from .order_manager import OrderManager
from .quality_tracker import ExecutionQualityTracker
from .risk.manager import RiskManager

logger = logging.getLogger(__name__)


class ExecutionGateway:
    """High-level facade unifying execution engine and risk management.

    Provides a single construction point and clean API for the full
    execution pipeline including risk checks, order management, and
    quality tracking.
    """

    def __init__(
        self,
        engine: ExecutionEngine,
        risk_manager: RiskManager,
        quality_tracker: ExecutionQualityTracker | None = None,
    ) -> None:
        self._engine = engine
        self._risk_manager = risk_manager
        self._quality_tracker = quality_tracker

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        adapter: IExchangeAdapter,
        event_bus: IEventBus,
        risk_config: RiskConfig | None = None,
        *,
        instruments: dict[str, Instrument] | None = None,
        portfolio_state_provider: Any = None,
        governance_gate: Any = None,
        tool_gateway: Any = None,
        redis_url: str | None = None,
        max_retries: int = 3,
    ) -> ExecutionGateway:
        """Construct a fully-wired ExecutionGateway from configuration.

        Builds the RiskManager and ExecutionEngine, wiring them together.

        Args:
            adapter: Exchange adapter (paper, backtest, or CCXT).
            event_bus: Event bus for publishing execution/risk events.
            risk_config: Risk configuration (defaults applied if None).
            instruments: Optional instrument metadata.
            portfolio_state_provider: Callable returning PortfolioState.
            governance_gate: Optional governance gate for legacy mode.
            tool_gateway: Optional control-plane gateway for CP mode.
            redis_url: Optional Redis URL for the kill switch.
            max_retries: Max order submission retries.
        """
        cfg = risk_config or RiskConfig()

        # Build risk manager
        risk_manager = RiskManager(
            config=cfg,
            event_bus=event_bus,
            instruments=instruments or {},
            redis_url=redis_url,
        )

        # Build execution engine
        engine = ExecutionEngine(
            adapter=adapter,
            event_bus=event_bus,
            risk_manager=risk_manager,
            kill_switch=None,
            portfolio_state_provider=portfolio_state_provider,
            max_retries=max_retries,
            governance_gate=governance_gate,
            tool_gateway=tool_gateway,
        )

        # Build quality tracker
        quality_tracker = ExecutionQualityTracker()

        return cls(
            engine=engine,
            risk_manager=risk_manager,
            quality_tracker=quality_tracker,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the execution engine (subscribes to event bus topics)."""
        await self._engine.start()
        logger.info("ExecutionGateway started")

    async def stop(self) -> None:
        """Stop the execution engine and clean up resources."""
        await self._engine.stop()
        await self._risk_manager.close()
        logger.info("ExecutionGateway stopped")

    # ------------------------------------------------------------------
    # Risk management passthrough
    # ------------------------------------------------------------------

    async def activate_kill_switch(
        self,
        reason: str,
        triggered_by: str = "execution_gateway",
    ) -> KillSwitchEvent:
        """Activate the global kill switch."""
        return await self._risk_manager.activate_kill_switch(
            reason=reason, triggered_by=triggered_by,
        )

    async def deactivate_kill_switch(self) -> KillSwitchEvent:
        """Deactivate the global kill switch."""
        return await self._risk_manager.deactivate_kill_switch()

    async def evaluate_circuit_breakers(
        self,
        values: dict[str, float],
        symbol: str = "",
    ) -> list[CircuitBreakerEvent]:
        """Evaluate circuit breakers and publish transition events."""
        return await self._risk_manager.evaluate_circuit_breakers(
            values, symbol=symbol,
        )

    def update_instruments(self, instruments: dict[str, Instrument]) -> None:
        """Update instrument metadata in the risk manager."""
        self._risk_manager.update_instruments(instruments)

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------

    @property
    def engine(self) -> ExecutionEngine:
        """Access the execution engine."""
        return self._engine

    @property
    def risk_manager(self) -> RiskManager:
        """Access the risk manager."""
        return self._risk_manager

    @property
    def order_manager(self) -> OrderManager:
        """Access the engine's order manager."""
        return self._engine._order_manager

    @property
    def quality_tracker(self) -> ExecutionQualityTracker | None:
        """Access the execution quality tracker."""
        return self._quality_tracker
