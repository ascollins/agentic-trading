"""Execution Agent.

Wraps the ExecutionEngine, exchange adapter, and RiskManager into a
single agent responsible for the full order lifecycle: intent reception,
risk gating, exchange submission, and fill handling.

This is an event-driven agent (no periodic loop) -- it subscribes to
``execution`` and ``system`` topics via the ExecutionEngine.

When a ``tool_gateway`` is provided, all exchange side effects are
routed through the institutional control plane.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentCapabilities
from agentic_trading.core.interfaces import (
    IEventBus,
    IExchangeAdapter,
    IRiskChecker,
    PortfolioState,
)

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    """Manages the order lifecycle from intent to fill.

    Orchestrates:
    - ExecutionEngine (intent processing, order management)
    - Exchange adapter (paper/live order submission)
    - RiskManager (pre-trade risk checks)
    - GovernanceGate (optional governance evaluation)
    - ToolGateway (institutional control plane, when available)

    Usage::

        agent = ExecutionAgent(
            adapter=paper_adapter,
            event_bus=event_bus,
            risk_manager=risk_manager,
            kill_switch_fn=risk_manager.kill_switch.is_active,
            tool_gateway=tool_gateway,
        )
        await agent.start()
    """

    def __init__(
        self,
        adapter: IExchangeAdapter,
        event_bus: IEventBus,
        risk_manager: IRiskChecker,
        kill_switch_fn: Any = None,
        portfolio_state_provider: Any = None,
        governance_gate: Any = None,
        max_retries: int = 3,
        tool_gateway: Any = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, interval=0)
        self._adapter = adapter
        self._event_bus = event_bus
        self._risk_manager = risk_manager
        self._kill_switch_fn = kill_switch_fn
        self._portfolio_state_provider = portfolio_state_provider
        self._governance_gate = governance_gate
        self._max_retries = max_retries
        self._tool_gateway = tool_gateway

        # Created during start
        self._execution_engine: Any = None

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.EXECUTION

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["execution", "system"],
            publishes_to=["execution"],
            description="Order lifecycle management: intent → risk → submit → fill",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        from agentic_trading.execution.engine import ExecutionEngine

        self._execution_engine = ExecutionEngine(
            adapter=self._adapter,
            event_bus=self._event_bus,
            risk_manager=self._risk_manager,
            kill_switch=self._kill_switch_fn,
            portfolio_state_provider=self._portfolio_state_provider,
            governance_gate=self._governance_gate,
            max_retries=self._max_retries,
            tool_gateway=self._tool_gateway,
        )
        await self._execution_engine.start()
        logger.info(
            "ExecutionAgent: ExecutionEngine started (cp_mode=%s)",
            self._tool_gateway is not None,
        )

    async def _on_stop(self) -> None:
        if self._execution_engine is not None:
            await self._execution_engine.stop()
            logger.info("ExecutionAgent: ExecutionEngine stopped")

        # Close the adapter if it has a close method
        if self._adapter is not None and hasattr(self._adapter, "close"):
            try:
                await self._adapter.close()
                logger.info("ExecutionAgent: adapter closed")
            except Exception:
                logger.warning(
                    "ExecutionAgent: adapter close failed",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def execution_engine(self) -> Any:
        """Access the underlying ExecutionEngine."""
        return self._execution_engine
