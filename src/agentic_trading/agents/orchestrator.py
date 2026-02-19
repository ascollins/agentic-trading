"""Agent orchestrator: coordinates agent creation, wiring, and lifecycle.

Replaces the procedural component wiring in ``main.py`` with a
declarative agent-based approach.  The orchestrator:

1. Reads the ``Settings`` config
2. Creates and registers all agents in the ``AgentRegistry``
3. Manages coordinated startup and shutdown
4. Provides discovery of running agents by type

The orchestrator does NOT own the event bus or clock -- those are
created externally and passed in via ``TradingContext``.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agentic_trading.agents.registry import AgentRegistry
from agentic_trading.core.config import Settings
from agentic_trading.core.enums import AgentType, Mode
from agentic_trading.core.interfaces import TradingContext

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Creates, registers, and manages all platform agents.

    Usage::

        orchestrator = AgentOrchestrator(settings, ctx)
        await orchestrator.setup()    # Create & register agents
        await orchestrator.start()    # Start all agents
        # ... trading runs ...
        await orchestrator.stop()     # Graceful shutdown
    """

    def __init__(
        self,
        settings: Settings,
        ctx: TradingContext,
        tool_gateway: Any = None,
    ) -> None:
        self._settings = settings
        self._ctx = ctx
        self._registry = AgentRegistry()
        self._adapter: Any = None
        self._tool_gateway = tool_gateway

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry."""
        return self._registry

    # ------------------------------------------------------------------
    # Setup: create and register all agents
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create all agents based on config and register them.

        Agent creation order matters for startup sequencing:
        1. RiskGateAgent (risk/governance must be ready first)
        2. MarketIntelligenceAgent (data feeds + features)
        3. ExecutionAgent (order lifecycle)
        4. GovernanceCanary (infrastructure watchdog)
        5. ReconciliationLoop (periodic reconciliation)
        6. OptimizerScheduler (periodic optimization)
        """
        settings = self._settings
        ctx = self._ctx

        # --- 1. Create exchange adapter ---
        self._adapter = await self._create_adapter(settings)

        # Fetch instrument metadata
        if self._adapter is not None:
            await self._fetch_instruments(settings, ctx)

        # --- 2. RiskGateAgent ---
        from agentic_trading.agents.risk_gate import RiskGateAgent

        risk_gate_agent = RiskGateAgent(
            event_bus=ctx.event_bus,
            risk_config=settings.risk,
            governance_config=(
                settings.governance
                if settings.governance.enabled
                else None
            ),
            instruments=ctx.instruments,
            agent_id="risk-gate",
        )
        self._registry.register(risk_gate_agent)

        # --- 3. MarketIntelligenceAgent ---
        from agentic_trading.agents.market_intelligence import (
            MarketIntelligenceAgent,
        )

        symbols = settings.symbols.symbols or []
        mi_agent = MarketIntelligenceAgent(
            event_bus=ctx.event_bus,
            exchange_configs=(
                settings.exchanges if settings.exchanges else None
            ),
            symbols=symbols,
            agent_id="market-intelligence",
        )
        self._registry.register(mi_agent)

        # --- 4. ExecutionAgent ---
        from agentic_trading.agents.execution import ExecutionAgent

        # We need the risk manager from the risk gate agent, but it's
        # not started yet. We'll wire it after start via a deferred
        # setup. For now, create a placeholder risk manager.
        from agentic_trading.risk.manager import RiskManager

        # Create a temporary risk manager for the execution engine.
        # The RiskGateAgent creates its own -- we share the config.
        risk_manager = RiskManager(
            config=settings.risk,
            event_bus=ctx.event_bus,
            instruments=ctx.instruments,
        )

        # Determine governance gate (will be available after risk gate starts)
        governance_gate = None

        # Portfolio sizing
        gov_sizing_fn = None
        sizing_mult = 1.0
        if settings.safe_mode.enabled:
            sizing_mult = settings.safe_mode.position_size_multiplier

        if self._adapter is not None:
            exec_agent = ExecutionAgent(
                adapter=self._adapter,
                event_bus=ctx.event_bus,
                risk_manager=risk_manager,
                kill_switch_fn=risk_manager.kill_switch.is_active,
                portfolio_state_provider=lambda: ctx.portfolio_state,
                governance_gate=governance_gate,
                tool_gateway=self._tool_gateway,
                agent_id="execution",
            )
            self._registry.register(exec_agent)

        # --- 5. GovernanceCanary ---
        if settings.governance.enabled:
            from agentic_trading.governance.canary import GovernanceCanary

            canary = GovernanceCanary(
                settings.governance.canary,
                event_bus=ctx.event_bus,
                agent_id="governance-canary",
            )
            self._registry.register(canary)

        # --- 6. OptimizerScheduler ---
        if settings.optimizer_scheduler.enabled:
            from agentic_trading.optimizer.scheduler import OptimizerScheduler

            optimizer = OptimizerScheduler(
                config=settings.optimizer_scheduler,
                data_dir=settings.backtest.data_dir,
                event_bus=ctx.event_bus,
                strategy_config=settings.strategies,
                governance_gate=governance_gate,
                agent_id="optimizer-scheduler",
            )
            self._registry.register(optimizer)

        logger.info(
            "AgentOrchestrator: %d agents registered",
            self._registry.count,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all registered agents in registration order."""
        await self._registry.start_all()
        logger.info(
            "AgentOrchestrator: all agents started\n%s",
            "\n".join(
                f"  - {s['name']} ({s['type']}) running={s['running']}"
                for s in self._registry.summary()
            ),
        )

    async def stop(self) -> None:
        """Stop all agents in reverse order."""
        await self._registry.stop_all()
        logger.info("AgentOrchestrator: all agents stopped")

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Any:
        """Look up an agent by ID."""
        return self._registry.get_agent(agent_id)

    def get_agents_by_type(self, agent_type: AgentType) -> list[Any]:
        """Find all agents of a given type."""
        return self._registry.get_agents_by_type(agent_type)

    @property
    def tool_gateway(self) -> Any:
        """Access the ToolGateway for exchange operations."""
        return self._tool_gateway

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _create_adapter(self, settings: Settings) -> Any:
        """Create the exchange adapter based on mode."""
        active_exchange = settings.exchanges[0].name if settings.exchanges else None

        if settings.mode == Mode.PAPER:
            from agentic_trading.core.enums import Exchange
            from agentic_trading.execution.adapters.paper import PaperAdapter

            exchange = (
                active_exchange
                if active_exchange is not None
                else Exchange.BINANCE
            )
            adapter = PaperAdapter(
                exchange=exchange,
                initial_balances={"USDT": Decimal("100000")},
            )
            logger.info(
                "Paper adapter ready with 100,000 USDT on %s",
                exchange.value,
            )
            return adapter

        if settings.mode == Mode.LIVE:
            from agentic_trading.execution.adapters.ccxt_adapter import (
                CCXTAdapter,
            )

            if not settings.exchanges:
                logger.error(
                    "Live mode requires at least one exchange config",
                )
                return None
            exc_cfg = settings.exchanges[0]
            adapter = CCXTAdapter(
                exchange_name=exc_cfg.name.value,
                api_key=exc_cfg.api_key,
                api_secret=exc_cfg.secret,
                sandbox=exc_cfg.testnet,
                demo=exc_cfg.demo,
                default_type="swap",
            )
            logger.info(
                "Live adapter ready: %s (testnet=%s, demo=%s)",
                exc_cfg.name.value,
                exc_cfg.testnet,
                exc_cfg.demo,
            )
            return adapter

        return None

    async def _fetch_instruments(
        self, settings: Settings, ctx: TradingContext
    ) -> None:
        """Fetch instrument metadata from the exchange."""
        symbols = settings.symbols.symbols or []
        if not symbols or self._adapter is None:
            return

        logger.info(
            "Fetching instrument metadata for %d symbols...",
            len(symbols),
        )
        for sym in symbols:
            try:
                inst = await self._adapter.get_instrument(sym)
                if inst is not None:
                    ctx.instruments[sym] = inst
                    logger.info(
                        "  %s: tick=%s step=%s min_qty=%s",
                        sym,
                        inst.tick_size,
                        inst.step_size,
                        inst.min_qty,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to fetch instrument for %s: %s", sym, e,
                )
        logger.info(
            "Instruments loaded: %d/%d symbols",
            len(ctx.instruments),
            len(symbols),
        )
