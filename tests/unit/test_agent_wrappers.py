"""Tests for the pipeline agent wrappers and orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.core.enums import AgentType
from agentic_trading.core.interfaces import IAgent


# ---------------------------------------------------------------------------
# MarketIntelligenceAgent
# ---------------------------------------------------------------------------


class TestMarketIntelligenceAgent:
    def test_satisfies_iagent(self):
        from agentic_trading.agents.market_intelligence import (
            MarketIntelligenceAgent,
        )

        agent = MarketIntelligenceAgent(
            event_bus=AsyncMock(),
            agent_id="mi-test",
        )
        assert isinstance(agent, IAgent)
        assert agent.agent_type == AgentType.MARKET_INTELLIGENCE

    def test_capabilities(self):
        from agentic_trading.agents.market_intelligence import (
            MarketIntelligenceAgent,
        )

        agent = MarketIntelligenceAgent(event_bus=AsyncMock())
        caps = agent.capabilities()
        assert "market.candle" in caps.subscribes_to
        assert "feature.vector" in caps.publishes_to

    @pytest.mark.asyncio
    async def test_start_stop_without_feeds(self):
        """Start/stop with no exchange configs (feature engine only)."""
        from agentic_trading.agents.market_intelligence import (
            MarketIntelligenceAgent,
        )

        event_bus = AsyncMock()
        agent = MarketIntelligenceAgent(
            event_bus=event_bus,
            agent_id="mi-test",
        )

        await agent.start()
        assert agent.is_running
        assert agent.feature_engine is not None
        assert agent.feed_manager is None

        await agent.stop()
        assert not agent.is_running

    @pytest.mark.asyncio
    async def test_feature_engine_accessible(self):
        from agentic_trading.agents.market_intelligence import (
            MarketIntelligenceAgent,
        )

        agent = MarketIntelligenceAgent(
            event_bus=AsyncMock(),
            indicator_config={"smc_enabled": False},
        )
        await agent.start()
        assert agent.feature_engine is not None
        await agent.stop()


# ---------------------------------------------------------------------------
# ExecutionAgent
# ---------------------------------------------------------------------------


class TestExecutionAgent:
    def test_satisfies_iagent(self):
        from agentic_trading.agents.execution import ExecutionAgent

        agent = ExecutionAgent(
            adapter=AsyncMock(),
            event_bus=AsyncMock(),
            risk_manager=AsyncMock(),
            agent_id="exec-test",
        )
        assert isinstance(agent, IAgent)
        assert agent.agent_type == AgentType.EXECUTION

    def test_capabilities(self):
        from agentic_trading.agents.execution import ExecutionAgent

        agent = ExecutionAgent(
            adapter=AsyncMock(),
            event_bus=AsyncMock(),
            risk_manager=AsyncMock(),
        )
        caps = agent.capabilities()
        assert "execution" in caps.subscribes_to
        assert "system" in caps.subscribes_to

    @pytest.mark.asyncio
    async def test_start_stop(self):
        from agentic_trading.agents.execution import ExecutionAgent

        event_bus = AsyncMock()
        adapter = AsyncMock()
        risk_manager = AsyncMock()

        agent = ExecutionAgent(
            adapter=adapter,
            event_bus=event_bus,
            risk_manager=risk_manager,
            agent_id="exec-test",
        )

        await agent.start()
        assert agent.is_running
        assert agent.execution_engine is not None

        await agent.stop()
        assert not agent.is_running

    @pytest.mark.asyncio
    async def test_adapter_accessible(self):
        from agentic_trading.agents.execution import ExecutionAgent

        adapter = AsyncMock()
        agent = ExecutionAgent(
            adapter=adapter,
            event_bus=AsyncMock(),
            risk_manager=AsyncMock(),
        )
        assert agent.adapter is adapter


# ---------------------------------------------------------------------------
# RiskGateAgent
# ---------------------------------------------------------------------------


class TestRiskGateAgent:
    def test_satisfies_iagent(self):
        from agentic_trading.agents.risk_gate import RiskGateAgent

        agent = RiskGateAgent(
            event_bus=AsyncMock(),
            agent_id="risk-test",
        )
        assert isinstance(agent, IAgent)
        assert agent.agent_type == AgentType.RISK_GATE

    def test_capabilities(self):
        from agentic_trading.agents.risk_gate import RiskGateAgent

        agent = RiskGateAgent(event_bus=AsyncMock())
        caps = agent.capabilities()
        assert "risk" in caps.publishes_to
        assert "governance" in caps.publishes_to

    @pytest.mark.asyncio
    async def test_start_stop_without_governance(self):
        from agentic_trading.agents.risk_gate import RiskGateAgent

        agent = RiskGateAgent(
            event_bus=AsyncMock(),
            agent_id="risk-test",
        )

        await agent.start()
        assert agent.is_running
        assert agent.risk_manager is not None
        assert agent.governance_gate is None

        await agent.stop()
        assert not agent.is_running

    @pytest.mark.asyncio
    async def test_start_with_governance(self):
        from agentic_trading.agents.risk_gate import RiskGateAgent
        from agentic_trading.core.config import GovernanceConfig

        gov_config = GovernanceConfig(enabled=True)
        agent = RiskGateAgent(
            event_bus=AsyncMock(),
            governance_config=gov_config,
            agent_id="risk-gov-test",
        )

        await agent.start()
        assert agent.risk_manager is not None
        assert agent.governance_gate is not None
        await agent.stop()


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------


class TestAgentOrchestrator:
    def _make_settings(self, mode: str = "paper") -> MagicMock:
        """Create a minimal Settings mock for orchestrator tests."""
        from agentic_trading.core.config import (
            GovernanceConfig,
            ObservabilityConfig,
            OptimizerSchedulerConfig,
            RiskConfig,
            SafeModeConfig,
            SymbolConfig,
        )
        from agentic_trading.core.enums import Mode

        settings = MagicMock()
        settings.mode = Mode(mode)
        settings.risk = RiskConfig()
        settings.governance = GovernanceConfig(enabled=False)
        settings.safe_mode = SafeModeConfig()
        settings.optimizer_scheduler = OptimizerSchedulerConfig(enabled=False)
        settings.symbols = SymbolConfig(symbols=["BTC/USDT"])
        settings.exchanges = []
        settings.backtest = MagicMock()
        settings.backtest.data_dir = "data/historical"
        settings.strategies = []
        settings.read_only = False
        settings.observability = ObservabilityConfig()
        return settings

    def _make_ctx(self) -> MagicMock:
        from agentic_trading.core.interfaces import PortfolioState

        ctx = MagicMock()
        ctx.event_bus = AsyncMock()
        ctx.instruments = {}
        ctx.portfolio_state = PortfolioState()
        return ctx

    @pytest.mark.asyncio
    async def test_setup_registers_agents(self):
        from agentic_trading.agents.orchestrator import AgentOrchestrator

        settings = self._make_settings("paper")
        ctx = self._make_ctx()

        orchestrator = AgentOrchestrator(settings, ctx)
        await orchestrator.setup()

        # Should have: risk-gate, market-intelligence, execution
        assert orchestrator.registry.count >= 3
        assert orchestrator.get_agent("risk-gate") is not None
        assert orchestrator.get_agent("market-intelligence") is not None
        assert orchestrator.get_agent("execution") is not None

    @pytest.mark.asyncio
    async def test_setup_no_execution_in_backtest(self):
        from agentic_trading.agents.orchestrator import AgentOrchestrator

        settings = self._make_settings("backtest")
        ctx = self._make_ctx()

        orchestrator = AgentOrchestrator(settings, ctx)
        await orchestrator.setup()

        # Backtest mode creates no adapter, so no execution agent
        assert orchestrator.get_agent("execution") is None

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        from agentic_trading.agents.orchestrator import AgentOrchestrator

        settings = self._make_settings("paper")
        ctx = self._make_ctx()

        orchestrator = AgentOrchestrator(settings, ctx)
        await orchestrator.setup()
        await orchestrator.start()

        # All agents should be running
        for summary in orchestrator.registry.summary():
            assert summary["running"], f"{summary['name']} not running"

        await orchestrator.stop()

        # All agents should be stopped
        for summary in orchestrator.registry.summary():
            assert not summary["running"], f"{summary['name']} still running"

    @pytest.mark.asyncio
    async def test_governance_canary_registered_when_enabled(self):
        from agentic_trading.agents.orchestrator import AgentOrchestrator
        from agentic_trading.core.config import GovernanceConfig

        settings = self._make_settings("paper")
        settings.governance = GovernanceConfig(enabled=True)
        ctx = self._make_ctx()

        orchestrator = AgentOrchestrator(settings, ctx)
        await orchestrator.setup()

        assert orchestrator.get_agent("governance-canary") is not None

    @pytest.mark.asyncio
    async def test_optimizer_registered_when_enabled(self):
        from agentic_trading.agents.orchestrator import AgentOrchestrator
        from agentic_trading.core.config import OptimizerSchedulerConfig

        settings = self._make_settings("paper")
        settings.optimizer_scheduler = OptimizerSchedulerConfig(
            enabled=True,
            interval_hours=1.0,
            strategies=["trend_following"],
        )
        ctx = self._make_ctx()

        orchestrator = AgentOrchestrator(settings, ctx)
        await orchestrator.setup()

        assert orchestrator.get_agent("optimizer-scheduler") is not None

    @pytest.mark.asyncio
    async def test_get_agents_by_type(self):
        from agentic_trading.agents.orchestrator import AgentOrchestrator

        settings = self._make_settings("paper")
        ctx = self._make_ctx()

        orchestrator = AgentOrchestrator(settings, ctx)
        await orchestrator.setup()

        risk_agents = orchestrator.get_agents_by_type(AgentType.RISK_GATE)
        assert len(risk_agents) == 1

        mi_agents = orchestrator.get_agents_by_type(
            AgentType.MARKET_INTELLIGENCE,
        )
        assert len(mi_agents) == 1

    @pytest.mark.asyncio
    async def test_adapter_accessible(self):
        from agentic_trading.agents.orchestrator import AgentOrchestrator

        settings = self._make_settings("paper")
        ctx = self._make_ctx()

        orchestrator = AgentOrchestrator(settings, ctx)
        await orchestrator.setup()

        # Paper mode should have created an adapter
        assert orchestrator.adapter is not None
