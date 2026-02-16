"""Tests for the multi-agent framework (BaseAgent, AgentRegistry)."""

from __future__ import annotations

import asyncio

import pytest

from agentic_trading.agents.base import BaseAgent
from agentic_trading.agents.registry import AgentRegistry
from agentic_trading.core.enums import AgentStatus, AgentType
from agentic_trading.core.events import AgentCapabilities, AgentHealthReport
from agentic_trading.core.interfaces import IAgent


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class DummyAgent(BaseAgent):
    """Minimal concrete agent for testing."""

    def __init__(
        self, *, agent_id: str | None = None, interval: float = 0
    ) -> None:
        super().__init__(agent_id=agent_id, interval=interval)
        self.work_count = 0
        self.started = False
        self.stopped = False

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CUSTOM

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["test.topic"],
            publishes_to=["test.output"],
            description="Dummy agent for tests",
        )

    async def _on_start(self) -> None:
        self.started = True

    async def _on_stop(self) -> None:
        self.stopped = True

    async def _work(self) -> None:
        self.work_count += 1


class FailingAgent(BaseAgent):
    """Agent whose _work raises to test error handling."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CUSTOM

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(description="Fails on work")

    async def _work(self) -> None:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------


class TestBaseAgent:
    def test_agent_id_auto_generated(self):
        agent = DummyAgent()
        assert len(agent.agent_id) == 36  # UUID format

    def test_agent_id_custom(self):
        agent = DummyAgent(agent_id="my-agent-001")
        assert agent.agent_id == "my-agent-001"

    def test_initial_status(self):
        agent = DummyAgent()
        assert agent.status == AgentStatus.CREATED
        assert not agent.is_running

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        agent = DummyAgent()
        assert agent.status == AgentStatus.CREATED

        await agent.start()
        assert agent.is_running
        assert agent.status == AgentStatus.RUNNING
        assert agent.started

        await agent.stop()
        assert not agent.is_running
        assert agent.status == AgentStatus.STOPPED
        assert agent.stopped

    @pytest.mark.asyncio
    async def test_periodic_work(self):
        agent = DummyAgent(interval=0.01)
        await agent.start()

        # Let a few cycles run
        await asyncio.sleep(0.05)
        await agent.stop()

        assert agent.work_count >= 2

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        agent = DummyAgent()
        await agent.start()
        await agent.start()  # Should not raise
        assert agent.is_running
        await agent.stop()

    def test_health_check_before_start(self):
        agent = DummyAgent()
        health = agent.health_check()
        assert not health.healthy
        assert "not running" in health.message

    @pytest.mark.asyncio
    async def test_health_check_while_running(self):
        agent = DummyAgent()
        await agent.start()
        health = agent.health_check()
        assert health.healthy
        assert health.error_count == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_error_count_increments(self):
        agent = FailingAgent(interval=0.01)
        await agent.start()
        await asyncio.sleep(0.05)
        await agent.stop()

        health = agent.health_check()
        assert health.error_count >= 2

    def test_capabilities(self):
        agent = DummyAgent()
        caps = agent.capabilities()
        assert "test.topic" in caps.subscribes_to
        assert "test.output" in caps.publishes_to

    def test_agent_name_defaults_to_class_name(self):
        agent = DummyAgent()
        assert agent.agent_name == "DummyAgent"

    def test_agent_type(self):
        agent = DummyAgent()
        assert agent.agent_type == AgentType.CUSTOM

    def test_satisfies_iagent_protocol(self):
        agent = DummyAgent()
        assert isinstance(agent, IAgent)


# ---------------------------------------------------------------------------
# AgentRegistry tests
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_register(self):
        registry = AgentRegistry()
        agent = DummyAgent(agent_id="agent-1")
        registry.register(agent)
        assert registry.count == 1

    def test_register_duplicate_raises(self):
        registry = AgentRegistry()
        agent = DummyAgent(agent_id="agent-1")
        registry.register(agent)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(agent)

    def test_unregister(self):
        registry = AgentRegistry()
        agent = DummyAgent(agent_id="agent-1")
        registry.register(agent)
        removed = registry.unregister("agent-1")
        assert removed is agent
        assert registry.count == 0

    def test_unregister_missing_returns_none(self):
        registry = AgentRegistry()
        assert registry.unregister("nonexistent") is None

    def test_get_agent(self):
        registry = AgentRegistry()
        agent = DummyAgent(agent_id="agent-1")
        registry.register(agent)
        assert registry.get_agent("agent-1") is agent
        assert registry.get_agent("nonexistent") is None

    def test_get_agents_by_type(self):
        registry = AgentRegistry()
        a1 = DummyAgent(agent_id="a1")
        a2 = DummyAgent(agent_id="a2")
        registry.register(a1)
        registry.register(a2)
        found = registry.get_agents_by_type(AgentType.CUSTOM)
        assert len(found) == 2

    @pytest.mark.asyncio
    async def test_start_all_stop_all(self):
        registry = AgentRegistry()
        a1 = DummyAgent(agent_id="a1")
        a2 = DummyAgent(agent_id="a2")
        registry.register(a1)
        registry.register(a2)

        await registry.start_all()
        assert a1.is_running
        assert a2.is_running

        await registry.stop_all()
        assert not a1.is_running
        assert not a2.is_running

    @pytest.mark.asyncio
    async def test_stop_all_reverse_order(self):
        """Agents are stopped in reverse registration order."""
        registry = AgentRegistry()
        stop_order: list[str] = []

        class TrackingAgent(DummyAgent):
            async def _on_stop(self) -> None:
                stop_order.append(self.agent_id)

        a1 = TrackingAgent(agent_id="first")
        a2 = TrackingAgent(agent_id="second")
        registry.register(a1)
        registry.register(a2)

        await registry.start_all()
        await registry.stop_all()

        assert stop_order == ["second", "first"]

    def test_health_check_all(self):
        registry = AgentRegistry()
        a1 = DummyAgent(agent_id="a1")
        registry.register(a1)
        reports = registry.health_check_all()
        assert "a1" in reports
        assert not reports["a1"].healthy  # Not started yet

    @pytest.mark.asyncio
    async def test_all_healthy(self):
        registry = AgentRegistry()
        agent = DummyAgent(agent_id="a1")
        registry.register(agent)
        await registry.start_all()
        assert registry.all_healthy()
        await registry.stop_all()

    def test_summary(self):
        registry = AgentRegistry()
        agent = DummyAgent(agent_id="a1234567-1234-1234-1234-123456789012")
        registry.register(agent)
        summary = registry.summary()
        assert len(summary) == 1
        assert summary[0]["type"] == "custom"
        assert summary[0]["name"] == "DummyAgent"

    @pytest.mark.asyncio
    async def test_start_all_handles_failure(self):
        """If one agent fails to start, others still start."""
        registry = AgentRegistry()

        class BrokenStartAgent(DummyAgent):
            async def _on_start(self) -> None:
                raise RuntimeError("Cannot start")

        broken = BrokenStartAgent(agent_id="broken")
        good = DummyAgent(agent_id="good")
        registry.register(broken)
        registry.register(good)

        await registry.start_all()
        # The good agent should still have started
        assert good.is_running
        await registry.stop_all()


# ---------------------------------------------------------------------------
# Proto-agent refactoring tests
# ---------------------------------------------------------------------------


class TestGovernanceCanaryAgent:
    """Verify GovernanceCanary satisfies IAgent after refactoring."""

    def test_satisfies_iagent(self):
        from agentic_trading.core.config import CanaryConfig
        from agentic_trading.governance.canary import GovernanceCanary

        config = CanaryConfig()
        canary = GovernanceCanary(config)
        assert isinstance(canary, IAgent)
        assert canary.agent_type == AgentType.GOVERNANCE_CANARY

    @pytest.mark.asyncio
    async def test_start_stop(self):
        from agentic_trading.core.config import CanaryConfig
        from agentic_trading.governance.canary import GovernanceCanary

        config = CanaryConfig(check_interval_seconds=60)
        canary = GovernanceCanary(config)
        await canary.start()
        assert canary.is_running
        await canary.stop()
        assert not canary.is_running

    @pytest.mark.asyncio
    async def test_start_periodic_backward_compat(self):
        from agentic_trading.core.config import CanaryConfig
        from agentic_trading.governance.canary import GovernanceCanary

        config = CanaryConfig(check_interval_seconds=60)
        canary = GovernanceCanary(config)
        await canary.start_periodic(interval=60)
        assert canary.is_running
        await canary.stop()


class TestReconciliationLoopAgent:
    """Verify ReconciliationLoop satisfies IAgent after refactoring."""

    def test_satisfies_iagent(self):
        from unittest.mock import AsyncMock

        from agentic_trading.execution.order_manager import OrderManager
        from agentic_trading.execution.reconciliation import ReconciliationLoop

        loop = ReconciliationLoop(
            adapter=AsyncMock(),
            event_bus=AsyncMock(),
            order_manager=OrderManager(),
        )
        assert isinstance(loop, IAgent)
        assert loop.agent_type == AgentType.RECONCILIATION


class TestOptimizerSchedulerAgent:
    """Verify OptimizerScheduler satisfies IAgent after refactoring."""

    def test_satisfies_iagent(self):
        from agentic_trading.core.config import OptimizerSchedulerConfig
        from agentic_trading.optimizer.scheduler import OptimizerScheduler

        config = OptimizerSchedulerConfig()
        scheduler = OptimizerScheduler(config)
        assert isinstance(scheduler, IAgent)
        assert scheduler.agent_type == AgentType.OPTIMIZER
