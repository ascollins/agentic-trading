"""Tests for Day 5: Eliminate direct adapter access.

Verifies:
    D1. ExecutionEngine accepts tool_gateway and routes through it
    D2. ExecutionAgent forwards tool_gateway to ExecutionEngine
    D3. AgentOrchestrator accepts and wires tool_gateway
    D4. ExecutionAgent no longer exposes .adapter property (B8)
    D5. AgentOrchestrator no longer exposes .adapter property (B9)
    D6. Grep audit: no direct adapter mutation calls outside ToolGateway
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ===========================================================================
# D1: ExecutionEngine CP wiring
# ===========================================================================


class TestD1EngineWiring:
    def test_engine_accepts_tool_gateway(self):
        from agentic_trading.execution.engine import ExecutionEngine

        gw = MagicMock()
        engine = ExecutionEngine(
            adapter=MagicMock(),
            event_bus=MagicMock(),
            risk_manager=MagicMock(),
            tool_gateway=gw,
        )
        assert engine.uses_control_plane is True
        assert engine.lifecycle_manager is not None

    def test_engine_without_gateway_is_legacy(self):
        from agentic_trading.execution.engine import ExecutionEngine

        engine = ExecutionEngine(
            adapter=MagicMock(),
            event_bus=MagicMock(),
            risk_manager=MagicMock(),
        )
        assert engine.uses_control_plane is False
        assert engine.lifecycle_manager is None


# ===========================================================================
# D2: ExecutionAgent forwards tool_gateway
# ===========================================================================


class TestD2ExecutionAgentWiring:
    @pytest.mark.asyncio
    async def test_execution_agent_accepts_tool_gateway(self):
        from agentic_trading.agents.execution import ExecutionAgent

        gw = MagicMock()
        agent = ExecutionAgent(
            adapter=MagicMock(),
            event_bus=AsyncMock(),
            risk_manager=MagicMock(),
            tool_gateway=gw,
        )
        assert agent._tool_gateway is gw

    @pytest.mark.asyncio
    async def test_execution_agent_starts_engine_with_gateway(self):
        from agentic_trading.agents.execution import ExecutionAgent

        gw = MagicMock()
        bus = AsyncMock()
        bus.subscribe = AsyncMock()

        agent = ExecutionAgent(
            adapter=MagicMock(),
            event_bus=bus,
            risk_manager=MagicMock(),
            tool_gateway=gw,
        )
        await agent._on_start()

        assert agent._execution_engine is not None
        assert agent._execution_engine.uses_control_plane is True


# ===========================================================================
# D3: AgentOrchestrator wiring
# ===========================================================================


class TestD3OrchestratorWiring:
    def test_orchestrator_accepts_tool_gateway(self):
        from agentic_trading.agents.orchestrator import AgentOrchestrator
        from agentic_trading.core.config import Settings
        from agentic_trading.core.interfaces import TradingContext

        gw = MagicMock()
        settings = MagicMock(spec=Settings)
        ctx = MagicMock(spec=TradingContext)

        orch = AgentOrchestrator(settings, ctx, tool_gateway=gw)
        assert orch.tool_gateway is gw


# ===========================================================================
# D4: ExecutionAgent .adapter removed (B8)
# ===========================================================================


class TestD4NoAdapterProperty:
    def test_execution_agent_no_adapter_property(self):
        """B8: ExecutionAgent must not expose .adapter property."""
        from agentic_trading.agents.execution import ExecutionAgent

        agent = ExecutionAgent(
            adapter=MagicMock(),
            event_bus=MagicMock(),
            risk_manager=MagicMock(),
        )
        assert not hasattr(agent, "adapter"), (
            "ExecutionAgent should not expose .adapter (B8 fix)"
        )


# ===========================================================================
# D5: AgentOrchestrator .adapter removed (B9)
# ===========================================================================


class TestD5OrchestratorNoAdapter:
    def test_orchestrator_no_adapter_property(self):
        """B9: AgentOrchestrator must not expose .adapter property."""
        from agentic_trading.agents.orchestrator import AgentOrchestrator
        from agentic_trading.core.config import Settings
        from agentic_trading.core.interfaces import TradingContext

        orch = AgentOrchestrator(
            MagicMock(spec=Settings),
            MagicMock(spec=TradingContext),
        )
        assert not hasattr(orch, "adapter"), (
            "AgentOrchestrator should not expose .adapter (B9 fix)"
        )


# ===========================================================================
# D6: Grep audit — no direct adapter mutation calls
# ===========================================================================


class TestD6GrepAudit:
    """Verify that direct adapter mutation calls only appear in
    allowed files: tool_gateway.py, ccxt_adapter.py, paper.py,
    and legacy fallback paths that guard with 'tool_gateway is None'.
    """

    def test_no_submit_order_outside_allowed_files(self):
        """adapter.submit_order should only appear in adapter impls, gateway,
        and the UI app (operator emergency close action)."""
        import subprocess

        result = subprocess.run(
            ["grep", "-rn", "adapter.submit_order",
             "src/agentic_trading/",
             "--include=*.py"],
            capture_output=True, text=True,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l]
        for line in lines:
            assert any(
                allowed in line
                for allowed in [
                    "tool_gateway.py",
                    "ccxt_adapter.py",
                    "paper.py",
                    "engine.py",  # legacy path
                    "ui/app.py",  # operator emergency close action (intentional bypass)
                ]
            ), f"Unexpected adapter.submit_order in: {line}"

    def test_no_set_trading_stop_outside_allowed(self):
        """adapter.set_trading_stop should only appear in gateway/adapter,
        main.py (guarded by tool_gateway check), or tpsl_watchdog.py
        (which routes via tool_gateway when available, else direct)."""
        import subprocess

        result = subprocess.run(
            ["grep", "-rn", r"adapter\.set_trading_stop",
             "src/agentic_trading/",
             "--include=*.py"],
            capture_output=True, text=True,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l]
        for line in lines:
            assert any(
                allowed in line
                for allowed in [
                    "tool_gateway.py",
                    "ccxt_adapter.py",
                    "paper.py",
                    "main.py",          # guarded by `else:` after tool_gateway check
                    "tpsl_watchdog.py", # routes via tool_gateway when available
                ]
            ), f"Unexpected adapter.set_trading_stop in: {line}"

    def test_no_adapter_cancel_outside_allowed(self):
        """adapter.cancel_order should only appear in gateway and adapter impls."""
        import subprocess

        result = subprocess.run(
            ["grep", "-rn", r"adapter\.cancel",
             "src/agentic_trading/",
             "--include=*.py"],
            capture_output=True, text=True,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l]
        for line in lines:
            assert any(
                allowed in line
                for allowed in [
                    "tool_gateway.py",
                    "ccxt_adapter.py",
                    "paper.py",
                ]
            ), f"Unexpected adapter.cancel in: {line}"

    def test_main_set_trading_stop_guarded(self):
        """main.py set_trading_stop calls must be guarded by tool_gateway check."""
        import re
        from pathlib import Path

        content = Path("src/agentic_trading/main.py").read_text()
        # Find all adapter.set_trading_stop calls
        for match in re.finditer(r"adapter\.set_trading_stop", content):
            # Get surrounding context (100 chars before)
            start = max(0, match.start() - 200)
            context = content[start:match.end()]
            assert any(
                guard in context
                for guard in ["else:", "tool_gateway is None", "tool_gateway is not None"]
            ), (
                f"adapter.set_trading_stop at position {match.start()} "
                "is not guarded by tool_gateway check"
            )
