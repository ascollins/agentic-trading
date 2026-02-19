"""Tests for the main.py ↔ Orchestrator integration.

Validates that ``main.run()`` now constructs an Orchestrator under the
hood and that the resulting components are correctly wired.

Tests cover:
1. ``run()`` imports and uses ``Orchestrator``
2. ``Orchestrator.from_config`` is invoked via ``main.run``
3. Backtest mode still routes to ``_run_backtest``
4. ``_setup_logging`` still works
5. Main module structure preserved (cli entry points, walk-forward, optimize)
"""

from __future__ import annotations

import importlib
import inspect

import pytest


# ---------------------------------------------------------------------------
# Module import structure
# ---------------------------------------------------------------------------

class TestMainImportsOrchestrator:
    """main.py now imports from orchestrator module."""

    def test_main_imports_orchestrator(self):
        from agentic_trading import main

        source = inspect.getsource(main)
        assert "Orchestrator" in source

    def test_main_no_longer_imports_create_event_bus(self):
        """The old direct ``create_event_bus`` import should be gone."""
        from agentic_trading import main

        source = inspect.getsource(main)
        assert "from .event_bus.bus import create_event_bus" not in source

    def test_main_no_longer_imports_simclock_wallclock(self):
        """Clock construction is delegated to Orchestrator."""
        from agentic_trading import main

        source = inspect.getsource(main)
        assert "from .core.clock import SimClock, WallClock" not in source

    def test_main_still_has_run_function(self):
        from agentic_trading.main import run

        assert callable(run)
        assert inspect.iscoroutinefunction(run)

    def test_main_still_has_setup_logging(self):
        from agentic_trading.main import _setup_logging

        assert callable(_setup_logging)

    def test_main_still_has_run_walk_forward(self):
        from agentic_trading.main import run_walk_forward

        assert callable(run_walk_forward)
        assert inspect.iscoroutinefunction(run_walk_forward)

    def test_main_still_has_run_optimize(self):
        from agentic_trading.main import run_optimize

        assert callable(run_optimize)
        assert inspect.iscoroutinefunction(run_optimize)

    def test_main_still_has_run_backtest(self):
        from agentic_trading.main import _run_backtest

        assert callable(_run_backtest)
        assert inspect.iscoroutinefunction(_run_backtest)

    def test_main_still_has_run_live_or_paper(self):
        from agentic_trading.main import _run_live_or_paper

        assert callable(_run_live_or_paper)
        assert inspect.iscoroutinefunction(_run_live_or_paper)


# ---------------------------------------------------------------------------
# Orchestrator construction from main.run
# ---------------------------------------------------------------------------

class TestMainRunCreatesOrchestrator:
    """main.run() constructs an Orchestrator and uses its components."""

    def test_run_source_uses_from_config(self):
        """Verify run() calls Orchestrator.from_config."""
        from agentic_trading import main

        source = inspect.getsource(main.run)
        assert "Orchestrator.from_config" in source

    def test_run_source_uses_orch_bus_start(self):
        """Verify run() starts the bus via orchestrator."""
        from agentic_trading import main

        source = inspect.getsource(main.run)
        assert "orch.bus.start()" in source

    def test_run_source_uses_orch_bus_stop(self):
        """Verify run() stops the bus via orchestrator."""
        from agentic_trading import main

        source = inspect.getsource(main.run)
        assert "orch.bus.stop()" in source

    def test_run_source_uses_orch_ctx(self):
        """Verify run() accesses ctx from orchestrator."""
        from agentic_trading import main

        source = inspect.getsource(main.run)
        assert "orch.ctx" in source

    def test_run_source_uses_legacy_bus(self):
        """Verify run() accesses the legacy bus from orchestrator."""
        from agentic_trading import main

        source = inspect.getsource(main.run)
        assert "orch.bus.legacy_bus" in source


# ---------------------------------------------------------------------------
# Orchestrator-constructed objects are valid
# ---------------------------------------------------------------------------

class TestOrchestratorObjectsFromMain:
    """Verify that the Orchestrator creates the same object types
    that main.py used to construct directly."""

    def test_backtest_creates_sim_clock(self):
        from agentic_trading.core.clock import SimClock
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.BACKTEST)
        orch = Orchestrator.from_config(settings)
        assert isinstance(orch.ctx.clock, SimClock)

    def test_paper_creates_wall_clock(self):
        from agentic_trading.core.clock import WallClock
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.PAPER)
        orch = Orchestrator.from_config(settings)
        assert isinstance(orch.ctx.clock, WallClock)

    def test_backtest_creates_memory_bus(self):
        from agentic_trading.bus.memory_bus import MemoryEventBus
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.BACKTEST)
        orch = Orchestrator.from_config(settings)
        assert isinstance(orch.bus.legacy_bus, MemoryEventBus)

    def test_ctx_has_risk_limits_from_settings(self):
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.BACKTEST)
        orch = Orchestrator.from_config(settings)
        assert "max_portfolio_leverage" in orch.ctx.risk_limits
        assert orch.ctx.risk_limits["max_portfolio_leverage"] == settings.risk.max_portfolio_leverage

    def test_ctx_has_empty_instruments(self):
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.BACKTEST)
        orch = Orchestrator.from_config(settings)
        assert orch.ctx.instruments == {}

    def test_ctx_has_portfolio_state(self):
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.core.interfaces import PortfolioState
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.BACKTEST)
        orch = Orchestrator.from_config(settings)
        assert isinstance(orch.ctx.portfolio_state, PortfolioState)


# ---------------------------------------------------------------------------
# CLI entry points preserved
# ---------------------------------------------------------------------------

class TestCLIPreserved:
    """The CLI module still works after the refactoring."""

    def test_cli_imports_from_main(self):
        """cli.py can import run from main."""
        from agentic_trading.cli import main as cli_main

        assert callable(cli_main)

    def test_cli_module_has_backtest_command(self):
        from agentic_trading import cli

        source = inspect.getsource(cli)
        assert "def backtest" in source

    def test_cli_module_has_paper_command(self):
        from agentic_trading import cli

        source = inspect.getsource(cli)
        assert "def paper" in source

    def test_cli_module_has_live_command(self):
        from agentic_trading import cli

        source = inspect.getsource(cli)
        assert "def live" in source

    def test_cli_module_has_walk_forward_command(self):
        from agentic_trading import cli

        source = inspect.getsource(cli)
        assert "def walk_forward" in source

    def test_cli_module_has_optimize_command(self):
        from agentic_trading import cli

        source = inspect.getsource(cli)
        assert "def optimize" in source


# ---------------------------------------------------------------------------
# Lifecycle integration
# ---------------------------------------------------------------------------

class TestLifecycleIntegration:
    """The bus start/stop lifecycle works through the Orchestrator."""

    @pytest.mark.asyncio
    async def test_bus_start_stop_via_orchestrator(self):
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.BACKTEST)
        orch = Orchestrator.from_config(settings)

        await orch.bus.start()
        assert orch.bus.is_running

        await orch.bus.stop()
        assert not orch.bus.is_running

    @pytest.mark.asyncio
    async def test_event_bus_accessible_via_legacy_bus(self):
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.BACKTEST)
        orch = Orchestrator.from_config(settings)

        event_bus = orch.bus.legacy_bus
        assert event_bus is not None

        # Should be the same object as ctx.event_bus
        assert event_bus is orch.ctx.event_bus

    @pytest.mark.asyncio
    async def test_full_orchestrator_start_stop(self):
        from agentic_trading.core.config import Settings
        from agentic_trading.core.enums import Mode
        from agentic_trading.orchestrator import Orchestrator

        settings = Settings(mode=Mode.BACKTEST)
        orch = Orchestrator.from_config(settings)

        await orch.start()
        assert orch.bus.is_running

        await orch.stop()
        assert not orch.bus.is_running
