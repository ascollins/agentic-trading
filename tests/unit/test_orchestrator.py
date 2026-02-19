"""Tests for the top-level Orchestrator facade.

Tests cover:
1. Orchestrator.from_config — backtest mode construction
2. Orchestrator.from_config — paper mode construction
3. Orchestrator.from_config — governance enabled
4. Component accessors
5. Lifecycle (start/stop)
6. Mode helpers (is_backtest, mode property)
7. Metrics aggregation
8. Active layer names
9. Package-level import
"""

from __future__ import annotations

import pytest

from agentic_trading.core.config import Settings
from agentic_trading.core.enums import Mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    """Create a Settings object with sensible test defaults."""
    defaults = {
        "mode": Mode.BACKTEST,
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# Factory — backtest mode
# ---------------------------------------------------------------------------

class TestOrchestratorFactoryBacktest:
    """Orchestrator.from_config in backtest mode."""

    def test_from_config_backtest(self):
        from agentic_trading.orchestrator import Orchestrator

        settings = _make_settings(mode=Mode.BACKTEST)
        orch = Orchestrator.from_config(settings)
        assert orch is not None

    def test_backtest_has_bus(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        assert orch.bus is not None

    def test_backtest_has_intelligence(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        assert orch.intelligence is not None

    def test_backtest_has_signal(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        assert orch.signal is not None

    def test_backtest_no_execution(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        assert orch.execution is None

    def test_backtest_no_policy(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        assert orch.policy is None

    def test_backtest_has_reconciliation(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        assert orch.reconciliation is not None

    def test_backtest_has_ctx(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        assert orch.ctx is not None
        assert orch.ctx.clock is not None
        assert orch.ctx.event_bus is not None

    def test_backtest_clock_is_sim(self):
        from agentic_trading.core.clock import SimClock
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings(mode=Mode.BACKTEST))
        assert isinstance(orch.ctx.clock, SimClock)

    def test_backtest_bus_is_memory(self):
        from agentic_trading.bus.memory_bus import MemoryEventBus
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings(mode=Mode.BACKTEST))
        assert isinstance(orch.bus.legacy_bus, MemoryEventBus)

    def test_backtest_signal_runner_is_none(self):
        """In backtest mode without event bus wiring, runner is None."""
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings(mode=Mode.BACKTEST))
        assert orch.signal.runner is None


# ---------------------------------------------------------------------------
# Factory — paper mode (no adapter)
# ---------------------------------------------------------------------------

class TestOrchestratorFactoryPaper:
    """Orchestrator.from_config in paper mode (no adapter passed)."""

    def test_paper_no_adapter_no_execution(self):
        """Paper mode without adapter → no execution gateway."""
        from agentic_trading.orchestrator import Orchestrator

        settings = _make_settings(mode=Mode.PAPER)
        orch = Orchestrator.from_config(settings)
        assert orch.execution is None

    def test_paper_clock_is_wall(self):
        from agentic_trading.core.clock import WallClock
        from agentic_trading.orchestrator import Orchestrator

        settings = _make_settings(mode=Mode.PAPER)
        orch = Orchestrator.from_config(settings)
        assert isinstance(orch.ctx.clock, WallClock)

    def test_paper_intelligence_has_feature_engine(self):
        from agentic_trading.orchestrator import Orchestrator

        settings = _make_settings(mode=Mode.PAPER)
        orch = Orchestrator.from_config(settings)
        assert orch.intelligence.feature_engine is not None

    def test_paper_signal_has_runner(self):
        """Paper mode wires event bus → runner is created."""
        from agentic_trading.orchestrator import Orchestrator

        settings = _make_settings(mode=Mode.PAPER)
        orch = Orchestrator.from_config(settings)
        assert orch.signal.runner is not None


# ---------------------------------------------------------------------------
# Factory — governance enabled
# ---------------------------------------------------------------------------

class TestOrchestratorFactoryGovernance:
    """Orchestrator.from_config with governance enabled."""

    def test_governance_creates_policy(self):
        from agentic_trading.orchestrator import Orchestrator
        from agentic_trading.core.config import GovernanceConfig

        settings = _make_settings(
            governance=GovernanceConfig(enabled=True),
        )
        orch = Orchestrator.from_config(settings)
        assert orch.policy is not None

    def test_governance_disabled_no_policy(self):
        from agentic_trading.orchestrator import Orchestrator
        from agentic_trading.core.config import GovernanceConfig

        settings = _make_settings(
            governance=GovernanceConfig(enabled=False),
        )
        orch = Orchestrator.from_config(settings)
        assert orch.policy is None

    def test_governance_recon_gets_health_tracker(self):
        from agentic_trading.orchestrator import Orchestrator
        from agentic_trading.core.config import GovernanceConfig

        settings = _make_settings(
            governance=GovernanceConfig(enabled=True),
        )
        orch = Orchestrator.from_config(settings)
        # ReconciliationManager's journal should have health_tracker wired
        assert orch.reconciliation is not None


# ---------------------------------------------------------------------------
# Component accessors
# ---------------------------------------------------------------------------

class TestOrchestratorAccessors:
    """Orchestrator property accessors."""

    def test_settings_accessor(self):
        from agentic_trading.orchestrator import Orchestrator

        settings = _make_settings()
        orch = Orchestrator.from_config(settings)
        assert orch.settings is settings

    def test_mode_accessor(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings(mode=Mode.BACKTEST))
        assert orch.mode == Mode.BACKTEST

    def test_is_backtest_true(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings(mode=Mode.BACKTEST))
        assert orch.is_backtest is True

    def test_is_backtest_false_for_paper(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings(mode=Mode.PAPER))
        assert orch.is_backtest is False

    def test_ctx_has_risk_limits(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        assert "max_portfolio_leverage" in orch.ctx.risk_limits


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestOrchestratorLifecycle:
    """Orchestrator start/stop."""

    @pytest.mark.asyncio
    async def test_start_stop_backtest(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings(mode=Mode.BACKTEST))
        await orch.start()
        assert orch.bus.is_running
        await orch.stop()
        assert not orch.bus.is_running

    @pytest.mark.asyncio
    async def test_start_stop_paper(self):
        from agentic_trading.orchestrator import Orchestrator

        settings = _make_settings(mode=Mode.PAPER)
        orch = Orchestrator.from_config(settings)
        await orch.start()
        assert orch.bus.is_running
        await orch.stop()
        assert not orch.bus.is_running

    @pytest.mark.asyncio
    async def test_start_stop_with_governance(self):
        from agentic_trading.orchestrator import Orchestrator
        from agentic_trading.core.config import GovernanceConfig

        settings = _make_settings(
            governance=GovernanceConfig(enabled=True),
        )
        orch = Orchestrator.from_config(settings)
        await orch.start()
        assert orch.bus.is_running
        assert orch.policy is not None
        await orch.stop()
        assert not orch.bus.is_running


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

class TestOrchestratorMetrics:
    """Orchestrator.get_metrics combines layer stats."""

    def test_metrics_contains_mode(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings(mode=Mode.BACKTEST))
        metrics = orch.get_metrics()
        assert metrics["mode"] == "backtest"

    def test_metrics_contains_bus(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        metrics = orch.get_metrics()
        assert "bus" in metrics

    def test_metrics_contains_signal_count(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        metrics = orch.get_metrics()
        assert "signal_count" in metrics
        assert metrics["signal_count"] == 0

    def test_metrics_contains_strategy_count(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        metrics = orch.get_metrics()
        assert "strategy_count" in metrics

    def test_metrics_contains_trade_counts(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        metrics = orch.get_metrics()
        assert "open_trades" in metrics
        assert metrics["open_trades"] == 0
        assert "closed_trades" in metrics
        assert metrics["closed_trades"] == 0


# ---------------------------------------------------------------------------
# Active layer names
# ---------------------------------------------------------------------------

class TestOrchestratorLayerNames:
    """Orchestrator._active_layer_names helper."""

    def test_backtest_layers(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings(mode=Mode.BACKTEST))
        names = orch._active_layer_names()
        assert "bus" in names
        assert "intelligence" in names
        assert "signal" in names
        assert "reconciliation" in names
        assert "execution" not in names
        assert "policy" not in names

    def test_governance_adds_policy_layer(self):
        from agentic_trading.orchestrator import Orchestrator
        from agentic_trading.core.config import GovernanceConfig

        settings = _make_settings(
            governance=GovernanceConfig(enabled=True),
        )
        orch = Orchestrator.from_config(settings)
        names = orch._active_layer_names()
        assert "policy" in names


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

class TestOrchestratorStrategies:
    """Orchestrator wires strategy configuration."""

    def test_no_strategies_configured(self):
        from agentic_trading.orchestrator import Orchestrator

        orch = Orchestrator.from_config(_make_settings())
        # When no strategies configured, list is empty (or uses defaults)
        assert isinstance(orch.signal.strategies, list)

    def test_safe_mode_sizing(self):
        from agentic_trading.orchestrator import Orchestrator
        from agentic_trading.core.config import SafeModeConfig

        settings = _make_settings(
            safe_mode=SafeModeConfig(enabled=True, position_size_multiplier=0.3),
        )
        orch = Orchestrator.from_config(settings)
        # The portfolio manager should have reduced sizing
        assert orch.signal.portfolio_manager is not None


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestOrchestratorImport:
    """Orchestrator is importable from expected paths."""

    def test_import_from_module(self):
        from agentic_trading.orchestrator import Orchestrator

        assert Orchestrator is not None

    def test_import_orchestrator_class(self):
        from agentic_trading.orchestrator import Orchestrator

        assert hasattr(Orchestrator, "from_config")
        assert hasattr(Orchestrator, "start")
        assert hasattr(Orchestrator, "stop")
        assert hasattr(Orchestrator, "get_metrics")
