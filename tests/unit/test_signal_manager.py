"""Tests for the SignalManager facade.

Tests cover:
1. SignalManager.from_config construction and wiring
2. Component accessors
3. Lifecycle (start)
4. Delegated operations — signal collection, target generation
5. Delegated operations — allocation with correlation
6. Delegated operations — intent conversion
7. Delegated operations — correlation tracking
8. Sizing multiplier
9. Package-level import
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_trading.core.enums import Exchange, OrderType, Side, SignalDirection, Timeframe
from agentic_trading.core.events import Signal, TargetPosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    strategy_id: str = "trend_following",
    symbol: str = "BTC/USDT",
    direction: SignalDirection = SignalDirection.LONG,
    confidence: float = 0.8,
) -> Signal:
    """Create a minimal Signal for testing."""
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        timeframe=Timeframe.M1,
    )


def _make_target(
    strategy_id: str = "trend_following",
    symbol: str = "BTC/USDT",
    qty: float = 0.01,
    side: Side = Side.BUY,
) -> TargetPosition:
    """Create a minimal TargetPosition for testing."""
    return TargetPosition(
        strategy_id=strategy_id,
        symbol=symbol,
        target_qty=Decimal(str(qty)),
        side=side,
    )


def _make_mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    return bus


def _make_mock_feature_engine():
    """Create a mock feature engine with get_buffer."""
    fe = MagicMock()
    fe.get_buffer = MagicMock(return_value=[])
    return fe


# ---------------------------------------------------------------------------
# Factory — minimal (no event bus)
# ---------------------------------------------------------------------------

class TestSignalManagerFactoryMinimal:
    """from_config without event bus creates a minimal manager."""

    def test_from_config_no_bus(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        assert mgr is not None

    def test_no_bus_no_runner(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        assert mgr.runner is None

    def test_has_portfolio_manager(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        assert mgr.portfolio_manager is not None

    def test_has_allocator(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        assert mgr.allocator is not None

    def test_has_correlation_analyzer(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        assert mgr.correlation_analyzer is not None

    def test_strategies_empty_without_runner(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        assert mgr.strategies == []

    def test_signal_count_zero_without_runner(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        assert mgr.signal_count == 0


# ---------------------------------------------------------------------------
# Factory — with event bus and feature engine
# ---------------------------------------------------------------------------

class TestSignalManagerFactoryWithBus:
    """from_config with event bus creates a runner."""

    def test_from_config_with_bus(self):
        from agentic_trading.signal.manager import SignalManager

        bus = _make_mock_event_bus()
        fe = _make_mock_feature_engine()
        mgr = SignalManager.from_config(
            feature_engine=fe,
            event_bus=bus,
        )
        assert mgr.runner is not None

    def test_runner_has_strategies_empty_by_default(self):
        from agentic_trading.signal.manager import SignalManager

        bus = _make_mock_event_bus()
        fe = _make_mock_feature_engine()
        mgr = SignalManager.from_config(
            feature_engine=fe,
            event_bus=bus,
        )
        assert mgr.strategies == []

    def test_custom_sizing_params(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config(
            max_position_pct=0.05,
            max_gross_exposure_pct=0.5,
            sizing_multiplier=0.75,
        )
        assert mgr.portfolio_manager._max_position_pct == 0.05
        assert mgr.portfolio_manager._max_gross_exposure == 0.5
        assert mgr.portfolio_manager._sizing_multiplier == 0.75

    def test_custom_allocator_params(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config(
            max_single_position_pct=0.15,
            max_correlated_exposure_pct=0.30,
        )
        assert mgr.allocator._max_single == 0.15
        assert mgr.allocator._max_correlated == 0.30

    def test_custom_correlation_params(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config(
            correlation_lookback=120,
            correlation_threshold=0.8,
        )
        assert mgr.correlation_analyzer._lookback == 120
        assert mgr.correlation_analyzer._threshold == 0.8


# ---------------------------------------------------------------------------
# Component accessors & types
# ---------------------------------------------------------------------------

class TestComponentAccessors:
    """SignalManager exposes sub-components with correct types."""

    def test_portfolio_manager_type(self):
        from agentic_trading.signal.manager import SignalManager
        from agentic_trading.signal.portfolio.manager import PortfolioManager

        mgr = SignalManager.from_config()
        assert isinstance(mgr.portfolio_manager, PortfolioManager)

    def test_allocator_type(self):
        from agentic_trading.signal.manager import SignalManager
        from agentic_trading.signal.portfolio.allocator import PortfolioAllocator

        mgr = SignalManager.from_config()
        assert isinstance(mgr.allocator, PortfolioAllocator)

    def test_correlation_analyzer_type(self):
        from agentic_trading.signal.manager import SignalManager
        from agentic_trading.signal.portfolio.correlation_risk import (
            CorrelationRiskAnalyzer,
        )

        mgr = SignalManager.from_config()
        assert isinstance(mgr.correlation_analyzer, CorrelationRiskAnalyzer)

    def test_runner_type(self):
        from agentic_trading.signal.manager import SignalManager
        from agentic_trading.signal.runner import StrategyRunner

        bus = _make_mock_event_bus()
        fe = _make_mock_feature_engine()
        mgr = SignalManager.from_config(
            feature_engine=fe,
            event_bus=bus,
        )
        assert isinstance(mgr.runner, StrategyRunner)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestSignalManagerLifecycle:
    """SignalManager lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_without_runner(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        await mgr.start(ctx=MagicMock())
        # Should not raise

    @pytest.mark.asyncio
    async def test_start_with_runner(self):
        from agentic_trading.signal.manager import SignalManager

        bus = _make_mock_event_bus()
        fe = _make_mock_feature_engine()
        mgr = SignalManager.from_config(
            feature_engine=fe,
            event_bus=bus,
        )
        await mgr.start(ctx=MagicMock())
        bus.subscribe.assert_called()


# ---------------------------------------------------------------------------
# Delegated operations — signal collection
# ---------------------------------------------------------------------------

class TestDelegatedSignalCollection:
    """SignalManager delegates signal collection to PortfolioManager."""

    def test_on_signal_collects(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        sig = _make_signal()
        mgr.on_signal(sig)
        assert len(mgr.portfolio_manager._pending_signals) == 1

    def test_on_signal_multiple(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        mgr.on_signal(_make_signal(strategy_id="s1"))
        mgr.on_signal(_make_signal(strategy_id="s2"))
        assert len(mgr.portfolio_manager._pending_signals) == 2


# ---------------------------------------------------------------------------
# Delegated operations — target generation
# ---------------------------------------------------------------------------

class TestDelegatedTargetGeneration:
    """SignalManager delegates to PortfolioManager for targets."""

    def test_generate_targets_empty(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        ctx = MagicMock()
        targets = mgr.generate_targets(ctx, capital=100_000)
        assert targets == []


# ---------------------------------------------------------------------------
# Delegated operations — intent conversion
# ---------------------------------------------------------------------------

class TestDelegatedIntentConversion:
    """SignalManager.build_intents converts targets to intents."""

    def test_build_intents_empty(self):
        from agentic_trading.signal.manager import SignalManager

        intents = SignalManager.build_intents(
            targets=[],
            exchange=Exchange.BYBIT,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert intents == []

    def test_build_intents_with_target(self):
        from agentic_trading.signal.manager import SignalManager

        target = _make_target(qty=0.01)
        intents = SignalManager.build_intents(
            targets=[target],
            exchange=Exchange.BYBIT,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert len(intents) == 1
        assert intents[0].symbol == "BTC/USDT"

    def test_build_intents_with_order_type(self):
        from agentic_trading.signal.manager import SignalManager

        target = _make_target(qty=0.01)
        intents = SignalManager.build_intents(
            targets=[target],
            exchange=Exchange.BYBIT,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            order_type=OrderType.LIMIT,
        )
        assert len(intents) == 1


# ---------------------------------------------------------------------------
# Delegated operations — correlation
# ---------------------------------------------------------------------------

class TestDelegatedCorrelation:
    """SignalManager delegates correlation tracking."""

    def test_update_returns(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        mgr.update_returns("BTC/USDT", 0.01)
        mgr.update_returns("BTC/USDT", -0.005)
        # Should not raise, data accumulates

    def test_find_clusters_empty(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        clusters = mgr.find_correlation_clusters()
        assert isinstance(clusters, list)

    def test_find_clusters_without_analyzer(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager(
            runner=None,
            portfolio_manager=MagicMock(),
            correlation_analyzer=None,
        )
        clusters = mgr.find_correlation_clusters()
        assert clusters == []


# ---------------------------------------------------------------------------
# Sizing multiplier
# ---------------------------------------------------------------------------

class TestSizingMultiplier:
    """SignalManager delegates sizing multiplier updates."""

    def test_set_sizing_multiplier(self):
        from agentic_trading.signal.manager import SignalManager

        mgr = SignalManager.from_config()
        mgr.set_sizing_multiplier(0.5)
        assert mgr.portfolio_manager._sizing_multiplier == 0.5


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageImport:
    """SignalManager is importable from package __init__."""

    def test_import_from_package(self):
        from agentic_trading.signal import SignalManager

        assert SignalManager is not None

    def test_import_from_module(self):
        from agentic_trading.signal.manager import SignalManager

        assert SignalManager is not None

    def test_import_identity(self):
        from agentic_trading.signal import SignalManager as PkgSM
        from agentic_trading.signal.manager import SignalManager as ModSM

        assert PkgSM is ModSM
