"""Tests for the ExecutionGateway facade and risk/ → execution/risk/ shim layer.

Tests cover:
1. ExecutionGateway.from_config construction and wiring
2. Component accessors
3. Lifecycle (start/stop)
4. Backward-compat shim imports (risk.* → execution.risk.*)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_trading.core.config import RiskConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_adapter():
    """Create a mock exchange adapter."""
    adapter = MagicMock()
    adapter.submit_order = AsyncMock()
    return adapter


def _make_mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    return bus


def _make_gateway(**kwargs):
    """Build an ExecutionGateway via the factory with mocks."""
    from agentic_trading.execution.gateway import ExecutionGateway

    return ExecutionGateway.from_config(
        adapter=kwargs.get("adapter", _make_mock_adapter()),
        event_bus=kwargs.get("event_bus", _make_mock_event_bus()),
        risk_config=kwargs.get("risk_config", RiskConfig()),
    )


# ---------------------------------------------------------------------------
# ExecutionGateway factory
# ---------------------------------------------------------------------------

class TestExecutionGatewayFactory:
    """ExecutionGateway.from_config produces a fully wired instance."""

    def test_from_config_creates_gateway(self):
        gw = _make_gateway()
        assert gw is not None
        assert gw.engine is not None
        assert gw.risk_manager is not None

    def test_from_config_creates_quality_tracker(self):
        gw = _make_gateway()
        assert gw.quality_tracker is not None

    def test_from_config_wires_risk_into_engine(self):
        gw = _make_gateway()
        # The engine's _risk_manager should be the same RiskManager
        assert gw.engine._risk_manager is gw.risk_manager

    def test_from_config_with_custom_risk_config(self):
        cfg = RiskConfig(max_single_position_pct=0.20)
        gw = _make_gateway(risk_config=cfg)
        assert gw.risk_manager._config.max_single_position_pct == 0.20

    def test_order_manager_accessor(self):
        from agentic_trading.execution.order_manager import OrderManager

        gw = _make_gateway()
        assert isinstance(gw.order_manager, OrderManager)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestExecutionGatewayLifecycle:
    """ExecutionGateway lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_bus(self):
        bus = _make_mock_event_bus()
        gw = _make_gateway(event_bus=bus)
        await gw.start()
        assert bus.subscribe.await_count >= 2  # execution + system topics

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        gw = _make_gateway()
        await gw.start()
        await gw.stop()
        # Should not raise


# ---------------------------------------------------------------------------
# Risk passthrough
# ---------------------------------------------------------------------------

class TestRiskPassthrough:
    """ExecutionGateway delegates risk operations."""

    def test_update_instruments(self):
        from agentic_trading.core.enums import Exchange, InstrumentType
        from agentic_trading.core.models import Instrument

        gw = _make_gateway()
        instruments = {
            "BTC/USDT": Instrument(
                symbol="BTC/USDT",
                exchange=Exchange.BYBIT,
                instrument_type=InstrumentType.PERP,
                base="BTC",
                quote="USDT",
                price_precision=2,
                qty_precision=3,
            ),
        }
        gw.update_instruments(instruments)
        assert gw.risk_manager._instruments == instruments


# ---------------------------------------------------------------------------
# Component accessors
# ---------------------------------------------------------------------------

class TestComponentAccessors:
    """ExecutionGateway exposes sub-components."""

    def test_engine_accessor(self):
        from agentic_trading.execution.engine import ExecutionEngine

        gw = _make_gateway()
        assert isinstance(gw.engine, ExecutionEngine)

    def test_risk_manager_accessor(self):
        from agentic_trading.execution.risk.manager import RiskManager

        gw = _make_gateway()
        assert isinstance(gw.risk_manager, RiskManager)


# ---------------------------------------------------------------------------
# Backward-compat shim imports (risk → execution.risk)
# ---------------------------------------------------------------------------

class TestRiskShimImports:
    """Verify all shim imports from risk.* resolve correctly."""

    def test_risk_init_imports(self):
        from agentic_trading.risk import (
            AlertEngine,
            CircuitBreaker,
            CircuitBreakerManager,
            DrawdownMonitor,
            ExposureSnapshot,
            ExposureTracker,
            KillSwitch,
            PostTradeChecker,
            PreTradeChecker,
            RiskManager,
            RiskMetrics,
        )
        # Verify these are the same classes as in execution.risk.*
        from agentic_trading.execution.risk import (
            RiskManager as CanonicalRM,
            KillSwitch as CanonicalKS,
        )
        assert RiskManager is CanonicalRM
        assert KillSwitch is CanonicalKS

    def test_risk_manager_shim(self):
        from agentic_trading.risk.manager import RiskManager as ShimRM
        from agentic_trading.execution.risk.manager import (
            RiskManager as CanonicalRM,
        )
        assert ShimRM is CanonicalRM

    def test_pre_trade_shim(self):
        from agentic_trading.risk.pre_trade import (
            PreTradeChecker as ShimPTC,
        )
        from agentic_trading.execution.risk.pre_trade import (
            PreTradeChecker as CanonicalPTC,
        )
        assert ShimPTC is CanonicalPTC

    def test_post_trade_shim(self):
        from agentic_trading.risk.post_trade import (
            PostTradeChecker as ShimPTC,
        )
        from agentic_trading.execution.risk.post_trade import (
            PostTradeChecker as CanonicalPTC,
        )
        assert ShimPTC is CanonicalPTC

    def test_circuit_breakers_shim(self):
        from agentic_trading.risk.circuit_breakers import (
            CircuitBreaker as ShimCB,
            CircuitBreakerManager as ShimCBM,
        )
        from agentic_trading.execution.risk.circuit_breakers import (
            CircuitBreaker as CanonicalCB,
            CircuitBreakerManager as CanonicalCBM,
        )
        assert ShimCB is CanonicalCB
        assert ShimCBM is CanonicalCBM

    def test_kill_switch_shim(self):
        from agentic_trading.risk.kill_switch import KillSwitch as ShimKS
        from agentic_trading.execution.risk.kill_switch import (
            KillSwitch as CanonicalKS,
        )
        assert ShimKS is CanonicalKS

    def test_drawdown_shim(self):
        from agentic_trading.risk.drawdown import (
            DrawdownMonitor as ShimDM,
        )
        from agentic_trading.execution.risk.drawdown import (
            DrawdownMonitor as CanonicalDM,
        )
        assert ShimDM is CanonicalDM

    def test_exposure_shim(self):
        from agentic_trading.risk.exposure import (
            ExposureTracker as ShimET,
            ExposureSnapshot as ShimES,
        )
        from agentic_trading.execution.risk.exposure import (
            ExposureTracker as CanonicalET,
            ExposureSnapshot as CanonicalES,
        )
        assert ShimET is CanonicalET
        assert ShimES is CanonicalES

    def test_alerts_shim(self):
        from agentic_trading.risk.alerts import AlertEngine as ShimAE
        from agentic_trading.execution.risk.alerts import (
            AlertEngine as CanonicalAE,
        )
        assert ShimAE is CanonicalAE

    def test_var_es_shim(self):
        from agentic_trading.risk.var_es import RiskMetrics as ShimRM
        from agentic_trading.execution.risk.var_es import (
            RiskMetrics as CanonicalRM,
        )
        assert ShimRM is CanonicalRM
