"""Tests for ExecutionEngine in control-plane mode (ToolGateway).

Verifies:
    - CP mode is activated when tool_gateway is provided
    - Orders are routed through ToolGateway.call()
    - OrderLifecycleManager tracks FSM per order
    - Policy blocks → lifecycle BLOCKED
    - Pending approval → lifecycle AWAITING_APPROVAL
    - Submission failures → lifecycle SUBMIT_FAILED
    - Immediate fills → lifecycle COMPLETE → POST_TRADE → TERMINAL
    - Pre-trade risk rejection → lifecycle BLOCKED
    - Deduplication still works
    - Legacy mode is preserved when tool_gateway is None
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_trading.control_plane.action_types import (
    ApprovalTier,
    CPPolicyDecision,
    ToolCallResult,
    ToolName,
)
from agentic_trading.control_plane.state_machine import OrderState
from agentic_trading.core.enums import OrderStatus, Side
from agentic_trading.core.events import (
    OrderAck,
    OrderIntent,
    RiskCheckResult,
)
from agentic_trading.core.interfaces import PortfolioState
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.execution.engine import ExecutionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(dedupe_key: str = "dk1") -> OrderIntent:
    """Create a valid OrderIntent for testing."""
    return OrderIntent(
        symbol="BTC/USDT",
        side=Side.BUY,
        qty=Decimal("0.1"),
        price=Decimal("50000"),
        dedupe_key=dedupe_key,
        strategy_id="trend",
        exchange="bybit",
        order_type="limit",
    )


def _make_success_result(order_id: str = "o1") -> ToolCallResult:
    """ToolGateway result for a successful order submission."""
    return ToolCallResult(
        tool_name=ToolName.SUBMIT_ORDER,
        success=True,
        response={
            "order_id": order_id,
            "client_order_id": "dk1",
            "symbol": "BTC/USDT",
            "exchange": "bybit",
            "status": "filled",
            "message": "",
        },
    )


def _make_submitted_result(order_id: str = "o1") -> ToolCallResult:
    """ToolGateway result for a submitted (not yet filled) order."""
    return ToolCallResult(
        tool_name=ToolName.SUBMIT_ORDER,
        success=True,
        response={
            "order_id": order_id,
            "client_order_id": "dk1",
            "symbol": "BTC/USDT",
            "exchange": "bybit",
            "status": "submitted",
            "message": "",
        },
    )


def _make_policy_blocked_result() -> ToolCallResult:
    return ToolCallResult(
        tool_name=ToolName.SUBMIT_ORDER,
        success=False,
        error="policy_blocked: max_notional_exceeded",
    )


def _make_pending_approval_result(request_id: str = "req1") -> ToolCallResult:
    return ToolCallResult(
        tool_name=ToolName.SUBMIT_ORDER,
        success=False,
        error=f"pending_approval:{request_id}",
        response={"pending_request_id": request_id},
    )


def _make_submit_failed_result() -> ToolCallResult:
    return ToolCallResult(
        tool_name=ToolName.SUBMIT_ORDER,
        success=False,
        error="exchange_error: connection_timeout",
    )


class _AlwaysPassRisk:
    def pre_trade_check(self, intent, state):
        return RiskCheckResult(
            passed=True,
            check_name="all_pass",
            reason="ok",
        )

    def post_trade_check(self, fill, state):
        return RiskCheckResult(
            passed=True,
            check_name="all_pass",
            reason="ok",
        )


class _AlwaysFailRisk:
    def pre_trade_check(self, intent, state):
        return RiskCheckResult(
            passed=False,
            check_name="max_position",
            reason="Position too large",
        )

    def post_trade_check(self, fill, state):
        return RiskCheckResult(
            passed=True,
            check_name="all_pass",
            reason="ok",
        )


def _make_engine(
    tool_gateway: MagicMock | None = None,
    risk_manager: object | None = None,
) -> tuple[ExecutionEngine, MemoryEventBus, MagicMock]:
    """Build an engine in CP mode with a mock ToolGateway."""
    bus = MemoryEventBus()
    adapter = MagicMock()  # Should NOT be called in CP mode
    gw = tool_gateway or MagicMock()
    risk = risk_manager or _AlwaysPassRisk()

    engine = ExecutionEngine(
        adapter=adapter,
        event_bus=bus,
        risk_manager=risk,
        tool_gateway=gw,
    )
    return engine, bus, gw


# ===========================================================================
# Activation
# ===========================================================================


class TestCPModeActivation:
    def test_uses_control_plane_when_gateway_provided(self):
        engine, _, _ = _make_engine()
        assert engine.uses_control_plane is True
        assert engine.lifecycle_manager is not None

    def test_legacy_mode_without_gateway(self):
        bus = MemoryEventBus()
        adapter = MagicMock()
        engine = ExecutionEngine(
            adapter=adapter,
            event_bus=bus,
            risk_manager=_AlwaysPassRisk(),
        )
        assert engine.uses_control_plane is False
        assert engine.lifecycle_manager is None


# ===========================================================================
# Happy path: successful submission
# ===========================================================================


class TestCPHappyPath:
    @pytest.mark.asyncio
    async def test_successful_submitted_order(self):
        """Order flows through CP: INTENT → PREFLIGHT → SUBMITTING → SUBMITTED → MONITORING."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_submitted_result())
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        intent = _make_intent()
        ack = await engine.handle_intent(intent)

        assert ack is not None
        assert ack.status == OrderStatus.SUBMITTED

        # Verify ToolGateway was called
        gw.call.assert_called_once()
        proposed = gw.call.call_args[0][0]
        assert proposed.tool_name == ToolName.SUBMIT_ORDER
        assert proposed.scope.strategy_id == "trend"
        assert proposed.scope.symbol == "BTC/USDT"
        assert proposed.idempotency_key == "dk1"

        # Verify lifecycle FSM
        lc = engine.lifecycle_manager.get("dk1")
        assert lc is not None
        assert lc.state == OrderState.MONITORING

    @pytest.mark.asyncio
    async def test_immediate_fill_completes_lifecycle(self):
        """When adapter returns FILLED, lifecycle goes through COMPLETE → TERMINAL."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_success_result())
        gw.read = AsyncMock(return_value={"positions": []})
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        ack = await engine.handle_intent(_make_intent())

        assert ack is not None
        assert ack.status == OrderStatus.FILLED

        # Lifecycle should be TERMINAL after fill
        lc = engine.lifecycle_manager.get("dk1")
        assert lc is not None
        assert lc.state == OrderState.TERMINAL
        assert lc.is_terminal
        assert len(lc.fills) == 1

    @pytest.mark.asyncio
    async def test_ack_published_on_success(self):
        """OrderAck is published to the execution topic."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_submitted_result())
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        captured = []
        await bus.subscribe("execution", "test", lambda e: captured.append(e))

        await engine.handle_intent(_make_intent())

        acks = [e for e in captured if isinstance(e, OrderAck)]
        assert len(acks) >= 1
        assert acks[0].status == OrderStatus.SUBMITTED


# ===========================================================================
# Policy block
# ===========================================================================


class TestCPPolicyBlock:
    @pytest.mark.asyncio
    async def test_policy_blocked_transitions_to_blocked(self):
        """When ToolGateway returns policy_blocked, lifecycle → BLOCKED."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_policy_blocked_result())
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        ack = await engine.handle_intent(_make_intent())

        assert ack is not None
        assert ack.status == OrderStatus.REJECTED
        assert "policy_blocked" in ack.message

        lc = engine.lifecycle_manager.get("dk1")
        assert lc is not None
        assert lc.state == OrderState.BLOCKED
        assert lc.is_terminal

    @pytest.mark.asyncio
    async def test_policy_blocked_ack_published(self):
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_policy_blocked_result())
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        captured = []
        await bus.subscribe("execution", "test", lambda e: captured.append(e))

        await engine.handle_intent(_make_intent())

        acks = [e for e in captured if isinstance(e, OrderAck)]
        assert len(acks) == 1
        assert acks[0].status == OrderStatus.REJECTED


# ===========================================================================
# Pending approval
# ===========================================================================


class TestCPPendingApproval:
    @pytest.mark.asyncio
    async def test_pending_approval_transitions_to_awaiting(self):
        """When ToolGateway returns pending_approval, lifecycle → AWAITING_APPROVAL."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_pending_approval_result("req42"))
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        ack = await engine.handle_intent(_make_intent())

        assert ack is not None
        assert ack.status == OrderStatus.PENDING

        lc = engine.lifecycle_manager.get("dk1")
        assert lc is not None
        assert lc.state == OrderState.AWAITING_APPROVAL
        assert not lc.is_terminal


# ===========================================================================
# Submission failure
# ===========================================================================


class TestCPSubmitFailed:
    @pytest.mark.asyncio
    async def test_submit_failure_transitions_to_submit_failed(self):
        """When ToolGateway returns a non-policy error, lifecycle → SUBMIT_FAILED."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_submit_failed_result())
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        ack = await engine.handle_intent(_make_intent())

        assert ack is not None
        assert ack.status == OrderStatus.REJECTED

        lc = engine.lifecycle_manager.get("dk1")
        assert lc is not None
        assert lc.state == OrderState.SUBMIT_FAILED
        assert lc.is_terminal


# ===========================================================================
# Pre-trade risk rejection
# ===========================================================================


class TestCPRiskRejection:
    @pytest.mark.asyncio
    async def test_risk_rejection_transitions_to_blocked(self):
        """Pre-trade risk failure blocks order at PREFLIGHT_POLICY → BLOCKED."""
        gw = AsyncMock()
        engine, bus, _ = _make_engine(
            tool_gateway=gw,
            risk_manager=_AlwaysFailRisk(),
        )
        await engine.start()

        ack = await engine.handle_intent(_make_intent())

        assert ack is not None
        assert ack.status == OrderStatus.REJECTED
        assert "Risk check failed" in ack.message

        # ToolGateway should NOT have been called
        gw.call.assert_not_called()

        lc = engine.lifecycle_manager.get("dk1")
        assert lc is not None
        assert lc.state == OrderState.BLOCKED
        assert lc.is_terminal
        assert "risk_check_failed" in lc.error


# ===========================================================================
# Deduplication
# ===========================================================================


class TestCPDeduplication:
    @pytest.mark.asyncio
    async def test_duplicate_raises(self):
        """Submitting the same dedupe_key twice raises DuplicateOrderError."""
        from agentic_trading.core.errors import DuplicateOrderError

        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_submitted_result())
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        await engine.handle_intent(_make_intent("dup1"))

        with pytest.raises(DuplicateOrderError):
            await engine.handle_intent(_make_intent("dup1"))


# ===========================================================================
# Lifecycle introspection
# ===========================================================================


class TestCPLifecycleIntrospection:
    @pytest.mark.asyncio
    async def test_lifecycle_manager_tracks_all_orders(self):
        """Multiple orders are tracked in the lifecycle manager."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_submitted_result())
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        await engine.handle_intent(_make_intent("a1"))
        await engine.handle_intent(_make_intent("a2"))

        assert engine.lifecycle_manager.count == 2
        assert engine.lifecycle_manager.active_count == 2

    @pytest.mark.asyncio
    async def test_lifecycle_tool_result_stored(self):
        """ToolCallResult is stored on the lifecycle object."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_submitted_result())
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        await engine.handle_intent(_make_intent())

        lc = engine.lifecycle_manager.get("dk1")
        assert lc.tool_result is not None
        assert lc.tool_result.success is True

    @pytest.mark.asyncio
    async def test_lifecycle_history_records_transitions(self):
        """Lifecycle records all state transitions."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_submitted_result())
        engine, bus, _ = _make_engine(tool_gateway=gw)
        await engine.start()

        await engine.handle_intent(_make_intent())

        lc = engine.lifecycle_manager.get("dk1")
        # INTENT -> PREFLIGHT -> SUBMITTING -> SUBMITTED -> MONITORING
        assert lc.transition_count == 4
        assert lc.history[0] == (OrderState.INTENT_RECEIVED, OrderState.PREFLIGHT_POLICY, lc.history[0][2])
        assert lc.history[-1][1] == OrderState.MONITORING


# ===========================================================================
# Integration: adapter NOT called in CP mode
# ===========================================================================


class TestCPAdapterIsolation:
    @pytest.mark.asyncio
    async def test_adapter_submit_not_called(self):
        """In CP mode, self._adapter.submit_order() is never called."""
        gw = AsyncMock()
        gw.call = AsyncMock(return_value=_make_submitted_result())

        bus = MemoryEventBus()
        adapter = MagicMock()
        engine = ExecutionEngine(
            adapter=adapter,
            event_bus=bus,
            risk_manager=_AlwaysPassRisk(),
            tool_gateway=gw,
        )
        await engine.start()

        await engine.handle_intent(_make_intent())

        # adapter.submit_order should NOT have been called
        adapter.submit_order.assert_not_called()
