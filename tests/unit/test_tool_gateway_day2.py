"""Day 2 tests: ToolGateway — sole adapter accessor.

Tests:
    - All 14 ToolName dispatch routes
    - Idempotency cache hit/miss
    - Rate limiting (sliding window)
    - Kill switch blocks mutating calls
    - Policy evaluator fail-closed
    - Approval service fail-closed
    - Audit log unavailable blocks all mutations
    - Read-only convenience method
    - Incident emission on failures
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.control_plane.action_types import (
    ActionScope,
    ApprovalDecision,
    ApprovalTier,
    CPPolicyDecision,
    MUTATING_TOOLS,
    ProposedAction,
    ToolCallResult,
    ToolName,
)
from agentic_trading.control_plane.audit_log import AuditLog
from agentic_trading.control_plane.tool_gateway import (
    AllowAllPolicy,
    AutoApproveService,
    ToolGateway,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Valid OrderIntent params (all required fields for Pydantic validation)
_VALID_INTENT = {
    "symbol": "BTC/USDT",
    "side": "buy",
    "qty": "1.0",
    "dedupe_key": "test_dedupe_001",
    "strategy_id": "trend",
    "exchange": "bybit",
    "order_type": "limit",
    "price": "50000",
}

_SUBMIT_PARAMS = {"intent": _VALID_INTENT}
_BATCH_PARAMS = {"intents": [_VALID_INTENT]}

# Simple mutating params that don't need Pydantic model construction
_CANCEL_PARAMS = {"order_id": "o1", "symbol": "BTC/USDT"}


def _make_adapter() -> MagicMock:
    """Create a mock exchange adapter with all methods."""
    adapter = AsyncMock()
    # Mutating methods return Pydantic-like mocks
    ack = MagicMock()
    ack.model_dump.return_value = {"order_id": "o1", "status": "NEW"}
    adapter.submit_order.return_value = ack
    adapter.cancel_order.return_value = ack
    adapter.cancel_all_orders.return_value = [ack]
    adapter.amend_order.return_value = ack
    adapter.batch_submit_orders.return_value = [ack]
    adapter.set_trading_stop.return_value = {"result": "ok"}
    adapter.set_leverage.return_value = {"result": "ok"}
    adapter.set_position_mode.return_value = {"result": "ok"}

    # Read methods
    pos = MagicMock()
    pos.model_dump.return_value = {"symbol": "BTC/USDT", "qty": "1.0"}
    adapter.get_positions.return_value = [pos]

    bal = MagicMock()
    bal.model_dump.return_value = {"coin": "USDT", "free": "10000"}
    adapter.get_balances.return_value = [bal]

    order = MagicMock()
    order.model_dump.return_value = {"order_id": "o2", "status": "NEW"}
    adapter.get_open_orders.return_value = [order]

    inst = MagicMock()
    inst.model_dump.return_value = {"symbol": "BTC/USDT", "tick_size": "0.01"}
    adapter.get_instrument.return_value = inst

    adapter.get_funding_rate.return_value = Decimal("0.0001")
    adapter.get_closed_pnl.return_value = [{"pnl": "100.0"}]

    return adapter


def _make_event_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


def _make_gateway(
    adapter: MagicMock | None = None,
    audit_log: AuditLog | None = None,
    event_bus: AsyncMock | None = None,
    policy_evaluator: object | None = None,
    approval_service: object | None = None,
    kill_switch_fn: object | None = None,
    rate_limits: dict[str, int] | None = None,
) -> ToolGateway:
    return ToolGateway(
        adapter=adapter or _make_adapter(),
        audit_log=audit_log or AuditLog(),
        event_bus=event_bus or _make_event_bus(),
        policy_evaluator=policy_evaluator,
        approval_service=approval_service,
        kill_switch_fn=kill_switch_fn,
        rate_limits=rate_limits,
    )


def _proposed(
    tool_name: ToolName = ToolName.CANCEL_ORDER,
    params: dict | None = None,
    idempotency_key: str = "",
    actor: str = "test_agent",
    strategy_id: str = "trend",
    symbol: str = "BTC/USDT",
) -> ProposedAction:
    return ProposedAction(
        tool_name=tool_name,
        scope=ActionScope(
            strategy_id=strategy_id,
            symbol=symbol,
            actor=actor,
        ),
        request_params=params or {},
        idempotency_key=idempotency_key,
    )


# ===========================================================================
# Dispatch — all 14 tool names route correctly
# ===========================================================================


class TestDispatchRouting:
    @pytest.mark.asyncio
    async def test_submit_order(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.SUBMIT_ORDER, _SUBMIT_PARAMS,
        ))
        assert result.success
        adapter.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert result.success
        adapter.cancel_order.assert_called_once_with("o1", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ALL_ORDERS,
            {"symbol": "BTC/USDT"},
        ))
        assert result.success
        assert "acks" in result.response

    @pytest.mark.asyncio
    async def test_amend_order(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.AMEND_ORDER,
            {"order_id": "o1", "symbol": "BTC/USDT", "qty": "2.0", "price": "50000"},
        ))
        assert result.success
        adapter.amend_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_submit_orders(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.BATCH_SUBMIT_ORDERS, _BATCH_PARAMS,
        ))
        assert result.success
        assert "acks" in result.response

    @pytest.mark.asyncio
    async def test_set_trading_stop(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.SET_TRADING_STOP,
            {"symbol": "BTC/USDT", "take_profit": "55000", "stop_loss": "45000"},
        ))
        assert result.success
        adapter.set_trading_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_leverage(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.SET_LEVERAGE,
            {"symbol": "BTC/USDT", "leverage": 10},
        ))
        assert result.success
        adapter.set_leverage.assert_called_once_with("BTC/USDT", 10)

    @pytest.mark.asyncio
    async def test_set_position_mode(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.SET_POSITION_MODE,
            {"symbol": "BTC/USDT", "mode": "one_way"},
        ))
        assert result.success
        adapter.set_position_mode.assert_called_once_with("BTC/USDT", "one_way")

    @pytest.mark.asyncio
    async def test_get_positions(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.GET_POSITIONS,
            {"symbol": "BTC/USDT"},
        ))
        assert result.success
        assert "positions" in result.response

    @pytest.mark.asyncio
    async def test_get_balances(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(ToolName.GET_BALANCES, {}))
        assert result.success
        assert "balances" in result.response

    @pytest.mark.asyncio
    async def test_get_open_orders(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(ToolName.GET_OPEN_ORDERS, {}))
        assert result.success
        assert "orders" in result.response

    @pytest.mark.asyncio
    async def test_get_instrument(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.GET_INSTRUMENT,
            {"symbol": "BTC/USDT"},
        ))
        assert result.success
        assert "symbol" in result.response

    @pytest.mark.asyncio
    async def test_get_funding_rate(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.GET_FUNDING_RATE,
            {"symbol": "BTC/USDT"},
        ))
        assert result.success
        assert "rate" in result.response

    @pytest.mark.asyncio
    async def test_get_closed_pnl(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.GET_CLOSED_PNL,
            {"symbol": "BTC/USDT", "limit": 10},
        ))
        assert result.success
        assert "entries" in result.response


# ===========================================================================
# Idempotency
# ===========================================================================


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_first_call_not_replay(self):
        gw = _make_gateway()
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
            idempotency_key="idem1",
        ))
        assert result.success
        assert not result.was_idempotent_replay

    @pytest.mark.asyncio
    async def test_second_call_is_replay(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)

        r1 = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS, idempotency_key="idem2",
        ))
        r2 = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS, idempotency_key="idem2",
        ))

        assert r1.success
        assert not r1.was_idempotent_replay
        assert r2.was_idempotent_replay
        # Adapter only called once
        assert adapter.cancel_order.call_count == 1

    @pytest.mark.asyncio
    async def test_different_keys_not_replay(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)

        r1 = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS, idempotency_key="key_a",
        ))
        r2 = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS, idempotency_key="key_b",
        ))

        assert not r1.was_idempotent_replay
        assert not r2.was_idempotent_replay
        assert adapter.cancel_order.call_count == 2

    @pytest.mark.asyncio
    async def test_no_key_no_caching(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)

        await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS, idempotency_key="",
        ))
        await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS, idempotency_key="",
        ))

        assert adapter.cancel_order.call_count == 2
        assert gw.idempotency_cache_size == 0

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        gw = _make_gateway()
        await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
            idempotency_key="idem_clear",
        ))
        assert gw.idempotency_cache_size == 1
        gw.clear_idempotency_cache()
        assert gw.idempotency_cache_size == 0

    @pytest.mark.asyncio
    async def test_idempotent_replay_preserves_response(self):
        """Replayed result has same response data as original."""
        gw = _make_gateway()
        r1 = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
            idempotency_key="idem_preserve",
        ))
        r2 = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
            idempotency_key="idem_preserve",
        ))
        assert r2.response == r1.response
        assert r2.action_id == r1.action_id


# ===========================================================================
# Rate Limiting
# ===========================================================================


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_under_limit_passes(self):
        gw = _make_gateway(rate_limits={"cancel_order": 5})
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert result.success

    @pytest.mark.asyncio
    async def test_over_limit_rejected(self):
        gw = _make_gateway(rate_limits={"cancel_order": 2})

        r1 = await gw.call(_proposed(ToolName.CANCEL_ORDER, _CANCEL_PARAMS))
        r2 = await gw.call(_proposed(ToolName.CANCEL_ORDER, _CANCEL_PARAMS))
        r3 = await gw.call(_proposed(ToolName.CANCEL_ORDER, _CANCEL_PARAMS))

        assert r1.success
        assert r2.success
        assert not r3.success
        assert "rate_limit_exceeded" in r3.error

    @pytest.mark.asyncio
    async def test_no_limit_allows_all(self):
        gw = _make_gateway(rate_limits={})

        for _ in range(10):
            result = await gw.call(_proposed(
                ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
            ))
            assert result.success

    @pytest.mark.asyncio
    async def test_different_tools_independent_limits(self):
        gw = _make_gateway(rate_limits={"cancel_order": 1, "set_leverage": 1})

        r1 = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        r2 = await gw.call(_proposed(
            ToolName.SET_LEVERAGE,
            {"symbol": "BTC/USDT", "leverage": 10},
        ))

        assert r1.success
        assert r2.success

    @pytest.mark.asyncio
    async def test_read_tools_also_rate_limited(self):
        gw = _make_gateway(rate_limits={"get_balances": 1})

        r1 = await gw.call(_proposed(ToolName.GET_BALANCES, {}))
        r2 = await gw.call(_proposed(ToolName.GET_BALANCES, {}))

        assert r1.success
        assert not r2.success
        assert "rate_limit_exceeded" in r2.error


# ===========================================================================
# Kill Switch
# ===========================================================================


class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_active_blocks_mutating(self):
        gw = _make_gateway(kill_switch_fn=lambda: True)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert not result.success
        assert "kill_switch_active" in result.error

    @pytest.mark.asyncio
    async def test_inactive_allows_mutating(self):
        gw = _make_gateway(kill_switch_fn=lambda: False)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert result.success

    @pytest.mark.asyncio
    async def test_no_kill_switch_allows_all(self):
        gw = _make_gateway(kill_switch_fn=None)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert result.success

    @pytest.mark.asyncio
    async def test_async_kill_switch(self):
        """Kill switch fn can be async."""
        async def async_kill():
            return True

        gw = _make_gateway(kill_switch_fn=async_kill)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert not result.success
        assert "kill_switch_active" in result.error

    @pytest.mark.asyncio
    async def test_kill_switch_does_not_block_reads(self):
        gw = _make_gateway(kill_switch_fn=lambda: True)
        result = await gw.call(_proposed(ToolName.GET_BALANCES, {}))
        assert result.success


# ===========================================================================
# Policy Evaluation (fail-closed)
# ===========================================================================


class TestPolicyFailClosed:
    @pytest.mark.asyncio
    async def test_policy_blocks_mutating(self):
        policy = MagicMock()
        policy.evaluate.return_value = CPPolicyDecision(
            action_id="a1",
            allowed=False,
            reasons=["max_notional_exceeded"],
        )

        gw = _make_gateway(policy_evaluator=policy)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))

        assert not result.success
        assert "policy_blocked" in result.error
        assert "max_notional_exceeded" in result.error

    @pytest.mark.asyncio
    async def test_policy_exception_blocks(self):
        """FAIL CLOSED: policy evaluator exception -> BLOCK."""
        policy = MagicMock()
        policy.evaluate.side_effect = RuntimeError("policy crashed")

        bus = _make_event_bus()
        gw = _make_gateway(policy_evaluator=policy, event_bus=bus)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))

        assert not result.success
        assert "policy_evaluator_error" in result.error
        bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_policy_not_called_for_reads(self):
        policy = MagicMock()
        gw = _make_gateway(policy_evaluator=policy)
        result = await gw.call(_proposed(ToolName.GET_BALANCES, {}))

        assert result.success
        policy.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_all_stub_passes(self):
        """Default AllowAllPolicy stub passes everything."""
        gw = _make_gateway()
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert result.success


# ===========================================================================
# Approval Service (fail-closed)
# ===========================================================================


class TestApprovalFailClosed:
    @pytest.mark.asyncio
    async def test_approval_denied_blocks(self):
        policy = MagicMock()
        policy.evaluate.return_value = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T2_APPROVE,
        )
        approval = AsyncMock()
        approval.request.return_value = ApprovalDecision(
            action_id="a1",
            approved=False,
            reason="operator_rejected",
        )

        gw = _make_gateway(policy_evaluator=policy, approval_service=approval)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))

        assert not result.success
        assert "approval_denied" in result.error

    @pytest.mark.asyncio
    async def test_approval_pending_returns_pending(self):
        policy = MagicMock()
        policy.evaluate.return_value = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T2_APPROVE,
        )
        approval = AsyncMock()
        approval.request.return_value = ApprovalDecision(
            action_id="a1",
            approved=False,
            pending_request_id="req_123",
        )

        gw = _make_gateway(policy_evaluator=policy, approval_service=approval)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))

        assert not result.success
        assert "pending_approval" in result.error
        assert result.response["pending_request_id"] == "req_123"

    @pytest.mark.asyncio
    async def test_approval_exception_blocks(self):
        """FAIL CLOSED: approval service exception -> BLOCK."""
        policy = MagicMock()
        policy.evaluate.return_value = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T2_APPROVE,
        )
        approval = AsyncMock()
        approval.request.side_effect = RuntimeError("approval service crashed")

        bus = _make_event_bus()
        gw = _make_gateway(
            policy_evaluator=policy, approval_service=approval, event_bus=bus,
        )
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))

        assert not result.success
        assert "approval_service_error" in result.error
        bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_t0_skips_approval(self):
        """T0_AUTONOMOUS tier skips approval service entirely."""
        policy = MagicMock()
        policy.evaluate.return_value = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T0_AUTONOMOUS,
        )
        approval = AsyncMock()

        gw = _make_gateway(policy_evaluator=policy, approval_service=approval)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))

        assert result.success
        approval.request.assert_not_called()


# ===========================================================================
# Audit Log (fail-closed)
# ===========================================================================


class TestAuditFailClosed:
    @pytest.mark.asyncio
    async def test_audit_unavailable_blocks_mutation(self):
        """FAIL CLOSED: audit log unavailable -> BLOCK all mutations."""
        audit = AuditLog()
        audit.set_available(False)

        bus = _make_event_bus()
        gw = _make_gateway(audit_log=audit, event_bus=bus)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))

        assert not result.success
        assert "audit_log_unavailable" in result.error
        bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_audit_unavailable_blocks_reads_too(self):
        """Audit log is mandatory for ALL calls, including reads."""
        audit = AuditLog()
        audit.set_available(False)

        gw = _make_gateway(audit_log=audit)
        result = await gw.call(_proposed(ToolName.GET_BALANCES, {}))

        assert not result.success
        assert "audit_log_unavailable" in result.error

    @pytest.mark.asyncio
    async def test_audit_records_pre_and_post(self):
        audit = AuditLog()
        gw = _make_gateway(audit_log=audit)

        proposed = _proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        )
        await gw.call(proposed)

        entries = audit.read(proposed.correlation_id)
        assert len(entries) >= 2
        event_types = [e.event_type for e in entries]
        assert "tool_call_pre_execution" in event_types
        assert "tool_call_recorded" in event_types


# ===========================================================================
# Adapter dispatch errors
# ===========================================================================


class TestDispatchErrors:
    @pytest.mark.asyncio
    async def test_adapter_exception_returns_failure(self):
        adapter = _make_adapter()
        adapter.cancel_order.side_effect = RuntimeError("exchange down")

        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))

        assert not result.success
        assert "exchange down" in result.error
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_failed_dispatch_still_audited(self):
        adapter = _make_adapter()
        adapter.get_balances.side_effect = RuntimeError("timeout")
        audit = AuditLog()

        gw = _make_gateway(adapter=adapter, audit_log=audit)
        proposed = _proposed(ToolName.GET_BALANCES, {})
        result = await gw.call(proposed)

        assert not result.success
        entries = audit.read(proposed.correlation_id)
        recorded = [e for e in entries if e.event_type == "tool_call_recorded"]
        assert len(recorded) == 1
        assert recorded[0].payload["success"] is False


# ===========================================================================
# Read convenience method
# ===========================================================================


class TestReadConvenience:
    @pytest.mark.asyncio
    async def test_read_returns_response_dict(self):
        gw = _make_gateway()
        result = await gw.read(ToolName.GET_BALANCES)
        assert "balances" in result

    @pytest.mark.asyncio
    async def test_read_rejects_mutating_tool(self):
        gw = _make_gateway()
        with pytest.raises(ValueError, match="mutating"):
            await gw.read(ToolName.SUBMIT_ORDER)

    @pytest.mark.asyncio
    async def test_read_raises_on_dispatch_failure(self):
        adapter = _make_adapter()
        adapter.get_balances.side_effect = RuntimeError("fail")
        gw = _make_gateway(adapter=adapter)

        with pytest.raises(RuntimeError, match="Read failed"):
            await gw.read(ToolName.GET_BALANCES)

    @pytest.mark.asyncio
    async def test_read_passes_params(self):
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        await gw.read(ToolName.GET_POSITIONS, params={"symbol": "ETH/USDT"})
        adapter.get_positions.assert_called_once_with("ETH/USDT")


# ===========================================================================
# Response hashing and latency
# ===========================================================================


class TestResponseIntegrity:
    @pytest.mark.asyncio
    async def test_response_hash_computed(self):
        gw = _make_gateway()
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert result.success
        assert len(result.response_hash) == 16

    @pytest.mark.asyncio
    async def test_latency_recorded(self):
        gw = _make_gateway()
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_failed_result_no_response_hash(self):
        adapter = _make_adapter()
        adapter.cancel_order.side_effect = RuntimeError("boom")
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.CANCEL_ORDER, _CANCEL_PARAMS,
        ))
        assert not result.success
        assert result.response_hash == ""


# ===========================================================================
# Stub implementations
# ===========================================================================


class TestStubs:
    def test_allow_all_policy(self):
        policy = AllowAllPolicy()
        action = _proposed(ToolName.CANCEL_ORDER)
        decision = policy.evaluate(action)
        assert decision.allowed
        assert decision.tier == ApprovalTier.T0_AUTONOMOUS
        assert decision.sizing_multiplier == 1.0

    @pytest.mark.asyncio
    async def test_auto_approve_service(self):
        service = AutoApproveService()
        decision = CPPolicyDecision(
            action_id="a1", allowed=True, tier=ApprovalTier.T2_APPROVE,
        )
        action = _proposed(ToolName.CANCEL_ORDER)
        approval = await service.request(decision, action)
        assert approval.approved
        assert "stub_auto" in approval.decided_by[0]


# ===========================================================================
# Full flow integration (happy path)
# ===========================================================================


class TestFullFlow:
    @pytest.mark.asyncio
    async def test_mutating_happy_path(self):
        """Full flow: propose -> policy -> approval -> audit -> dispatch."""
        adapter = _make_adapter()
        audit = AuditLog()
        bus = _make_event_bus()

        gw = ToolGateway(
            adapter=adapter,
            audit_log=audit,
            event_bus=bus,
        )

        proposed = _proposed(
            ToolName.SUBMIT_ORDER, _SUBMIT_PARAMS,
            idempotency_key="full_flow_1",
        )

        result = await gw.call(proposed)

        assert result.success
        assert not result.was_idempotent_replay
        assert result.response_hash
        assert result.latency_ms >= 0
        assert audit.entry_count >= 2
        adapter.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_happy_path(self):
        """Read flow: propose -> audit -> dispatch (no policy/approval)."""
        adapter = _make_adapter()
        audit = AuditLog()

        gw = ToolGateway(
            adapter=adapter,
            audit_log=audit,
            event_bus=_make_event_bus(),
        )

        result = await gw.read(ToolName.GET_POSITIONS, {"symbol": "BTC/USDT"})
        assert "positions" in result
        assert audit.entry_count >= 2

    @pytest.mark.asyncio
    async def test_submit_order_with_valid_intent(self):
        """SUBMIT_ORDER dispatches correctly with full OrderIntent data."""
        adapter = _make_adapter()
        gw = _make_gateway(adapter=adapter)
        result = await gw.call(_proposed(
            ToolName.SUBMIT_ORDER,
            _SUBMIT_PARAMS,
            idempotency_key="submit_test",
        ))

        assert result.success
        adapter.submit_order.assert_called_once()
        # Verify idempotency cached
        assert gw.idempotency_cache_size == 1
