"""Integration test: control plane pipeline end-to-end.

Verifies the full flow:
    ProposedAction -> CPPolicyEvaluator -> CPApprovalService -> AuditLog -> ToolGateway -> Adapter
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from agentic_trading.control_plane.action_types import (
    ActionScope,
    ApprovalTier,
    CPPolicyDecision,
    ProposedAction,
    ToolCallResult,
    ToolName,
)
from agentic_trading.control_plane.approval_service import CPApprovalService
from agentic_trading.control_plane.audit_log import AuditLog
from agentic_trading.control_plane.policy_evaluator import (
    CPPolicyEvaluator,
    build_default_evaluator,
)
from agentic_trading.control_plane.tool_gateway import (
    AllowAllPolicy,
    AutoApproveService,
    ToolGateway,
)
from agentic_trading.core.enums import (
    Exchange,
    GovernanceAction,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from agentic_trading.core.events import OrderAck, OrderIntent
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.policy.engine import PolicyEngine
from agentic_trading.policy.models import (
    Operator,
    PolicyMode,
    PolicyRule,
    PolicySet,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubAdapter:
    """Minimal exchange adapter that records submissions and returns FILLED."""

    def __init__(self) -> None:
        self.submitted: list[OrderIntent] = []

    async def submit_order(self, intent: OrderIntent) -> OrderAck:
        self.submitted.append(intent)
        return OrderAck(
            order_id="stub-order-001",
            client_order_id=intent.dedupe_key,
            symbol=intent.symbol,
            exchange=intent.exchange,
            status=OrderStatus.FILLED,
        )

    async def get_positions(self, symbol: str | None = None) -> list:
        return []

    async def get_balances(self) -> list:
        return []

    async def cancel_order(self, order_id: str, symbol: str) -> OrderAck:
        return OrderAck(
            order_id=order_id,
            client_order_id="",
            symbol=symbol,
            exchange=Exchange.BYBIT,
            status=OrderStatus.CANCELLED,
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_submit_action(
    *,
    symbol: str = "BTC/USDT",
    strategy_id: str = "trend_following",
    actor: str = "exec-agent",
    actor_role: str = "",
    required_role: str | None = None,
    idempotency_key: str = "",
    order_notional_usd: float = 10_000.0,
) -> ProposedAction:
    """Build a SUBMIT_ORDER ProposedAction."""
    return ProposedAction(
        tool_name=ToolName.SUBMIT_ORDER,
        scope=ActionScope(
            strategy_id=strategy_id,
            symbol=symbol,
            actor=actor,
            actor_role=actor_role,
        ),
        request_params={
            "intent": {
                "symbol": symbol,
                "exchange": "bybit",
                "side": "buy",
                "order_type": "market",
                "qty": "0.1",
                "time_in_force": "GTC",
                "dedupe_key": "test-dedupe-001",
                "strategy_id": strategy_id,
            },
            "order_notional_usd": order_notional_usd,
        },
        idempotency_key=idempotency_key,
        required_role=required_role,
    )


def _make_read_action(
    tool_name: ToolName = ToolName.GET_POSITIONS,
    symbol: str = "BTC/USDT",
) -> ProposedAction:
    """Build a read-only ProposedAction."""
    return ProposedAction(
        tool_name=tool_name,
        scope=ActionScope(symbol=symbol, actor="system"),
        request_params={"symbol": symbol},
    )


def _make_blocking_policy_set(
    *,
    field: str = "order_notional_usd",
    threshold: float = 50_000.0,
    action: GovernanceAction = GovernanceAction.BLOCK,
) -> PolicySet:
    """Create a PolicySet with one rule that blocks when field > threshold.

    Rule semantics: the condition is the PASS check.  So LE means
    "field must be <= threshold"; if value exceeds, the rule FAILS.
    """
    return PolicySet(
        set_id="test_block_policy",
        name="Test Block Policy",
        version=1,
        mode=PolicyMode.ENFORCED,
        action=action,
        rules=[
            PolicyRule(
                rule_id="max_notional",
                name="Maximum Notional",
                field=field,
                operator=Operator.LE,
                threshold=threshold,
                action=action,
                description=f"Require {field} <= {threshold}",
            ),
        ],
    )


def _make_gateway(
    adapter: _StubAdapter | None = None,
    audit_log: AuditLog | None = None,
    event_bus: MemoryEventBus | None = None,
    policy_evaluator=None,
    approval_service=None,
    kill_switch_fn=None,
    rate_limits: dict[str, int] | None = None,
) -> tuple[ToolGateway, _StubAdapter, AuditLog, MemoryEventBus]:
    """Build a fully wired ToolGateway for testing."""
    adapter = adapter or _StubAdapter()
    audit_log = audit_log or AuditLog()
    event_bus = event_bus or MemoryEventBus()
    gw = ToolGateway(
        adapter=adapter,
        audit_log=audit_log,
        event_bus=event_bus,
        policy_evaluator=policy_evaluator,
        approval_service=approval_service,
        kill_switch_fn=kill_switch_fn,
        rate_limits=rate_limits,
    )
    return gw, adapter, audit_log, event_bus


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestControlPlanePipeline:
    """End-to-end integration tests for the ToolGateway pipeline."""

    @pytest.mark.asyncio
    async def test_t0_autonomous_order_flows_through(self):
        """AllowAll policy + auto-approve -> order submitted to adapter."""
        gw, adapter, audit_log, _ = _make_gateway()

        action = _make_submit_action()
        result = await gw.call(action)

        assert result.success is True
        assert len(adapter.submitted) == 1
        assert adapter.submitted[0].symbol == "BTC/USDT"
        # Audit log should have pre + post entries
        assert audit_log.entry_count >= 2

    @pytest.mark.asyncio
    async def test_read_only_call_bypasses_policy_and_approval(self):
        """GET_POSITIONS skips policy/approval but still goes through audit."""
        gw, adapter, audit_log, _ = _make_gateway()

        action = _make_read_action(ToolName.GET_POSITIONS)
        result = await gw.call(action)

        assert result.success is True
        assert result.response == {"positions": []}
        # Adapter submit not called
        assert len(adapter.submitted) == 0
        # Audit still records the call
        assert audit_log.entry_count >= 1

    @pytest.mark.asyncio
    async def test_policy_block_prevents_execution(self):
        """PolicySet that blocks high notional prevents order submission."""
        engine = PolicyEngine()
        # Block when order_notional_usd > 5000
        engine.register(_make_blocking_policy_set(threshold=5_000.0))
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        gw, adapter, _, _ = _make_gateway(policy_evaluator=evaluator)

        # Action with notional 10_000 > 5_000 threshold
        action = _make_submit_action(order_notional_usd=10_000.0)
        result = await gw.call(action)

        assert result.success is False
        assert "policy_blocked" in (result.error or "")
        assert len(adapter.submitted) == 0

    @pytest.mark.asyncio
    async def test_policy_reduce_size_allows_with_multiplier(self):
        """REDUCE_SIZE action allows execution but signals reduced sizing."""
        engine = PolicyEngine()
        # Reduce when notional > 5000 (GovernanceAction.REDUCE_SIZE)
        ps = _make_blocking_policy_set(
            threshold=5_000.0,
            action=GovernanceAction.REDUCE_SIZE,
        )
        engine.register(ps)
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        gw, adapter, _, _ = _make_gateway(policy_evaluator=evaluator)

        action = _make_submit_action(order_notional_usd=10_000.0)
        result = await gw.call(action)

        # REDUCE_SIZE allows but the policy decision has sizing_multiplier < 1.0
        # The ToolGateway still proceeds since allowed=True
        assert result.success is True
        assert len(adapter.submitted) == 1

    @pytest.mark.asyncio
    async def test_t1_notify_auto_approves_and_emits_notification(self):
        """T1_NOTIFY tier auto-approves and publishes SystemHealth notification."""
        bus = MemoryEventBus()
        captured_events: list = []

        async def capture(event):
            captured_events.append(event)

        await bus.subscribe("system", "test_group", capture)

        # Build a policy evaluator that returns T1_NOTIFY tier
        engine = PolicyEngine()
        ps = _make_blocking_policy_set(
            threshold=5_000.0,
            action=GovernanceAction.REDUCE_SIZE,
        )
        engine.register(ps)
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        from agentic_trading.policy.approval_manager import ApprovalManager

        approval_mgr = ApprovalManager(auto_approve_l1=True)
        approval_svc = CPApprovalService(
            approval_manager=approval_mgr,
            audit_log=AuditLog(),
            event_bus=bus,
        )

        gw, adapter, _, _ = _make_gateway(
            event_bus=bus,
            policy_evaluator=evaluator,
            approval_service=approval_svc,
        )

        action = _make_submit_action(order_notional_usd=10_000.0)
        result = await gw.call(action)

        assert result.success is True
        # T1 emits a SystemHealth notification on the "system" topic
        system_events = [e for e in captured_events if hasattr(e, "component")]
        assert any("approval_notify" in e.component for e in system_events)

    @pytest.mark.asyncio
    async def test_t2_approve_returns_pending(self):
        """T2_APPROVE tier returns pending_approval with request ID."""
        engine = PolicyEngine()
        # BLOCK triggers T2_APPROVE tier in the evaluator
        ps = _make_blocking_policy_set(
            threshold=5_000.0,
            action=GovernanceAction.BLOCK,
        )
        engine.register(ps)
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        # Note: BLOCK action returns allowed=False from evaluator,
        # so ToolGateway will block at step 3 before reaching approval.
        # To test T2 pending, we need allowed=True with tier=T2_APPROVE.
        # Use a custom evaluator that returns T2 tier with allowed=True.
        class _T2PolicyEvaluator:
            def evaluate(self, proposed: ProposedAction) -> CPPolicyDecision:
                return CPPolicyDecision(
                    action_id=proposed.action_id,
                    correlation_id=proposed.correlation_id,
                    allowed=True,
                    tier=ApprovalTier.T2_APPROVE,
                    sizing_multiplier=1.0,
                    reasons=["size_threshold_exceeded"],
                    policy_set_version="test_v1",
                    context_snapshot={"order_notional_usd": 100_000},
                )

        from agentic_trading.policy.approval_manager import ApprovalManager

        approval_mgr = ApprovalManager(auto_approve_l1=False)
        approval_svc = CPApprovalService(
            approval_manager=approval_mgr,
            audit_log=AuditLog(),
        )

        gw, adapter, _, _ = _make_gateway(
            policy_evaluator=_T2PolicyEvaluator(),
            approval_service=approval_svc,
        )

        action = _make_submit_action(order_notional_usd=100_000.0)
        result = await gw.call(action)

        # Should be pending, not executed
        assert result.success is False
        assert "pending_approval" in (result.error or "")
        assert result.response.get("pending_request_id") is not None
        assert len(adapter.submitted) == 0

    @pytest.mark.asyncio
    async def test_audit_log_unavailable_blocks_mutating_calls(self):
        """When AuditLog.set_available(False), ToolGateway rejects mutating calls."""
        audit_log = AuditLog()
        audit_log.set_available(False)

        gw, adapter, _, _ = _make_gateway(audit_log=audit_log)

        action = _make_submit_action()
        result = await gw.call(action)

        assert result.success is False
        assert "audit_log_unavailable" in (result.error or "")
        assert len(adapter.submitted) == 0

    @pytest.mark.asyncio
    async def test_audit_entries_track_full_correlation(self):
        """After a successful call, AuditLog.read(correlation_id) returns entries."""
        gw, adapter, audit_log, _ = _make_gateway()

        action = _make_submit_action()
        result = await gw.call(action)

        assert result.success is True

        # Read audit trail by correlation_id
        entries = audit_log.read(action.correlation_id)
        assert len(entries) >= 2  # pre-execution + post-execution
        event_types = [e.event_type for e in entries]
        assert "tool_call_pre_execution" in event_types
        assert "tool_call_recorded" in event_types

    @pytest.mark.asyncio
    async def test_idempotency_key_returns_cached_result(self):
        """Second call with same idempotency_key returns cached result."""
        gw, adapter, _, _ = _make_gateway()

        action = _make_submit_action(idempotency_key="idem-001")

        result1 = await gw.call(action)
        assert result1.success is True
        assert result1.was_idempotent_replay is False
        assert len(adapter.submitted) == 1

        # Second call with same idempotency key
        action2 = _make_submit_action(idempotency_key="idem-001")
        result2 = await gw.call(action2)
        assert result2.success is True
        assert result2.was_idempotent_replay is True
        # Adapter should NOT have been called again
        assert len(adapter.submitted) == 1

    @pytest.mark.asyncio
    async def test_information_barrier_blocks_wrong_role(self):
        """ProposedAction with required_role != actor_role is rejected."""
        gw, adapter, _, _ = _make_gateway()

        action = _make_submit_action(
            actor_role="trader",
            required_role="risk",  # trader cannot perform risk-only action
        )
        result = await gw.call(action)

        assert result.success is False
        assert "information_barrier" in (result.error or "")
        assert len(adapter.submitted) == 0
