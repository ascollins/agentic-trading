"""Day 3 tests: CPPolicyEvaluator + CPApprovalService.

Acceptance tests:
    A1: ProposedAction with valid context → CPPolicyDecision (pass)
    A2: ProposedAction violating policy → CPPolicyDecision (block)
    A3: CPPolicyDecision with T2 tier → ApprovalService creates pending request
    A4: CPPolicyDecision with T0 tier → ApprovalService auto-approves

Additional tests:
    - PolicyEvaluator fail-closed on exception
    - Tier override for protective tools (cancel → T0)
    - Sizing multiplier propagation
    - Shadow violations tracked but allowed
    - ApprovalService T1 notify
    - ApprovalService T3 dual-approve
    - ApprovalService fail-closed on exception
    - Integration: evaluator + service end-to-end
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.control_plane.action_types import (
    ActionScope,
    ApprovalDecision as CPApprovalDecision,
    ApprovalTier,
    CPPolicyDecision,
    ProposedAction,
    ToolName,
)
from agentic_trading.control_plane.audit_log import AuditLog
from agentic_trading.control_plane.approval_service import CPApprovalService
from agentic_trading.control_plane.policy_evaluator import (
    CPPolicyEvaluator,
    build_default_evaluator,
)
from agentic_trading.core.enums import GovernanceAction
from agentic_trading.governance.approval_manager import ApprovalManager
from agentic_trading.governance.approval_models import (
    ApprovalRule,
    ApprovalStatus,
    ApprovalTrigger,
    EscalationLevel,
)
from agentic_trading.governance.policy_engine import PolicyEngine
from agentic_trading.governance.policy_models import (
    Operator,
    PolicyMode,
    PolicyRule,
    PolicySet,
    PolicyType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy_set(
    set_id: str = "test_risk",
    mode: PolicyMode = PolicyMode.ENFORCED,
    rules: list[PolicyRule] | None = None,
) -> PolicySet:
    return PolicySet(
        set_id=set_id,
        name=f"Test {set_id}",
        version=1,
        mode=mode,
        rules=rules or [],
    )


def _max_notional_rule(threshold: float = 100_000.0) -> PolicyRule:
    return PolicyRule(
        rule_id="max_notional",
        name="Max Notional",
        field="order_notional_usd",
        operator=Operator.LE,
        threshold=threshold,
        action=GovernanceAction.BLOCK,
        policy_type=PolicyType.RISK_LIMIT,
    )


def _max_leverage_rule(threshold: float = 5.0) -> PolicyRule:
    return PolicyRule(
        rule_id="max_leverage",
        name="Max Leverage",
        field="projected_leverage",
        operator=Operator.LE,
        threshold=threshold,
        action=GovernanceAction.BLOCK,
        policy_type=PolicyType.RISK_LIMIT,
    )


def _reduce_size_rule(threshold: float = 0.5) -> PolicyRule:
    return PolicyRule(
        rule_id="correlated_exposure",
        name="Correlated Exposure",
        field="correlated_exposure_pct",
        operator=Operator.LE,
        threshold=threshold,
        action=GovernanceAction.REDUCE_SIZE,
        policy_type=PolicyType.RISK_LIMIT,
    )


def _proposed(
    tool_name: ToolName = ToolName.SUBMIT_ORDER,
    params: dict | None = None,
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
    )


def _make_engine_with_rules(rules: list[PolicyRule]) -> PolicyEngine:
    engine = PolicyEngine()
    ps = _make_policy_set(rules=rules)
    engine.register(ps)
    return engine


# ===========================================================================
# A1: Proposed action with valid context -> CPPolicyDecision (pass)
# ===========================================================================


class TestA1PolicyPass:
    def test_valid_context_passes_all_rules(self):
        """A1: ProposedAction with context satisfying all rules → allowed."""
        engine = _make_engine_with_rules([
            _max_notional_rule(100_000),
            _max_leverage_rule(5.0),
        ])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        proposed = _proposed(params={
            "order_notional_usd": 50_000,
            "projected_leverage": 2.0,
        })
        decision = evaluator.evaluate(proposed)

        assert decision.allowed
        assert decision.sizing_multiplier == 1.0
        assert decision.tier == ApprovalTier.T0_AUTONOMOUS
        assert not decision.failed_rules

    def test_no_policy_sets_registered(self):
        """No policy sets → allowed with default tier."""
        engine = PolicyEngine()
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        decision = evaluator.evaluate(_proposed())
        assert decision.allowed
        assert "no_policy_sets_registered" in decision.reasons

    def test_context_includes_scope_fields(self):
        """Context should include strategy_id, symbol, exchange from scope."""
        engine = PolicyEngine()
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        proposed = _proposed(strategy_id="mean_rev", symbol="ETH/USDT")
        decision = evaluator.evaluate(proposed)

        assert decision.context_snapshot["strategy_id"] == "mean_rev"
        assert decision.context_snapshot["symbol"] == "ETH/USDT"

    def test_snapshot_hash_computed(self):
        """Decision includes a snapshot hash for replay verification."""
        engine = _make_engine_with_rules([_max_notional_rule(100_000)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        decision = evaluator.evaluate(_proposed(
            params={"order_notional_usd": 1000},
        ))
        assert decision.snapshot_hash
        assert len(decision.snapshot_hash) == 16

    def test_policy_set_version_tracked(self):
        """Decision tracks which policy set versions were evaluated."""
        engine = _make_engine_with_rules([_max_notional_rule(100_000)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        decision = evaluator.evaluate(_proposed(
            params={"order_notional_usd": 1000},
        ))
        assert "test_risk:v1" in decision.policy_set_version


# ===========================================================================
# A2: Proposed action violating policy -> CPPolicyDecision (block)
# ===========================================================================


class TestA2PolicyBlock:
    def test_notional_exceeded_blocks(self):
        """A2: order_notional_usd > threshold → BLOCK, not allowed."""
        engine = _make_engine_with_rules([_max_notional_rule(50_000)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        proposed = _proposed(params={"order_notional_usd": 100_000})
        decision = evaluator.evaluate(proposed)

        assert not decision.allowed
        assert decision.sizing_multiplier == 0.0
        assert "max_notional" in decision.failed_rules[0]

    def test_leverage_exceeded_blocks(self):
        """Leverage violation → BLOCK."""
        engine = _make_engine_with_rules([_max_leverage_rule(3.0)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        proposed = _proposed(params={"projected_leverage": 5.0})
        decision = evaluator.evaluate(proposed)

        assert not decision.allowed
        assert decision.tier == ApprovalTier.T2_APPROVE

    def test_reduce_size_allowed_with_reduced_multiplier(self):
        """REDUCE_SIZE action → allowed but with sizing < 1.0."""
        engine = _make_engine_with_rules([_reduce_size_rule(0.3)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        proposed = _proposed(params={"correlated_exposure_pct": 0.5})
        decision = evaluator.evaluate(proposed)

        assert decision.allowed  # REDUCE_SIZE is not a block
        assert decision.sizing_multiplier < 1.0
        assert decision.tier == ApprovalTier.T1_NOTIFY

    def test_most_severe_action_wins(self):
        """Multiple violations: most severe action determines decision."""
        engine = _make_engine_with_rules([
            _reduce_size_rule(0.3),  # REDUCE_SIZE
            _max_notional_rule(50_000),  # BLOCK (more severe)
        ])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        proposed = _proposed(params={
            "correlated_exposure_pct": 0.5,
            "order_notional_usd": 100_000,
        })
        decision = evaluator.evaluate(proposed)

        assert not decision.allowed  # BLOCK wins over REDUCE_SIZE
        assert len(decision.failed_rules) == 2

    def test_missing_field_fails_closed(self):
        """Missing context field → rule fails → BLOCK (B10 fix)."""
        engine = _make_engine_with_rules([_max_notional_rule(50_000)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        # No order_notional_usd in context
        proposed = _proposed(params={"something_else": 42})
        decision = evaluator.evaluate(proposed)

        assert not decision.allowed
        assert any("required_field_missing" in r for r in decision.reasons)


# ===========================================================================
# Shadow mode
# ===========================================================================


class TestShadowMode:
    def test_shadow_violations_tracked_but_allowed(self):
        """Shadow mode: violations logged but action still allowed."""
        engine = PolicyEngine()
        ps = _make_policy_set(
            mode=PolicyMode.SHADOW,
            rules=[_max_notional_rule(50_000)],
        )
        engine.register(ps)
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        proposed = _proposed(params={"order_notional_usd": 100_000})
        decision = evaluator.evaluate(proposed)

        assert decision.allowed
        assert len(decision.shadow_violations) > 0


# ===========================================================================
# Tier overrides (protective tools → T0 fast path)
# ===========================================================================


class TestTierOverrides:
    def test_cancel_order_always_t0(self):
        """Cancel orders get T0 regardless of policy failures."""
        engine = _make_engine_with_rules([_max_notional_rule(50_000)])
        evaluator = CPPolicyEvaluator(
            policy_engine=engine,
            tier_overrides={ToolName.CANCEL_ORDER: ApprovalTier.T0_AUTONOMOUS},
        )

        proposed = _proposed(
            tool_name=ToolName.CANCEL_ORDER,
            params={"order_notional_usd": 100_000},
        )
        decision = evaluator.evaluate(proposed)

        # Still blocked by policy, but tier is forced T0
        assert not decision.allowed
        assert decision.tier == ApprovalTier.T0_AUTONOMOUS

    def test_set_trading_stop_always_t0(self):
        """TP/SL orders get T0 (design decision D1)."""
        evaluator = build_default_evaluator()

        proposed = _proposed(
            tool_name=ToolName.SET_TRADING_STOP,
            params={"symbol": "BTC/USDT"},
        )
        decision = evaluator.evaluate(proposed)

        assert decision.allowed
        assert decision.tier == ApprovalTier.T0_AUTONOMOUS

    def test_build_default_evaluator_has_protective_overrides(self):
        """build_default_evaluator configures cancel/TP/SL as T0."""
        evaluator = build_default_evaluator()
        overrides = evaluator.tier_overrides
        assert overrides[ToolName.CANCEL_ORDER] == ApprovalTier.T0_AUTONOMOUS
        assert overrides[ToolName.CANCEL_ALL_ORDERS] == ApprovalTier.T0_AUTONOMOUS
        assert overrides[ToolName.SET_TRADING_STOP] == ApprovalTier.T0_AUTONOMOUS


# ===========================================================================
# PolicyEvaluator fail-closed
# ===========================================================================


class TestPolicyEvaluatorFailClosed:
    def test_engine_exception_blocks(self):
        """PolicyEngine exception → BLOCK (fail-closed)."""
        engine = MagicMock(spec=PolicyEngine)
        engine.registered_sets = ["test_set"]
        engine.evaluate_all.side_effect = RuntimeError("engine crashed")

        evaluator = CPPolicyEvaluator(policy_engine=engine)
        decision = evaluator.evaluate(_proposed())

        assert not decision.allowed
        assert "policy_evaluator_internal_error" in decision.reasons[0]

    def test_context_builder_exception_blocks(self):
        """Custom context builder exception → BLOCK."""
        engine = PolicyEngine()

        def bad_builder(proposed, ctx):
            raise ValueError("builder crashed")

        evaluator = CPPolicyEvaluator(
            policy_engine=engine,
            context_builder=bad_builder,
        )
        decision = evaluator.evaluate(_proposed())

        assert not decision.allowed

    def test_deterministic_same_input_same_output(self):
        """Same input always produces same decision (determinism)."""
        engine = _make_engine_with_rules([_max_notional_rule(50_000)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        proposed = _proposed(params={"order_notional_usd": 75_000})
        d1 = evaluator.evaluate(proposed)
        d2 = evaluator.evaluate(proposed)

        assert d1.allowed == d2.allowed
        assert d1.snapshot_hash == d2.snapshot_hash
        assert d1.failed_rules == d2.failed_rules
        assert d1.sizing_multiplier == d2.sizing_multiplier


# ===========================================================================
# A3: CPPolicyDecision with T2 tier → pending approval request
# ===========================================================================


class TestA3ApprovalPending:
    @pytest.mark.asyncio
    async def test_t2_creates_pending_request(self):
        """A3: T2_APPROVE → hold for human, return pending."""
        manager = ApprovalManager(auto_approve_l1=True)
        service = CPApprovalService(approval_manager=manager)

        policy_decision = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T2_APPROVE,
            reasons=["max_notional_exceeded"],
            context_snapshot={"order_notional_usd": 100_000},
        )
        proposed = _proposed()

        result = await service.request(policy_decision, proposed)

        assert not result.approved
        assert result.pending_request_id is not None
        assert "awaiting" in result.reason
        # Verify request exists in manager
        assert manager.has_pending(proposed.scope.strategy_id)

    @pytest.mark.asyncio
    async def test_t3_creates_pending_with_higher_escalation(self):
        """T3_DUAL_APPROVE → pending with L3_RISK escalation."""
        manager = ApprovalManager(auto_approve_l1=True)
        service = CPApprovalService(approval_manager=manager)

        policy_decision = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T3_DUAL_APPROVE,
            reasons=["critical_risk"],
            context_snapshot={},
        )
        proposed = _proposed()

        result = await service.request(policy_decision, proposed)

        assert not result.approved
        assert result.pending_request_id is not None
        # Verify escalation level in the underlying request
        pending = manager.get_pending(proposed.scope.strategy_id)
        assert len(pending) == 1
        assert pending[0].escalation_level == EscalationLevel.L3_RISK

    @pytest.mark.asyncio
    async def test_pending_can_be_approved_later(self):
        """After T2 creates pending, manual approval resolves it."""
        manager = ApprovalManager(auto_approve_l1=True)
        service = CPApprovalService(approval_manager=manager)

        policy_decision = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T2_APPROVE,
            reasons=["policy_violation"],
            context_snapshot={},
        )
        proposed = _proposed()

        result = await service.request(policy_decision, proposed)
        request_id = result.pending_request_id

        # Manually approve
        approval = await manager.approve(
            request_id, decided_by="operator_1", reason="reviewed_ok",
        )
        assert approval is not None
        assert approval.status == ApprovalStatus.APPROVED


# ===========================================================================
# A4: CPPolicyDecision with T0 tier → auto-approve
# ===========================================================================


class TestA4AutoApprove:
    @pytest.mark.asyncio
    async def test_t0_auto_approved(self):
        """A4: T0_AUTONOMOUS → approved immediately, no pending."""
        manager = ApprovalManager()
        service = CPApprovalService(approval_manager=manager)

        policy_decision = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T0_AUTONOMOUS,
        )
        proposed = _proposed()

        result = await service.request(policy_decision, proposed)

        assert result.approved
        assert result.tier == ApprovalTier.T0_AUTONOMOUS
        assert result.pending_request_id is None
        assert "system:autonomous" in result.decided_by

    @pytest.mark.asyncio
    async def test_t1_auto_approved_with_notification(self):
        """T1_NOTIFY → approved immediately, notification emitted."""
        manager = ApprovalManager()
        bus = AsyncMock()
        bus.publish = AsyncMock()
        service = CPApprovalService(
            approval_manager=manager, event_bus=bus,
        )

        policy_decision = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T1_NOTIFY,
            reasons=["minor_violation"],
        )
        proposed = _proposed()

        result = await service.request(policy_decision, proposed)

        assert result.approved
        assert result.tier == ApprovalTier.T1_NOTIFY
        # Notification event emitted
        bus.publish.assert_called()


# ===========================================================================
# ApprovalService fail-closed
# ===========================================================================


class TestApprovalServiceFailClosed:
    @pytest.mark.asyncio
    async def test_manager_exception_rejects(self):
        """ApprovalManager exception → rejection (fail-closed)."""
        manager = MagicMock(spec=ApprovalManager)
        manager.request_approval = AsyncMock(
            side_effect=RuntimeError("manager crashed")
        )
        service = CPApprovalService(approval_manager=manager)

        policy_decision = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T2_APPROVE,
            reasons=["test"],
            context_snapshot={},
        )
        proposed = _proposed()

        result = await service.request(policy_decision, proposed)

        assert not result.approved
        assert "approval_service_internal_error" in result.reason


# ===========================================================================
# Audit integration
# ===========================================================================


class TestApprovalAudit:
    @pytest.mark.asyncio
    async def test_t0_decision_audited(self):
        """T0 approval decision is recorded in audit log."""
        audit = AuditLog()
        manager = ApprovalManager()
        service = CPApprovalService(
            approval_manager=manager, audit_log=audit,
        )

        policy_decision = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T0_AUTONOMOUS,
        )
        proposed = _proposed()

        await service.request(policy_decision, proposed)

        entries = audit.read(proposed.correlation_id)
        assert len(entries) >= 1
        assert entries[0].event_type == "approval_decision"
        assert entries[0].payload["approved"] is True

    @pytest.mark.asyncio
    async def test_t2_pending_decision_audited(self):
        """T2 pending decision is recorded in audit log."""
        audit = AuditLog()
        manager = ApprovalManager()
        service = CPApprovalService(
            approval_manager=manager, audit_log=audit,
        )

        policy_decision = CPPolicyDecision(
            action_id="a1",
            allowed=True,
            tier=ApprovalTier.T2_APPROVE,
            reasons=["test_violation"],
            context_snapshot={},
        )
        proposed = _proposed()

        await service.request(policy_decision, proposed)

        entries = audit.read(proposed.correlation_id)
        assert len(entries) >= 1
        assert entries[0].payload["approved"] is False
        assert entries[0].payload["pending_request_id"] is not None


# ===========================================================================
# End-to-end: Evaluator → ApprovalService
# ===========================================================================


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_pass_through_flow(self):
        """Action passes policy → T0 → auto-approved."""
        engine = _make_engine_with_rules([_max_notional_rule(100_000)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        manager = ApprovalManager()
        service = CPApprovalService(approval_manager=manager)

        proposed = _proposed(params={"order_notional_usd": 50_000})

        # Step 1: Policy evaluation
        policy_decision = evaluator.evaluate(proposed)
        assert policy_decision.allowed

        # Step 2: Approval (T0 → auto)
        approval = await service.request(policy_decision, proposed)
        assert approval.approved

    @pytest.mark.asyncio
    async def test_violation_hold_flow(self):
        """Action violates policy → BLOCK → T2 → pending."""
        engine = _make_engine_with_rules([_max_notional_rule(50_000)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        manager = ApprovalManager()
        service = CPApprovalService(approval_manager=manager)

        proposed = _proposed(params={"order_notional_usd": 100_000})

        # Step 1: Policy evaluation
        policy_decision = evaluator.evaluate(proposed)
        assert not policy_decision.allowed
        assert policy_decision.tier == ApprovalTier.T2_APPROVE

        # Step 2: For blocked actions, ToolGateway would reject before approval.
        # But if somehow allowed=True with T2, approval would hold.

    @pytest.mark.asyncio
    async def test_reduce_size_notify_flow(self):
        """Action triggers REDUCE_SIZE → T1 → auto-approved with notification."""
        engine = _make_engine_with_rules([_reduce_size_rule(0.3)])
        evaluator = CPPolicyEvaluator(policy_engine=engine)

        manager = ApprovalManager()
        bus = AsyncMock()
        bus.publish = AsyncMock()
        service = CPApprovalService(
            approval_manager=manager, event_bus=bus,
        )

        proposed = _proposed(params={"correlated_exposure_pct": 0.5})

        # Step 1: Policy evaluation
        policy_decision = evaluator.evaluate(proposed)
        assert policy_decision.allowed  # REDUCE_SIZE allows execution
        assert policy_decision.tier == ApprovalTier.T1_NOTIFY

        # Step 2: Approval with notification
        approval = await service.request(policy_decision, proposed)
        assert approval.approved
        bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_protective_tool_fast_path(self):
        """Cancel order → T0 fast-path even with violations (D1)."""
        engine = _make_engine_with_rules([_max_notional_rule(50_000)])
        evaluator = build_default_evaluator(engine)

        manager = ApprovalManager()
        service = CPApprovalService(approval_manager=manager)

        # Cancel order with context that would fail notional check
        proposed = _proposed(
            tool_name=ToolName.CANCEL_ORDER,
            params={"order_notional_usd": 100_000},
        )

        policy_decision = evaluator.evaluate(proposed)
        # Tier forced to T0 even though policy failed
        assert policy_decision.tier == ApprovalTier.T0_AUTONOMOUS

        # T0 → auto-approve (ToolGateway skips approval for T0)
