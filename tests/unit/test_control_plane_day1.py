"""Day 1 tests: control plane types, audit log, and fail-closed fixes.

Tests:
    - Control plane action types (ProposedAction, CPPolicyDecision, etc.)
    - AuditLog append/read/fail-closed contract
    - B5: GovernanceGate policy engine exception -> BLOCK
    - B6: GovernanceGate approval manager exception -> BLOCK
    - B10: PolicyEngine missing field -> rule FAILS
    - B11: PolicyEngine type mismatch -> rule FAILS
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.control_plane.action_types import (
    ActionScope,
    ApprovalDecision,
    ApprovalTier,
    AuditEntry,
    CPPolicyDecision,
    DegradedMode,
    MUTATING_TOOLS,
    ProposedAction,
    TIER_RANK,
    ToolCallResult,
    ToolName,
)
from agentic_trading.control_plane.audit_log import AuditLog
from agentic_trading.core.enums import GovernanceAction, ImpactTier, MaturityLevel
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


def _make_rule(
    rule_id: str = "test_rule",
    field: str = "value",
    operator: Operator = Operator.LE,
    threshold: float = 100.0,
    action: GovernanceAction = GovernanceAction.BLOCK,
    **kwargs,
) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        name=f"Test {rule_id}",
        field=field,
        operator=operator,
        threshold=threshold,
        action=action,
        **kwargs,
    )


def _make_policy_set(
    rules: list[PolicyRule] | None = None,
    mode: PolicyMode = PolicyMode.ENFORCED,
    **kwargs,
) -> PolicySet:
    return PolicySet(
        set_id=kwargs.pop("set_id", "test_set"),
        name=kwargs.pop("name", "Test Set"),
        version=kwargs.pop("version", 1),
        mode=mode,
        rules=rules or [],
        **kwargs,
    )


# ===========================================================================
# Control Plane Action Types
# ===========================================================================


class TestProposedAction:
    def test_creates_with_defaults(self):
        action = ProposedAction(tool_name=ToolName.SUBMIT_ORDER)
        assert action.action_id
        assert action.correlation_id
        assert action.request_hash
        assert action.tool_name == ToolName.SUBMIT_ORDER

    def test_request_hash_deterministic(self):
        scope = ActionScope(strategy_id="s1", symbol="BTC/USDT")
        a1 = ProposedAction(
            tool_name=ToolName.SUBMIT_ORDER,
            scope=scope,
            idempotency_key="abc",
        )
        a2 = ProposedAction(
            tool_name=ToolName.SUBMIT_ORDER,
            scope=scope,
            idempotency_key="abc",
        )
        assert a1.request_hash == a2.request_hash

    def test_different_keys_different_hash(self):
        a1 = ProposedAction(
            tool_name=ToolName.SUBMIT_ORDER,
            idempotency_key="abc",
        )
        a2 = ProposedAction(
            tool_name=ToolName.SUBMIT_ORDER,
            idempotency_key="xyz",
        )
        assert a1.request_hash != a2.request_hash


class TestToolName:
    def test_mutating_tools_all_defined(self):
        for tool in MUTATING_TOOLS:
            assert tool in ToolName

    def test_read_tools_not_mutating(self):
        read_tools = {
            ToolName.GET_POSITIONS,
            ToolName.GET_BALANCES,
            ToolName.GET_OPEN_ORDERS,
            ToolName.GET_INSTRUMENT,
            ToolName.GET_FUNDING_RATE,
            ToolName.GET_CLOSED_PNL,
        }
        for tool in read_tools:
            assert tool not in MUTATING_TOOLS


class TestApprovalTier:
    def test_tier_rank_ordering(self):
        assert TIER_RANK[ApprovalTier.T0_AUTONOMOUS] < TIER_RANK[ApprovalTier.T1_NOTIFY]
        assert TIER_RANK[ApprovalTier.T1_NOTIFY] < TIER_RANK[ApprovalTier.T2_APPROVE]
        assert TIER_RANK[ApprovalTier.T2_APPROVE] < TIER_RANK[ApprovalTier.T3_DUAL_APPROVE]


class TestAuditEntry:
    def test_payload_hash_computed(self):
        entry = AuditEntry(
            correlation_id="c1",
            event_type="test",
            payload={"key": "value"},
        )
        assert entry.payload_hash
        assert len(entry.payload_hash) == 16

    def test_same_payload_same_hash(self):
        e1 = AuditEntry(
            correlation_id="c1",
            event_type="test",
            payload={"a": 1, "b": 2},
        )
        e2 = AuditEntry(
            correlation_id="c1",
            event_type="test",
            payload={"b": 2, "a": 1},  # different order, same content
        )
        assert e1.payload_hash == e2.payload_hash


class TestCPPolicyDecision:
    def test_allowed_decision(self):
        d = CPPolicyDecision(allowed=True, action_id="a1")
        assert d.allowed
        assert d.tier == ApprovalTier.T0_AUTONOMOUS
        assert d.sizing_multiplier == 1.0

    def test_blocked_decision(self):
        d = CPPolicyDecision(
            allowed=False,
            action_id="a1",
            reasons=["max_notional_exceeded"],
        )
        assert not d.allowed
        assert len(d.reasons) == 1


# ===========================================================================
# AuditLog
# ===========================================================================


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_append_and_read(self):
        log = AuditLog()
        entry = AuditEntry(
            correlation_id="c1",
            event_type="test",
            payload={"x": 1},
        )
        await log.append(entry)
        assert log.entry_count == 1
        entries = log.read("c1")
        assert len(entries) == 1
        assert entries[0].payload == {"x": 1}

    @pytest.mark.asyncio
    async def test_read_by_action(self):
        log = AuditLog()
        entry = AuditEntry(
            correlation_id="c1",
            causation_id="a1",
            event_type="tool_call",
        )
        await log.append(entry)
        entries = log.read_by_action("a1")
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_unavailable_raises(self):
        log = AuditLog()
        log.set_available(False)
        with pytest.raises(RuntimeError, match="unavailable"):
            await log.append(AuditEntry(
                correlation_id="c1", event_type="test",
            ))

    @pytest.mark.asyncio
    async def test_set_available_toggle(self):
        log = AuditLog()
        assert log.is_available
        log.set_available(False)
        assert not log.is_available
        log.set_available(True)
        assert log.is_available

    @pytest.mark.asyncio
    async def test_memory_cap(self):
        log = AuditLog(max_memory_entries=5)
        for i in range(10):
            await log.append(AuditEntry(
                correlation_id=f"c{i}", event_type="test",
            ))
        assert log.entry_count == 5

    @pytest.mark.asyncio
    async def test_read_empty(self):
        log = AuditLog()
        entries = log.read("nonexistent")
        assert entries == []

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        log = AuditLog(persist_path=path)
        await log.append(AuditEntry(
            correlation_id="c1", event_type="test", payload={"k": "v"},
        ))
        # Verify file written
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert "c1" in lines[0]


# ===========================================================================
# B5: GovernanceGate policy engine error -> BLOCK (fail-closed)
# ===========================================================================


class TestB5PolicyEngineFailClosed:
    @pytest.mark.asyncio
    async def test_policy_engine_exception_blocks(self):
        """GIVEN a GovernanceGate with a policy engine that raises
        WHEN evaluate() is called
        THEN the decision is BLOCK (not ALLOW)
        """
        from agentic_trading.governance.gate import GovernanceGate
        from agentic_trading.core.config import GovernanceConfig

        # Create minimal mocks
        config = MagicMock(spec=GovernanceConfig)
        config.enabled = True
        config.execution_tokens = MagicMock()
        config.execution_tokens.require_tokens = False

        maturity = MagicMock()
        maturity.get_level.return_value = MaturityLevel.L3_CONSTRAINED
        maturity.can_execute.return_value = True
        maturity.get_sizing_cap.return_value = 1.0

        health = MagicMock()
        health.get_score.return_value = 1.0
        health.get_sizing_multiplier.return_value = 1.0

        impact = MagicMock()
        impact.classify.return_value = ImpactTier.LOW

        drift = MagicMock()
        drift.check_drift.return_value = []

        # Policy engine that THROWS
        broken_policy = MagicMock()
        broken_policy.registered_sets = ["test_set"]
        broken_policy.evaluate.side_effect = RuntimeError("policy engine crashed")

        gate = GovernanceGate(
            config=config,
            maturity=maturity,
            health=health,
            impact=impact,
            drift=drift,
            policy_engine=broken_policy,
        )

        decision = await gate.evaluate(
            strategy_id="test",
            symbol="BTC/USDT",
            notional_usd=1000.0,
        )

        # B5: MUST be BLOCK, not ALLOW
        assert decision.action == GovernanceAction.BLOCK
        assert "policy_engine_error" in decision.reason


# ===========================================================================
# B6: GovernanceGate approval manager error -> BLOCK (fail-closed)
# ===========================================================================


class TestB6ApprovalManagerFailClosed:
    @pytest.mark.asyncio
    async def test_approval_manager_exception_blocks(self):
        """GIVEN a GovernanceGate with an approval manager that raises
        WHEN evaluate() is called and approval is required
        THEN the decision is BLOCK (not ALLOW)
        """
        from agentic_trading.governance.gate import GovernanceGate
        from agentic_trading.core.config import GovernanceConfig

        config = MagicMock(spec=GovernanceConfig)
        config.enabled = True
        config.execution_tokens = MagicMock()
        config.execution_tokens.require_tokens = False

        maturity = MagicMock()
        maturity.get_level.return_value = MaturityLevel.L3_CONSTRAINED
        maturity.can_execute.return_value = True
        maturity.get_sizing_cap.return_value = 1.0

        health = MagicMock()
        health.get_score.return_value = 1.0
        health.get_sizing_multiplier.return_value = 1.0

        impact = MagicMock()
        impact.classify.return_value = ImpactTier.MEDIUM

        drift = MagicMock()
        drift.check_drift.return_value = []

        # Approval manager that THROWS
        broken_approval = MagicMock()
        broken_approval.check_approval_required.side_effect = RuntimeError(
            "approval manager crashed"
        )

        gate = GovernanceGate(
            config=config,
            maturity=maturity,
            health=health,
            impact=impact,
            drift=drift,
            approval_manager=broken_approval,
        )

        decision = await gate.evaluate(
            strategy_id="test",
            symbol="BTC/USDT",
            notional_usd=1000.0,
        )

        # B6: MUST be BLOCK, not ALLOW
        assert decision.action == GovernanceAction.BLOCK
        assert "approval_manager_error" in decision.reason


# ===========================================================================
# B10: PolicyEngine missing context field -> rule FAILS
# ===========================================================================


class TestB10MissingFieldFailClosed:
    def test_missing_field_fails_rule(self):
        """GIVEN a rule checking 'order_notional_usd'
        WHEN the context is missing that field
        THEN the rule FAILS (not passes)
        """
        engine = PolicyEngine()
        policy_set = _make_policy_set(
            rules=[_make_rule(
                rule_id="max_notional",
                field="order_notional_usd",
                operator=Operator.LE,
                threshold=500_000.0,
            )],
        )
        engine.register(policy_set)

        # Context is MISSING order_notional_usd
        result = engine.evaluate("test_set", {"strategy_id": "test"})

        assert not result.all_passed
        assert len(result.failed_rules) == 1
        assert "required_field_missing" in result.failed_rules[0].reason

    def test_missing_nested_field_fails_rule(self):
        """GIVEN a rule checking a dot-path 'position.leverage'
        WHEN the context has 'position' but not 'leverage'
        THEN the rule FAILS
        """
        engine = PolicyEngine()
        policy_set = _make_policy_set(
            rules=[_make_rule(
                rule_id="max_leverage",
                field="position.leverage",
                operator=Operator.LE,
                threshold=10,
            )],
        )
        engine.register(policy_set)

        result = engine.evaluate("test_set", {"position": {}})
        assert not result.all_passed
        assert "required_field_missing" in result.failed_rules[0].reason

    def test_present_field_still_works(self):
        """GIVEN a rule checking 'value' and the context provides it
        WHEN the value satisfies the condition
        THEN the rule passes
        """
        engine = PolicyEngine()
        policy_set = _make_policy_set(
            rules=[_make_rule(
                field="value",
                operator=Operator.LE,
                threshold=100.0,
            )],
        )
        engine.register(policy_set)

        result = engine.evaluate("test_set", {"value": 50.0})
        assert result.all_passed


# ===========================================================================
# B11: PolicyEngine type mismatch -> rule FAILS
# ===========================================================================


class TestB11TypeMismatchFailClosed:
    def test_type_mismatch_fails_rule(self):
        """GIVEN a rule comparing numeric field with LE
        WHEN the field value is a non-comparable string
        THEN the rule FAILS (not passes)
        """
        engine = PolicyEngine()
        policy_set = _make_policy_set(
            rules=[_make_rule(
                field="value",
                operator=Operator.LE,
                threshold=100.0,
            )],
        )
        engine.register(policy_set)

        # Pass a string that can't be compared with LE to a float
        result = engine.evaluate("test_set", {"value": "not_a_number"})
        assert not result.all_passed
        assert len(result.failed_rules) == 1

    def test_between_type_mismatch_fails(self):
        """GIVEN a BETWEEN rule
        WHEN the field value is a string
        THEN the rule FAILS
        """
        engine = PolicyEngine()
        policy_set = _make_policy_set(
            rules=[_make_rule(
                field="value",
                operator=Operator.BETWEEN,
                threshold=[0, 100],
            )],
        )
        engine.register(policy_set)

        result = engine.evaluate("test_set", {"value": "oops"})
        assert not result.all_passed

    def test_valid_comparison_still_works(self):
        """Regression: ensure valid comparisons are not broken."""
        engine = PolicyEngine()
        policy_set = _make_policy_set(
            rules=[_make_rule(
                field="value",
                operator=Operator.LE,
                threshold=100.0,
            )],
        )
        engine.register(policy_set)

        result = engine.evaluate("test_set", {"value": 50.0})
        assert result.all_passed

        result = engine.evaluate("test_set", {"value": 150.0})
        assert not result.all_passed
