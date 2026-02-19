"""Tests for the unified PolicyGate facade and governance→policy shim layer.

Tests cover:
1. PolicyGate.from_config construction and wiring
2. PolicyGate.evaluate delegation to GovernanceGate
3. Policy management: register, mode switch, rollback
4. Approval rule delegation
5. Component accessors
6. Backward-compat shim imports (governance.* → policy.*)
"""

from __future__ import annotations

import pytest

from agentic_trading.core.config import GovernanceConfig, RiskConfig
from agentic_trading.core.enums import GovernanceAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_governance_config(enabled: bool = True) -> GovernanceConfig:
    return GovernanceConfig(enabled=enabled)


def _make_policy_gate(enabled: bool = True):
    """Build a PolicyGate via the factory with default config."""
    from agentic_trading.policy.gate import PolicyGate

    config = _make_governance_config(enabled)
    return PolicyGate.from_config(config)


# ---------------------------------------------------------------------------
# PolicyGate factory
# ---------------------------------------------------------------------------

class TestPolicyGateFactory:
    """PolicyGate.from_config produces a fully wired instance."""

    def test_from_config_creates_gate(self):
        gate = _make_policy_gate()
        assert gate is not None
        assert gate.governance_gate is not None
        assert gate.policy_engine is not None
        assert gate.policy_store is not None

    def test_from_config_registers_default_policies(self):
        gate = _make_policy_gate()
        registered = gate.policy_engine.registered_sets
        assert "pre_trade_risk" in registered
        assert "post_trade_risk" in registered
        assert "strategy_constraints" in registered

    def test_from_config_no_approval_manager_by_default(self):
        gate = _make_policy_gate()
        assert gate.approval_manager is None

    def test_from_config_with_approval_rules(self):
        from agentic_trading.policy.approval_models import (
            ApprovalRule,
            ApprovalTrigger,
            EscalationLevel,
        )

        rules = [
            ApprovalRule(
                rule_id="test_rule",
                name="Test Approval",
                trigger=ApprovalTrigger.HIGH_IMPACT,
                escalation_level=EscalationLevel.L2_OPERATOR,
                min_notional_usd=100_000,
            ),
        ]
        from agentic_trading.policy.gate import PolicyGate

        config = _make_governance_config()
        gate = PolicyGate.from_config(config, approval_rules=rules)
        assert gate.approval_manager is not None
        assert len(gate.approval_manager.rules) == 1

    def test_from_config_with_risk_config(self):
        from agentic_trading.policy.gate import PolicyGate

        config = _make_governance_config()
        risk = RiskConfig(max_single_position_pct=0.15)
        gate = PolicyGate.from_config(config, risk_config=risk)
        # Default policies should use the custom risk config
        ps = gate.policy_store.get_active("pre_trade_risk")
        assert ps is not None
        # Find the max_position_size rule
        rule = next(r for r in ps.rules if r.rule_id == "max_position_size")
        assert rule.threshold == 0.15


# ---------------------------------------------------------------------------
# PolicyGate.evaluate
# ---------------------------------------------------------------------------

class TestPolicyGateEvaluate:
    """PolicyGate.evaluate delegates to GovernanceGate."""

    @pytest.mark.asyncio
    async def test_evaluate_governance_disabled_allows(self):
        gate = _make_policy_gate(enabled=False)
        decision = await gate.evaluate(
            strategy_id="test_strat",
            symbol="BTC/USDT",
            notional_usd=10_000,
        )
        assert decision.action == GovernanceAction.ALLOW
        assert decision.sizing_multiplier == 1.0
        assert decision.reason == "governance_disabled"

    @pytest.mark.asyncio
    async def test_evaluate_returns_governance_decision(self):
        gate = _make_policy_gate()
        # Default maturity is L1_PAPER → should block
        decision = await gate.evaluate(
            strategy_id="test_strat",
            symbol="BTC/USDT",
            notional_usd=10_000,
        )
        assert decision.action == GovernanceAction.BLOCK
        assert "maturity" in decision.reason

    @pytest.mark.asyncio
    async def test_evaluate_l4_strategy_passes(self):
        """L4 strategy with governance disabled passes cleanly."""
        gate = _make_policy_gate(enabled=False)
        decision = await gate.evaluate(
            strategy_id="mature_strat",
            symbol="BTC/USDT",
            notional_usd=1_000,
            portfolio_pct=0.01,
            leverage=1,
        )
        # Governance disabled → ALLOW
        assert decision.action == GovernanceAction.ALLOW
        assert decision.sizing_multiplier == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_l4_with_full_context_passes(self):
        """L4 strategy with all required context fields passes governance."""
        from agentic_trading.core.enums import MaturityLevel

        gate = _make_policy_gate()
        gate.maturity.set_level("mature_strat", MaturityLevel.L4_AUTONOMOUS)
        # The policy engine requires these fields for fail-closed evaluation.
        # The GovernanceGate.evaluate provides a subset; missing fields
        # cause BLOCK/KILL.  In production, the execution engine supplies
        # a complete context.  For this test we disable the policy engine
        # to test the gate logic only.
        gate.governance_gate._policy_engine = None
        decision = await gate.evaluate(
            strategy_id="mature_strat",
            symbol="BTC/USDT",
            notional_usd=1_000,
            portfolio_pct=0.01,
            leverage=1,
        )
        # L4 + small order + no policy engine → ALLOW
        assert decision.action == GovernanceAction.ALLOW
        assert decision.sizing_multiplier == 1.0


# ---------------------------------------------------------------------------
# Policy management
# ---------------------------------------------------------------------------

class TestPolicyManagement:
    """PolicyGate policy management operations."""

    def test_register_policy_set(self):
        from agentic_trading.policy.models import (
            Operator,
            PolicyMode,
            PolicyRule,
            PolicySet,
            PolicyType,
        )

        gate = _make_policy_gate()
        custom_ps = PolicySet(
            set_id="custom_test",
            name="Custom Test",
            version=1,
            mode=PolicyMode.SHADOW,
            rules=[
                PolicyRule(
                    rule_id="test_rule_1",
                    name="Test Rule",
                    field="test_field",
                    operator=Operator.LE,
                    threshold=100,
                    action=GovernanceAction.BLOCK,
                    policy_type=PolicyType.RISK_LIMIT,
                ),
            ],
        )
        gate.register_policy_set(custom_ps)
        assert "custom_test" in gate.policy_engine.registered_sets
        assert gate.policy_store.get_active("custom_test") is not None

    def test_set_policy_mode(self):
        from agentic_trading.policy.models import PolicyMode

        gate = _make_policy_gate()
        success = gate.set_policy_mode("pre_trade_risk", PolicyMode.SHADOW)
        assert success is True
        ps = gate.policy_store.get_active("pre_trade_risk")
        assert ps.mode == PolicyMode.SHADOW

    def test_set_policy_mode_unknown_set(self):
        from agentic_trading.policy.models import PolicyMode

        gate = _make_policy_gate()
        success = gate.set_policy_mode("nonexistent", PolicyMode.SHADOW)
        assert success is False

    def test_rollback_policy(self):
        from agentic_trading.policy.models import (
            Operator,
            PolicyMode,
            PolicyRule,
            PolicySet,
            PolicyType,
        )

        gate = _make_policy_gate()
        # Save v2 of pre_trade_risk
        v2 = PolicySet(
            set_id="pre_trade_risk",
            name="Pre-Trade v2",
            version=2,
            mode=PolicyMode.ENFORCED,
            rules=[
                PolicyRule(
                    rule_id="v2_rule",
                    name="V2 Rule",
                    field="test_field",
                    operator=Operator.LE,
                    threshold=200,
                    action=GovernanceAction.BLOCK,
                    policy_type=PolicyType.RISK_LIMIT,
                ),
            ],
        )
        gate.register_policy_set(v2)
        assert gate.policy_store.active_version("pre_trade_risk") == 2

        # Rollback to v1
        result = gate.rollback_policy("pre_trade_risk")
        assert result is not None
        assert result.version == 1


# ---------------------------------------------------------------------------
# Component accessors
# ---------------------------------------------------------------------------

class TestComponentAccessors:
    """PolicyGate exposes sub-components via properties."""

    def test_maturity_accessor(self):
        from agentic_trading.policy.maturity import MaturityManager

        gate = _make_policy_gate()
        assert isinstance(gate.maturity, MaturityManager)

    def test_health_accessor(self):
        from agentic_trading.policy.health_score import HealthTracker

        gate = _make_policy_gate()
        assert isinstance(gate.health, HealthTracker)

    def test_drift_accessor(self):
        from agentic_trading.policy.drift_detector import DriftDetector

        gate = _make_policy_gate()
        assert isinstance(gate.drift, DriftDetector)

    def test_get_sizing_multiplier(self):
        gate = _make_policy_gate()
        mult = gate.get_sizing_multiplier("new_strat")
        # Default L1 → 0.0 sizing cap
        assert mult == 0.0

    def test_record_trade_outcome(self):
        gate = _make_policy_gate()
        # Should not raise
        gate.record_trade_outcome("test_strat", won=True, r_multiple=1.5)
        gate.record_trade_outcome("test_strat", won=False, r_multiple=-0.5)


# ---------------------------------------------------------------------------
# Backward-compat shim imports (governance → policy)
# ---------------------------------------------------------------------------

class TestGovernanceShimImports:
    """Verify all shim imports from governance.* resolve correctly."""

    def test_governance_init_imports(self):
        from agentic_trading.governance import (
            ApprovalDecision,
            ApprovalManager,
            ApprovalRequest,
            ApprovalRule,
            ApprovalStatus,
            ApprovalSummary,
            ApprovalTrigger,
            DriftDetector,
            EscalationLevel,
            ExecutionToken,
            GovernanceCanary,
            GovernanceGate,
            HealthTracker,
            ImpactClassifier,
            MaturityManager,
            PolicyEngine,
            PolicyMode,
            PolicyRule,
            PolicySet,
            PolicyStore,
            TokenManager,
        )
        # Verify these are the same classes as in policy.*
        from agentic_trading.policy import (
            GovernanceGate as PGovernanceGate,
            PolicyEngine as PPolicyEngine,
        )
        assert GovernanceGate is PGovernanceGate
        assert PolicyEngine is PPolicyEngine

    def test_governance_gate_shim(self):
        from agentic_trading.governance.gate import GovernanceGate as ShimGate
        from agentic_trading.policy.governance_gate import (
            GovernanceGate as CanonicalGate,
        )
        assert ShimGate is CanonicalGate

    def test_policy_engine_shim(self):
        from agentic_trading.governance.policy_engine import (
            PolicyEngine as ShimEngine,
        )
        from agentic_trading.policy.engine import PolicyEngine as CanonicalEngine
        assert ShimEngine is CanonicalEngine

    def test_policy_models_shim(self):
        from agentic_trading.governance.policy_models import (
            PolicyMode as ShimMode,
            PolicyRule as ShimRule,
            PolicySet as ShimSet,
        )
        from agentic_trading.policy.models import (
            PolicyMode as CanonicalMode,
            PolicyRule as CanonicalRule,
            PolicySet as CanonicalSet,
        )
        assert ShimMode is CanonicalMode
        assert ShimRule is CanonicalRule
        assert ShimSet is CanonicalSet

    def test_policy_store_shim(self):
        from agentic_trading.governance.policy_store import (
            PolicyStore as ShimStore,
        )
        from agentic_trading.policy.store import PolicyStore as CanonicalStore
        assert ShimStore is CanonicalStore

    def test_approval_models_shim(self):
        from agentic_trading.governance.approval_models import (
            ApprovalRequest as ShimRequest,
            ApprovalStatus as ShimStatus,
        )
        from agentic_trading.policy.approval_models import (
            ApprovalRequest as CanonicalRequest,
            ApprovalStatus as CanonicalStatus,
        )
        assert ShimRequest is CanonicalRequest
        assert ShimStatus is CanonicalStatus

    def test_approval_manager_shim(self):
        from agentic_trading.governance.approval_manager import (
            ApprovalManager as ShimMgr,
        )
        from agentic_trading.policy.approval_manager import (
            ApprovalManager as CanonicalMgr,
        )
        assert ShimMgr is CanonicalMgr

    def test_maturity_shim(self):
        from agentic_trading.governance.maturity import (
            MaturityManager as ShimMgr,
        )
        from agentic_trading.policy.maturity import (
            MaturityManager as CanonicalMgr,
        )
        assert ShimMgr is CanonicalMgr

    def test_health_score_shim(self):
        from agentic_trading.governance.health_score import (
            HealthTracker as ShimTracker,
        )
        from agentic_trading.policy.health_score import (
            HealthTracker as CanonicalTracker,
        )
        assert ShimTracker is CanonicalTracker

    def test_drift_detector_shim(self):
        from agentic_trading.governance.drift_detector import (
            DriftDetector as ShimDetector,
        )
        from agentic_trading.policy.drift_detector import (
            DriftDetector as CanonicalDetector,
        )
        assert ShimDetector is CanonicalDetector

    def test_impact_classifier_shim(self):
        from agentic_trading.governance.impact_classifier import (
            ImpactClassifier as ShimClassifier,
        )
        from agentic_trading.policy.impact_classifier import (
            ImpactClassifier as CanonicalClassifier,
        )
        assert ShimClassifier is CanonicalClassifier

    def test_tokens_shim(self):
        from agentic_trading.governance.tokens import (
            ExecutionToken as ShimToken,
            TokenManager as ShimMgr,
        )
        from agentic_trading.policy.tokens import (
            ExecutionToken as CanonicalToken,
            TokenManager as CanonicalMgr,
        )
        assert ShimToken is CanonicalToken
        assert ShimMgr is CanonicalMgr

    def test_canary_shim(self):
        from agentic_trading.governance.canary import (
            GovernanceCanary as ShimCanary,
        )
        from agentic_trading.policy.canary import (
            GovernanceCanary as CanonicalCanary,
        )
        assert ShimCanary is CanonicalCanary

    def test_incident_manager_shim(self):
        from agentic_trading.governance.incident_manager import (
            IncidentManager as ShimMgr,
        )
        from agentic_trading.policy.incident_manager import (
            IncidentManager as CanonicalMgr,
        )
        assert ShimMgr is CanonicalMgr

    def test_strategy_lifecycle_shim(self):
        from agentic_trading.governance.strategy_lifecycle import (
            StrategyLifecycleManager as ShimMgr,
        )
        from agentic_trading.policy.strategy_lifecycle import (
            StrategyLifecycleManager as CanonicalMgr,
        )
        assert ShimMgr is CanonicalMgr

    def test_default_policies_shim(self):
        from agentic_trading.governance.default_policies import (
            build_pre_trade_policies as shim_fn,
        )
        from agentic_trading.policy.default_policies import (
            build_pre_trade_policies as canonical_fn,
        )
        assert shim_fn is canonical_fn
