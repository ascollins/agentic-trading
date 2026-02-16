"""Tests for the policy-as-code engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agentic_trading.core.enums import GovernanceAction, MaturityLevel
from agentic_trading.governance.policy_engine import PolicyEngine
from agentic_trading.governance.policy_models import (
    Operator,
    PolicyDecision,
    PolicyEvalResult,
    PolicyMode,
    PolicyRule,
    PolicySet,
    PolicyType,
)
from agentic_trading.governance.policy_store import PolicyStore


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


# ---------------------------------------------------------------------------
# PolicyRule model tests
# ---------------------------------------------------------------------------


class TestPolicyRule:
    def test_basic_creation(self):
        rule = _make_rule()
        assert rule.rule_id == "test_rule"
        assert rule.operator == Operator.LE
        assert rule.enabled

    def test_scoped_rule(self):
        rule = _make_rule(
            strategy_ids=["trend_following"],
            symbols=["BTC/USDT"],
        )
        assert rule.strategy_ids == ["trend_following"]
        assert rule.symbols == ["BTC/USDT"]

    def test_disabled_rule(self):
        rule = _make_rule(enabled=False)
        assert not rule.enabled


class TestPolicySet:
    def test_active_rules_excludes_disabled(self):
        rules = [
            _make_rule(rule_id="r1", enabled=True),
            _make_rule(rule_id="r2", enabled=False),
            _make_rule(rule_id="r3", enabled=True),
        ]
        ps = _make_policy_set(rules=rules)
        assert len(ps.active_rules) == 2
        assert ps.active_rules[0].rule_id == "r1"
        assert ps.active_rules[1].rule_id == "r3"

    def test_serialization_roundtrip(self):
        rules = [_make_rule(rule_id="r1")]
        ps = _make_policy_set(rules=rules)
        data = ps.model_dump(mode="json")
        restored = PolicySet.model_validate(data)
        assert restored.set_id == ps.set_id
        assert len(restored.rules) == 1
        assert restored.rules[0].rule_id == "r1"


# ---------------------------------------------------------------------------
# PolicyEngine: operator tests
# ---------------------------------------------------------------------------


class TestPolicyEngineOperators:
    """Test each comparison operator."""

    def setup_method(self):
        self.engine = PolicyEngine()

    def _eval_single(
        self,
        field_value: float,
        operator: Operator,
        threshold,
    ) -> PolicyEvalResult:
        rule = _make_rule(operator=operator, threshold=threshold)
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)
        decision = self.engine.evaluate("test_set", {"value": field_value})
        return decision.results[0]

    def test_le_pass(self):
        result = self._eval_single(50, Operator.LE, 100)
        assert result.passed

    def test_le_fail(self):
        result = self._eval_single(150, Operator.LE, 100)
        assert not result.passed

    def test_le_equal(self):
        result = self._eval_single(100, Operator.LE, 100)
        assert result.passed

    def test_lt_pass(self):
        result = self._eval_single(50, Operator.LT, 100)
        assert result.passed

    def test_lt_fail_equal(self):
        result = self._eval_single(100, Operator.LT, 100)
        assert not result.passed

    def test_gt_pass(self):
        result = self._eval_single(150, Operator.GT, 100)
        assert result.passed

    def test_gt_fail(self):
        result = self._eval_single(50, Operator.GT, 100)
        assert not result.passed

    def test_ge_pass(self):
        result = self._eval_single(100, Operator.GE, 100)
        assert result.passed

    def test_eq_pass(self):
        result = self._eval_single(100, Operator.EQ, 100)
        assert result.passed

    def test_eq_fail(self):
        result = self._eval_single(99, Operator.EQ, 100)
        assert not result.passed

    def test_ne_pass(self):
        result = self._eval_single(99, Operator.NE, 100)
        assert result.passed

    def test_ne_fail(self):
        result = self._eval_single(100, Operator.NE, 100)
        assert not result.passed

    def test_in_pass(self):
        rule = _make_rule(operator=Operator.IN, threshold=["a", "b", "c"])
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)
        decision = self.engine.evaluate("test_set", {"value": "b"})
        assert decision.results[0].passed

    def test_in_fail(self):
        rule = _make_rule(operator=Operator.IN, threshold=["a", "b", "c"])
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)
        decision = self.engine.evaluate("test_set", {"value": "d"})
        assert not decision.results[0].passed

    def test_not_in_pass(self):
        rule = _make_rule(operator=Operator.NOT_IN, threshold=["x", "y"])
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)
        decision = self.engine.evaluate("test_set", {"value": "a"})
        assert decision.results[0].passed

    def test_between_pass(self):
        rule = _make_rule(operator=Operator.BETWEEN, threshold=[10, 50])
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)
        decision = self.engine.evaluate("test_set", {"value": 30})
        assert decision.results[0].passed

    def test_between_fail(self):
        rule = _make_rule(operator=Operator.BETWEEN, threshold=[10, 50])
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)
        decision = self.engine.evaluate("test_set", {"value": 60})
        assert not decision.results[0].passed


# ---------------------------------------------------------------------------
# PolicyEngine: evaluation logic
# ---------------------------------------------------------------------------


class TestPolicyEngineEvaluation:
    def setup_method(self):
        self.engine = PolicyEngine()

    def test_all_pass(self):
        rules = [
            _make_rule(rule_id="r1", field="leverage", threshold=3.0),
            _make_rule(rule_id="r2", field="notional", threshold=500_000),
        ]
        ps = _make_policy_set(rules=rules)
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {
            "leverage": 2.0,
            "notional": 100_000,
        })
        assert decision.all_passed
        assert decision.action == GovernanceAction.ALLOW
        assert decision.sizing_multiplier == 1.0
        assert len(decision.failed_rules) == 0

    def test_one_fails(self):
        rules = [
            _make_rule(rule_id="r1", field="leverage", threshold=3.0),
            _make_rule(rule_id="r2", field="notional", threshold=500_000),
        ]
        ps = _make_policy_set(rules=rules)
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {
            "leverage": 5.0,  # exceeds 3.0
            "notional": 100_000,
        })
        assert not decision.all_passed
        assert decision.action == GovernanceAction.BLOCK
        assert decision.sizing_multiplier == 0.0
        assert len(decision.failed_rules) == 1
        assert decision.failed_rules[0].rule_id == "r1"

    def test_missing_field_fails_closed(self):
        """B10: Missing field in context -> rule FAILS (fail-closed)."""
        rules = [_make_rule(field="nonexistent")]
        ps = _make_policy_set(rules=rules)
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {"other": 42})
        assert not decision.all_passed
        assert "required_field_missing" in decision.failed_rules[0].reason

    def test_reduce_size_action(self):
        rules = [
            _make_rule(
                field="exposure",
                threshold=0.25,
                action=GovernanceAction.REDUCE_SIZE,
            ),
        ]
        ps = _make_policy_set(rules=rules)
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {"exposure": 0.5})
        assert not decision.all_passed
        assert decision.action == GovernanceAction.REDUCE_SIZE
        assert decision.sizing_multiplier == 0.5

    def test_most_severe_action_wins(self):
        rules = [
            _make_rule(
                rule_id="r1",
                field="a",
                threshold=10,
                action=GovernanceAction.REDUCE_SIZE,
            ),
            _make_rule(
                rule_id="r2",
                field="b",
                threshold=10,
                action=GovernanceAction.KILL,
            ),
        ]
        ps = _make_policy_set(rules=rules)
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {"a": 20, "b": 20})
        assert decision.action == GovernanceAction.KILL

    def test_unregistered_set_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            self.engine.evaluate("nonexistent", {})

    def test_context_snapshot_included(self):
        rules = [_make_rule(field="x", threshold=100)]
        ps = _make_policy_set(rules=rules)
        self.engine.register(ps)

        ctx = {"x": 50, "y": "extra"}
        decision = self.engine.evaluate("test_set", ctx)
        assert decision.context_snapshot == ctx


# ---------------------------------------------------------------------------
# PolicyEngine: shadow mode
# ---------------------------------------------------------------------------


class TestPolicyEngineShadowMode:
    def setup_method(self):
        self.engine = PolicyEngine()

    def test_shadow_mode_never_blocks(self):
        rules = [_make_rule(field="leverage", threshold=3.0)]
        ps = _make_policy_set(rules=rules, mode=PolicyMode.SHADOW)
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {"leverage": 10.0})
        # Shadow mode: all_passed is True because no enforced failures
        assert decision.all_passed
        assert decision.action == GovernanceAction.ALLOW
        assert decision.sizing_multiplier == 1.0
        assert len(decision.shadow_violations) == 1

    def test_shadow_violations_recorded(self):
        rules = [
            _make_rule(rule_id="r1", field="a", threshold=10),
            _make_rule(rule_id="r2", field="b", threshold=10),
        ]
        ps = _make_policy_set(rules=rules, mode=PolicyMode.SHADOW)
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {"a": 20, "b": 5})
        assert len(decision.shadow_violations) == 1
        assert decision.shadow_violations[0].rule_id == "r1"


# ---------------------------------------------------------------------------
# PolicyEngine: scoping
# ---------------------------------------------------------------------------


class TestPolicyEngineScoping:
    def setup_method(self):
        self.engine = PolicyEngine()

    def test_strategy_scope_match(self):
        rule = _make_rule(
            field="leverage",
            threshold=3.0,
            strategy_ids=["trend_following"],
        )
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {
            "strategy_id": "trend_following",
            "leverage": 5.0,
        })
        assert not decision.all_passed

    def test_strategy_scope_mismatch_skips(self):
        rule = _make_rule(
            field="leverage",
            threshold=3.0,
            strategy_ids=["trend_following"],
        )
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)

        # mean_reversion is not in scope -- rule should be skipped
        decision = self.engine.evaluate("test_set", {
            "strategy_id": "mean_reversion",
            "leverage": 5.0,
        })
        assert decision.all_passed
        assert len(decision.results) == 0

    def test_symbol_scope(self):
        rule = _make_rule(
            field="notional",
            threshold=100_000,
            symbols=["BTC/USDT"],
        )
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)

        # ETH/USDT not in scope
        decision = self.engine.evaluate("test_set", {
            "symbol": "ETH/USDT",
            "notional": 200_000,
        })
        assert decision.all_passed

    def test_no_scope_applies_to_all(self):
        rule = _make_rule(field="leverage", threshold=3.0)
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {
            "strategy_id": "anything",
            "leverage": 5.0,
        })
        assert not decision.all_passed


# ---------------------------------------------------------------------------
# PolicyEngine: dot-path field resolution
# ---------------------------------------------------------------------------


class TestDotPathResolution:
    def setup_method(self):
        self.engine = PolicyEngine()

    def test_nested_field(self):
        rule = _make_rule(field="position.leverage", threshold=3.0)
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {
            "position": {"leverage": 5.0},
        })
        assert not decision.all_passed

    def test_deeply_nested(self):
        rule = _make_rule(field="a.b.c", threshold=10)
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {
            "a": {"b": {"c": 5}},
        })
        assert decision.all_passed

    def test_missing_nested_fails_closed(self):
        """B10: Missing nested field -> rule FAILS (fail-closed)."""
        rule = _make_rule(field="a.b.missing", threshold=10)
        ps = _make_policy_set(rules=[rule])
        self.engine.register(ps)

        decision = self.engine.evaluate("test_set", {
            "a": {"b": {"c": 5}},
        })
        assert not decision.all_passed
        assert "required_field_missing" in decision.failed_rules[0].reason


# ---------------------------------------------------------------------------
# PolicyEngine: evaluate_all
# ---------------------------------------------------------------------------


class TestEvaluateAll:
    def test_evaluates_all_sets(self):
        engine = PolicyEngine()

        ps1 = _make_policy_set(
            set_id="set1",
            name="Set 1",
            rules=[_make_rule(rule_id="r1", field="a", threshold=10)],
        )
        ps2 = _make_policy_set(
            set_id="set2",
            name="Set 2",
            rules=[_make_rule(rule_id="r2", field="b", threshold=10)],
        )
        engine.register(ps1)
        engine.register(ps2)

        decisions = engine.evaluate_all({"a": 5, "b": 20})
        assert len(decisions) == 2

        set1_decision = next(d for d in decisions if d.set_id == "set1")
        set2_decision = next(d for d in decisions if d.set_id == "set2")
        assert set1_decision.all_passed
        assert not set2_decision.all_passed


# ---------------------------------------------------------------------------
# PolicyStore tests
# ---------------------------------------------------------------------------


class TestPolicyStore:
    def test_save_and_get_active(self):
        store = PolicyStore()
        ps = _make_policy_set()
        store.save(ps)

        active = store.get_active("test_set")
        assert active is not None
        assert active.set_id == "test_set"

    def test_versioning(self):
        store = PolicyStore()
        v1 = _make_policy_set(version=1)
        v2 = _make_policy_set(version=2)
        store.save(v1)
        store.save(v2)

        assert store.active_version("test_set") == 2
        assert store.list_versions("test_set") == [1, 2]

    def test_activate_specific_version(self):
        store = PolicyStore()
        v1 = _make_policy_set(version=1)
        v2 = _make_policy_set(version=2)
        store.save(v1)
        store.save(v2)

        store.activate("test_set", 1)
        assert store.active_version("test_set") == 1

    def test_rollback(self):
        store = PolicyStore()
        v1 = _make_policy_set(version=1)
        v2 = _make_policy_set(version=2)
        store.save(v1)
        store.save(v2)

        rolled = store.rollback("test_set")
        assert rolled is not None
        assert store.active_version("test_set") == 1

    def test_rollback_at_earliest_returns_none(self):
        store = PolicyStore()
        v1 = _make_policy_set(version=1)
        store.save(v1)

        assert store.rollback("test_set") is None

    def test_set_mode(self):
        store = PolicyStore()
        ps = _make_policy_set(mode=PolicyMode.ENFORCED)
        store.save(ps)

        assert store.set_mode("test_set", PolicyMode.SHADOW)
        active = store.get_active("test_set")
        assert active.mode == PolicyMode.SHADOW

    def test_file_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PolicyStore(persist_dir=tmpdir)
            rules = [_make_rule(rule_id="r1")]
            ps = _make_policy_set(rules=rules)
            store.save(ps)

            # Check file exists
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1

            # Load into a new store
            store2 = PolicyStore(persist_dir=tmpdir)
            loaded = store2.load_from_dir()
            assert loaded == 1
            active = store2.get_active("test_set")
            assert active is not None
            assert len(active.rules) == 1

    def test_summary(self):
        store = PolicyStore()
        v1 = _make_policy_set(version=1, rules=[_make_rule()])
        v2 = _make_policy_set(version=2, rules=[_make_rule(), _make_rule(rule_id="r2")])
        store.save(v1, activate=False)
        store.save(v2, activate=True)

        summary = store.summary()
        assert len(summary) == 2
        active_entry = next(s for s in summary if s["active"])
        assert active_entry["version"] == 2


# ---------------------------------------------------------------------------
# Default policies tests
# ---------------------------------------------------------------------------


class TestDefaultPolicies:
    def test_pre_trade_policies_build(self):
        from agentic_trading.core.config import RiskConfig
        from agentic_trading.governance.default_policies import (
            build_pre_trade_policies,
        )

        config = RiskConfig()
        ps = build_pre_trade_policies(config)
        assert ps.set_id == "pre_trade_risk"
        assert len(ps.active_rules) >= 5

    def test_pre_trade_leverage_rule(self):
        from agentic_trading.core.config import RiskConfig
        from agentic_trading.governance.default_policies import (
            build_pre_trade_policies,
        )

        config = RiskConfig(max_portfolio_leverage=2.0)
        ps = build_pre_trade_policies(config)

        engine = PolicyEngine()
        engine.register(ps)

        # Full context so all rules have their required fields (B10 fix)
        base_ctx = {
            "position_pct_of_equity": 0.05,
            "order_notional_usd": 1000.0,
            "projected_exposure_pct": 0.5,
            "order_qty_above_min": True,
            "daily_loss_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "correlated_exposure_pct": 0.0,
        }

        # Within leverage limit
        d1 = engine.evaluate(
            "pre_trade_risk", {**base_ctx, "projected_leverage": 1.5},
        )
        assert d1.all_passed

        # Exceeds leverage limit
        d2 = engine.evaluate(
            "pre_trade_risk", {**base_ctx, "projected_leverage": 3.0},
        )
        assert not d2.all_passed

    def test_pre_trade_notional_rule(self):
        from agentic_trading.governance.default_policies import (
            build_pre_trade_policies,
        )

        ps = build_pre_trade_policies(max_notional=100_000)
        engine = PolicyEngine()
        engine.register(ps)

        d = engine.evaluate("pre_trade_risk", {"order_notional_usd": 200_000})
        assert not d.all_passed
        failed_ids = [r.rule_id for r in d.failed_rules]
        assert "max_order_notional" in failed_ids

    def test_post_trade_policies_build(self):
        from agentic_trading.governance.default_policies import (
            build_post_trade_policies,
        )

        ps = build_post_trade_policies()
        assert ps.set_id == "post_trade_risk"
        assert len(ps.active_rules) >= 4

    def test_shadow_mode_default_policies(self):
        from agentic_trading.governance.default_policies import (
            build_pre_trade_policies,
        )

        ps = build_pre_trade_policies(mode=PolicyMode.SHADOW)
        engine = PolicyEngine()
        engine.register(ps)

        # Even with violations, shadow mode never blocks
        d = engine.evaluate("pre_trade_risk", {
            "projected_leverage": 100.0,
            "order_notional_usd": 10_000_000,
        })
        assert d.all_passed
        assert len(d.shadow_violations) >= 2

    def test_strategy_constraint_policies_build(self):
        from agentic_trading.governance.default_policies import (
            build_strategy_constraint_policies,
        )

        ps = build_strategy_constraint_policies()
        assert ps.set_id == "strategy_constraints"
        assert len(ps.active_rules) >= 2


# ---------------------------------------------------------------------------
# GovernanceGate + PolicyEngine integration
# ---------------------------------------------------------------------------


class TestGovernanceGateWithPolicyEngine:
    @pytest.mark.asyncio
    async def test_policy_engine_blocks_via_gate(self):
        from agentic_trading.core.config import GovernanceConfig
        from agentic_trading.governance.drift_detector import DriftDetector
        from agentic_trading.governance.gate import GovernanceGate
        from agentic_trading.governance.health_score import HealthTracker
        from agentic_trading.governance.impact_classifier import ImpactClassifier
        from agentic_trading.governance.maturity import MaturityManager

        config = GovernanceConfig(enabled=True)

        # Create a policy engine with a strict notional limit
        engine = PolicyEngine()
        rules = [
            _make_rule(
                rule_id="strict_notional",
                field="order_notional_usd",
                threshold=10_000,
            ),
        ]
        ps = _make_policy_set(rules=rules, set_id="strict")
        engine.register(ps)

        gate = GovernanceGate(
            config=config,
            maturity=MaturityManager(config.maturity),
            health=HealthTracker(config.health_score),
            impact=ImpactClassifier(config.impact_classifier),
            drift=DriftDetector(config.drift_detector),
            policy_engine=engine,
        )

        # Promote strategy to L2 so maturity doesn't block
        gate.maturity._levels["test_strat"] = MaturityLevel.L4_AUTONOMOUS

        decision = await gate.evaluate(
            strategy_id="test_strat",
            symbol="BTC/USDT",
            notional_usd=50_000,  # exceeds policy limit of 10k
        )
        assert decision.action == GovernanceAction.BLOCK
        assert "policy_violation" in decision.reason

    @pytest.mark.asyncio
    async def test_gate_works_without_policy_engine(self):
        """GovernanceGate still works without a PolicyEngine (backward compat)."""
        from agentic_trading.core.config import GovernanceConfig
        from agentic_trading.governance.drift_detector import DriftDetector
        from agentic_trading.governance.gate import GovernanceGate
        from agentic_trading.governance.health_score import HealthTracker
        from agentic_trading.governance.impact_classifier import ImpactClassifier
        from agentic_trading.governance.maturity import MaturityManager

        config = GovernanceConfig(enabled=True)
        gate = GovernanceGate(
            config=config,
            maturity=MaturityManager(config.maturity),
            health=HealthTracker(config.health_score),
            impact=ImpactClassifier(config.impact_classifier),
            drift=DriftDetector(config.drift_detector),
            # No policy_engine
        )

        gate.maturity._levels["test_strat"] = MaturityLevel.L4_AUTONOMOUS

        decision = await gate.evaluate(
            strategy_id="test_strat",
            symbol="BTC/USDT",
            notional_usd=50_000,
        )
        assert decision.action == GovernanceAction.ALLOW

    @pytest.mark.asyncio
    async def test_shadow_policy_doesnt_block_gate(self):
        """Shadow policies are logged but don't affect the gate decision."""
        from agentic_trading.core.config import GovernanceConfig
        from agentic_trading.governance.drift_detector import DriftDetector
        from agentic_trading.governance.gate import GovernanceGate
        from agentic_trading.governance.health_score import HealthTracker
        from agentic_trading.governance.impact_classifier import ImpactClassifier
        from agentic_trading.governance.maturity import MaturityManager

        config = GovernanceConfig(enabled=True)

        engine = PolicyEngine()
        rules = [
            _make_rule(
                rule_id="shadow_notional",
                field="order_notional_usd",
                threshold=1_000,  # Very strict, but shadow
            ),
        ]
        ps = _make_policy_set(
            rules=rules,
            set_id="shadow_test",
            mode=PolicyMode.SHADOW,
        )
        engine.register(ps)

        gate = GovernanceGate(
            config=config,
            maturity=MaturityManager(config.maturity),
            health=HealthTracker(config.health_score),
            impact=ImpactClassifier(config.impact_classifier),
            drift=DriftDetector(config.drift_detector),
            policy_engine=engine,
        )
        gate.maturity._levels["test_strat"] = MaturityLevel.L4_AUTONOMOUS

        decision = await gate.evaluate(
            strategy_id="test_strat",
            symbol="BTC/USDT",
            notional_usd=50_000,  # Violates shadow rule
        )
        # Shadow should not block
        assert decision.action == GovernanceAction.ALLOW
        # But details should show shadow violation was recorded
        assert "policy_shadow_test" in decision.details
