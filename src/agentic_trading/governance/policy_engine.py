"""Policy-as-code engine.

Evaluates declarative :class:`PolicySet` rules against an order context
to produce :class:`PolicyDecision` results.  This replaces imperative
risk-checking logic with a data-driven approach that supports:

- Declarative rule definitions (field, operator, threshold)
- Shadow mode (log-only, no blocking)
- Per-decision audit trail
- Scoped rules (by strategy, symbol, exchange)
- Versioned policy sets

Usage::

    engine = PolicyEngine()
    engine.register(pre_trade_policy_set)
    engine.register(compliance_policy_set)

    decision = engine.evaluate(
        set_id="pre_trade_risk_v1",
        context={
            "order_notional_usd": 75_000,
            "projected_leverage": 2.5,
            "strategy_id": "trend_following",
            "symbol": "BTC/USDT",
        },
    )
    if not decision.all_passed:
        # order should be blocked or resized
        pass
"""

from __future__ import annotations

import logging
import operator as op
from typing import Any

from agentic_trading.core.enums import GovernanceAction

from .policy_models import (
    Operator,
    PolicyDecision,
    PolicyEvalResult,
    PolicyMode,
    PolicyRule,
    PolicySet,
)

logger = logging.getLogger(__name__)

# Map Operator enum to Python comparison functions
_OPS: dict[Operator, Any] = {
    Operator.LT: op.lt,
    Operator.LE: op.le,
    Operator.GT: op.gt,
    Operator.GE: op.ge,
    Operator.EQ: op.eq,
    Operator.NE: op.ne,
}


class PolicyEngine:
    """Evaluates declarative policy rules against order contexts.

    The engine maintains a registry of :class:`PolicySet` instances
    and evaluates them on demand.  Each evaluation produces a
    :class:`PolicyDecision` with full audit trail.
    """

    def __init__(self) -> None:
        self._policy_sets: dict[str, PolicySet] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, policy_set: PolicySet) -> None:
        """Register (or replace) a policy set."""
        self._policy_sets[policy_set.set_id] = policy_set
        logger.info(
            "PolicyEngine: registered %s v%d (%s, %d rules)",
            policy_set.set_id,
            policy_set.version,
            policy_set.mode.value,
            len(policy_set.active_rules),
        )

    def unregister(self, set_id: str) -> PolicySet | None:
        """Remove a policy set. Returns it or None."""
        return self._policy_sets.pop(set_id, None)

    def get_policy_set(self, set_id: str) -> PolicySet | None:
        """Look up a policy set by ID."""
        return self._policy_sets.get(set_id)

    @property
    def registered_sets(self) -> list[str]:
        """List all registered policy set IDs."""
        return list(self._policy_sets.keys())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        set_id: str,
        context: dict[str, Any],
    ) -> PolicyDecision:
        """Evaluate all rules in a policy set against the given context.

        Args:
            set_id: The policy set to evaluate.
            context: Key-value pairs representing the order/trade context.
                Common keys: ``order_notional_usd``, ``projected_leverage``,
                ``strategy_id``, ``symbol``, ``exchange``, ``portfolio_pct``,
                ``order_qty``, ``is_reduce_only``, etc.

        Returns:
            A :class:`PolicyDecision` with per-rule results and
            an aggregate action/sizing_multiplier.

        Raises:
            KeyError: If ``set_id`` is not registered.
        """
        policy_set = self._policy_sets.get(set_id)
        if policy_set is None:
            raise KeyError(f"Policy set not registered: {set_id}")

        is_shadow = policy_set.mode == PolicyMode.SHADOW

        results: list[PolicyEvalResult] = []
        failed: list[PolicyEvalResult] = []
        shadow_violations: list[PolicyEvalResult] = []

        for rule in policy_set.active_rules:
            # Check scope: skip rules that don't apply
            if not self._rule_in_scope(rule, context):
                continue

            result = self._evaluate_rule(rule, context, is_shadow)
            results.append(result)

            if not result.passed:
                if is_shadow or result.shadow:
                    shadow_violations.append(result)
                else:
                    failed.append(result)

        # Determine aggregate action
        all_passed = len(failed) == 0
        action = GovernanceAction.ALLOW
        reason = "all_rules_passed"
        sizing_multiplier = 1.0

        if failed:
            # Use the most severe action from failed rules
            action = self._most_severe_action(failed)
            reasons = [r.reason for r in failed]
            reason = "; ".join(reasons)

            if action == GovernanceAction.REDUCE_SIZE:
                sizing_multiplier = 0.5
            elif action in (
                GovernanceAction.BLOCK,
                GovernanceAction.PAUSE,
                GovernanceAction.KILL,
            ):
                sizing_multiplier = 0.0

        # Log shadow violations
        for sv in shadow_violations:
            logger.info(
                "PolicyEngine [SHADOW] violation: set=%s rule=%s "
                "field=%s value=%s threshold=%s",
                set_id,
                sv.rule_id,
                sv.field,
                sv.field_value,
                sv.threshold,
            )

        return PolicyDecision(
            set_id=policy_set.set_id,
            set_name=policy_set.name,
            version=policy_set.version,
            mode=policy_set.mode,
            all_passed=all_passed,
            action=action,
            reason=reason,
            sizing_multiplier=sizing_multiplier,
            results=results,
            failed_rules=failed,
            shadow_violations=shadow_violations,
            context_snapshot=dict(context),
        )

    def evaluate_all(
        self,
        context: dict[str, Any],
    ) -> list[PolicyDecision]:
        """Evaluate all registered policy sets against the context.

        Returns a list of :class:`PolicyDecision`, one per set.
        """
        decisions: list[PolicyDecision] = []
        for set_id in self._policy_sets:
            decisions.append(self.evaluate(set_id, context))
        return decisions

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rule(
        self,
        rule: PolicyRule,
        context: dict[str, Any],
        shadow: bool,
    ) -> PolicyEvalResult:
        """Evaluate a single rule against the context."""
        field_value = self._resolve_field(rule.field, context)

        if field_value is None:
            # B10 FIX: Fail-closed. Missing field = FAIL the rule.
            return PolicyEvalResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=False,
                field=rule.field,
                field_value=None,
                threshold=rule.threshold,
                operator=rule.operator.value,
                action=rule.action,
                reason=f"required_field_missing: {rule.field}",
                policy_type=rule.policy_type,
                severity=rule.severity,
                shadow=shadow,
            )

        passed = self._check_condition(
            field_value, rule.operator, rule.threshold,
        )

        reason = ""
        if not passed:
            reason = (
                f"{rule.name}: {rule.field}={field_value} "
                f"violates {rule.operator.value} {rule.threshold}"
            )

        return PolicyEvalResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            passed=passed,
            field=rule.field,
            field_value=field_value,
            threshold=rule.threshold,
            operator=rule.operator.value,
            action=rule.action if not passed else GovernanceAction.ALLOW,
            reason=reason,
            policy_type=rule.policy_type,
            severity=rule.severity,
            shadow=shadow,
        )

    @staticmethod
    def _resolve_field(
        field: str, context: dict[str, Any],
    ) -> Any:
        """Resolve a dot-path field from the context.

        Supports simple keys (``"order_notional_usd"``) and
        dot-paths (``"position.leverage"``).
        """
        if "." not in field:
            return context.get(field)

        parts = field.split(".")
        current: Any = context
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current

    @staticmethod
    def _check_condition(
        value: Any, operator: Operator, threshold: Any,
    ) -> bool:
        """Evaluate a comparison condition. Returns True if passed."""
        try:
            if operator in _OPS:
                return _OPS[operator](value, threshold)

            if operator == Operator.IN:
                return value in threshold

            if operator == Operator.NOT_IN:
                return value not in threshold

            if operator == Operator.BETWEEN:
                low, high = threshold
                return low <= value <= high

        except (TypeError, ValueError):
            # B11 FIX: Fail-closed. Type mismatch = FAIL the condition.
            logger.warning(
                "PolicyEngine: type mismatch evaluating %s %s %s — FAILING (fail-closed)",
                value,
                operator.value,
                threshold,
            )
            return False

        # Unknown operator: fail-closed
        logger.warning(
            "PolicyEngine: unknown operator %s — FAILING (fail-closed)",
            operator.value,
        )
        return False

    @staticmethod
    def _rule_in_scope(
        rule: PolicyRule, context: dict[str, Any],
    ) -> bool:
        """Check if a rule applies to the given context."""
        if rule.strategy_ids is not None:
            strategy_id = context.get("strategy_id")
            if strategy_id and strategy_id not in rule.strategy_ids:
                return False

        if rule.symbols is not None:
            symbol = context.get("symbol")
            if symbol and symbol not in rule.symbols:
                return False

        if rule.exchanges is not None:
            exchange = context.get("exchange")
            if exchange and exchange not in rule.exchanges:
                return False

        return True

    @staticmethod
    def _most_severe_action(
        failed: list[PolicyEvalResult],
    ) -> GovernanceAction:
        """Return the most severe action from a list of failed rules."""
        severity_order = [
            GovernanceAction.KILL,
            GovernanceAction.PAUSE,
            GovernanceAction.BLOCK,
            GovernanceAction.DEMOTE,
            GovernanceAction.REDUCE_SIZE,
        ]
        for action in severity_order:
            for result in failed:
                if result.action == action:
                    return action
        return GovernanceAction.BLOCK
