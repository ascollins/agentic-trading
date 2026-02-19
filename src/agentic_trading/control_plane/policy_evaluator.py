"""Deterministic policy evaluator for the institutional control plane.

Wraps the existing :class:`PolicyEngine` and provides the
:class:`IPolicyEvaluator` interface expected by :class:`ToolGateway`.

Responsibilities:
    1. Map a ProposedAction to a PolicyEngine evaluation context.
    2. Evaluate all registered policy sets against the context.
    3. Determine the required ApprovalTier from the evaluation result.
    4. Build a CPPolicyDecision (immutable, hashable, replayable).

The evaluator is DETERMINISTIC: given the same ProposedAction and the
same policy sets, it will always produce the same CPPolicyDecision.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from agentic_trading.core.enums import GovernanceAction
from agentic_trading.governance.policy_engine import PolicyEngine
from agentic_trading.governance.policy_models import PolicyDecision, PolicyMode

from .action_types import (
    ApprovalTier,
    CPPolicyDecision,
    DegradedMode,
    MUTATING_TOOLS,
    ProposedAction,
    ToolName,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier mapping: GovernanceAction severity -> approval tier
# ---------------------------------------------------------------------------

# Actions that definitely need human approval
_BLOCK_ACTIONS = frozenset({
    GovernanceAction.BLOCK,
    GovernanceAction.PAUSE,
    GovernanceAction.KILL,
})

# Actions that reduce size (lower severity)
_REDUCE_ACTIONS = frozenset({
    GovernanceAction.REDUCE_SIZE,
    GovernanceAction.DEMOTE,
})

# Tools that are protective (TP/SL, cancels) get fast-path T0
_PROTECTIVE_TOOLS = frozenset({
    ToolName.CANCEL_ORDER,
    ToolName.CANCEL_ALL_ORDERS,
    ToolName.SET_TRADING_STOP,
})

# Degraded mode: tools allowed per mode
_READ_ONLY_TOOLS = frozenset({
    ToolName.GET_POSITIONS,
    ToolName.GET_BALANCES,
    ToolName.GET_OPEN_ORDERS,
    ToolName.GET_INSTRUMENT,
    ToolName.GET_FUNDING_RATE,
    ToolName.GET_CLOSED_PNL,
})

_RISK_OFF_TOOLS = _READ_ONLY_TOOLS | frozenset({
    ToolName.CANCEL_ORDER,
    ToolName.CANCEL_ALL_ORDERS,
    ToolName.SET_TRADING_STOP,
})


class CPPolicyEvaluator:
    """Deterministic policy evaluator implementing IPolicyEvaluator.

    Construction:
        - policy_engine: The existing PolicyEngine with registered policy sets.
        - context_builder: Optional callable that enriches the context dict
          from ProposedAction.request_params. If None, request_params is used
          directly.
        - tier_overrides: Optional dict mapping ToolName -> forced ApprovalTier.
          Used for design decisions like "TP/SL always T0".

    The evaluator does NOT modify the PolicyEngine; it only reads from it.
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        context_builder: Any = None,
        tier_overrides: dict[ToolName, ApprovalTier] | None = None,
        default_tier: ApprovalTier = ApprovalTier.T0_AUTONOMOUS,
    ) -> None:
        self._engine = policy_engine
        self._context_builder = context_builder
        self._tier_overrides = tier_overrides or {}
        self._default_tier = default_tier
        self._system_state: dict[str, str] = {
            "degraded_mode": DegradedMode.NORMAL.value,
        }

    # ------------------------------------------------------------------
    # IPolicyEvaluator interface
    # ------------------------------------------------------------------

    def evaluate(self, proposed: ProposedAction) -> CPPolicyDecision:
        """Evaluate all registered policy sets against the proposed action.

        Never raises. Exceptions are caught and result in BLOCK (fail-closed).
        """
        try:
            return self._do_evaluate(proposed)
        except Exception as exc:
            logger.error(
                "CPPolicyEvaluator: unhandled error — BLOCKING (fail-closed): %s",
                exc,
                exc_info=True,
            )
            return CPPolicyDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                allowed=False,
                tier=ApprovalTier.T0_AUTONOMOUS,
                sizing_multiplier=0.0,
                reasons=[f"policy_evaluator_internal_error: {exc}"],
                policy_set_version="error",
            )

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _do_evaluate(self, proposed: ProposedAction) -> CPPolicyDecision:
        """Core evaluation logic."""
        # 0. Degraded mode pre-check — blocks before any policy evaluation
        degraded_decision = self._check_degraded_mode(proposed)
        if degraded_decision is not None:
            return degraded_decision

        # 1. Build context from ProposedAction
        context = self._build_context(proposed)

        # 2. Check for tier overrides (e.g., protective tools → T0)
        forced_tier = self._tier_overrides.get(proposed.tool_name)

        # 3. If no policy sets registered, allow with default tier
        registered = self._engine.registered_sets
        if not registered:
            return CPPolicyDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                allowed=True,
                tier=forced_tier or self._default_tier,
                sizing_multiplier=1.0,
                reasons=["no_policy_sets_registered"],
                policy_set_version="none",
                context_snapshot=context,
            )

        # 4. Evaluate all registered policy sets
        all_decisions: list[PolicyDecision] = self._engine.evaluate_all(context)

        # 5. Aggregate results
        all_passed = True
        sizing_multiplier = 1.0
        reasons: list[str] = []
        failed_rules: list[str] = []
        shadow_violations: list[str] = []
        most_severe_action = GovernanceAction.ALLOW
        versions: list[str] = []

        for decision in all_decisions:
            versions.append(f"{decision.set_id}:v{decision.version}")

            if not decision.all_passed:
                all_passed = False

                for fail in decision.failed_rules:
                    failed_rules.append(f"{decision.set_id}/{fail.rule_id}")
                    reasons.append(fail.reason)

                # Track most severe action
                if _action_severity(decision.action) > _action_severity(most_severe_action):
                    most_severe_action = decision.action

                # Take minimum sizing multiplier
                sizing_multiplier = min(
                    sizing_multiplier, decision.sizing_multiplier,
                )

            for sv in decision.shadow_violations:
                shadow_violations.append(f"{decision.set_id}/{sv.rule_id}")

        # 6. Determine tier
        if forced_tier is not None:
            tier = forced_tier
        elif not all_passed:
            tier = self._tier_for_action(most_severe_action)
        else:
            tier = self._default_tier

        # 7. Compute snapshot hash for replay verification
        snapshot_hash = hashlib.sha256(
            json.dumps(context, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # 8. Determine if action is allowed
        # If any BLOCK/KILL/PAUSE action: not allowed
        allowed = True
        if most_severe_action in _BLOCK_ACTIONS:
            allowed = False
        # REDUCE_SIZE is allowed but with reduced sizing
        elif most_severe_action in _REDUCE_ACTIONS:
            allowed = True

        return CPPolicyDecision(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            allowed=allowed,
            tier=tier,
            sizing_multiplier=sizing_multiplier,
            reasons=reasons if reasons else ["all_rules_passed"],
            failed_rules=failed_rules,
            shadow_violations=shadow_violations,
            policy_set_version="; ".join(versions),
            snapshot_hash=snapshot_hash,
            context_snapshot=context,
        )

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_context(self, proposed: ProposedAction) -> dict[str, Any]:
        """Build the evaluation context from ProposedAction.

        Merges scope fields (strategy_id, symbol, exchange) with
        request_params to form the context dict that PolicyEngine expects.
        """
        # Start with request_params
        context: dict[str, Any] = dict(proposed.request_params)

        # Inject scope fields (always available)
        context.setdefault("strategy_id", proposed.scope.strategy_id)
        context.setdefault("symbol", proposed.scope.symbol)
        context.setdefault("exchange", proposed.scope.exchange)
        context.setdefault("actor", proposed.scope.actor)
        context.setdefault("tool_name", proposed.tool_name.value)

        # If a custom context builder is provided, use it to enrich
        if self._context_builder is not None:
            enriched = self._context_builder(proposed, context)
            if enriched is not None:
                context = enriched

        return context

    # ------------------------------------------------------------------
    # Tier determination
    # ------------------------------------------------------------------

    @staticmethod
    def _tier_for_action(action: GovernanceAction) -> ApprovalTier:
        """Map a GovernanceAction to the required approval tier.

        Design decisions:
        - BLOCK/PAUSE/KILL → T2_APPROVE (hold for human)
        - REDUCE_SIZE/DEMOTE → T1_NOTIFY (execute reduced, notify)
        - ALLOW → T0_AUTONOMOUS
        """
        if action in _BLOCK_ACTIONS:
            return ApprovalTier.T2_APPROVE
        if action in _REDUCE_ACTIONS:
            return ApprovalTier.T1_NOTIFY
        return ApprovalTier.T0_AUTONOMOUS

    # ------------------------------------------------------------------
    # Degraded mode enforcement
    # ------------------------------------------------------------------

    def _check_degraded_mode(
        self, proposed: ProposedAction,
    ) -> CPPolicyDecision | None:
        """Check if the current degraded mode blocks this tool.

        Returns a BLOCK CPPolicyDecision if blocked, or None to continue.
        """
        mode_str = self._system_state.get("degraded_mode", "normal")
        try:
            mode = DegradedMode(mode_str)
        except ValueError:
            mode = DegradedMode.NORMAL

        if mode == DegradedMode.NORMAL:
            return None

        tool = proposed.tool_name

        if mode == DegradedMode.FULL_STOP:
            return CPPolicyDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                allowed=False,
                tier=ApprovalTier.T0_AUTONOMOUS,
                sizing_multiplier=0.0,
                reasons=[f"degraded_mode_full_stop: all tools blocked"],
                policy_set_version="system",
            )

        if mode == DegradedMode.READ_ONLY:
            if tool not in _READ_ONLY_TOOLS:
                return CPPolicyDecision(
                    action_id=proposed.action_id,
                    correlation_id=proposed.correlation_id,
                    allowed=False,
                    tier=ApprovalTier.T0_AUTONOMOUS,
                    sizing_multiplier=0.0,
                    reasons=[f"degraded_mode_read_only: {tool.value} blocked"],
                    policy_set_version="system",
                )
            return None  # Read-only tools allowed

        if mode == DegradedMode.RISK_OFF_ONLY:
            if tool not in _RISK_OFF_TOOLS:
                return CPPolicyDecision(
                    action_id=proposed.action_id,
                    correlation_id=proposed.correlation_id,
                    allowed=False,
                    tier=ApprovalTier.T0_AUTONOMOUS,
                    sizing_multiplier=0.0,
                    reasons=[f"degraded_mode_risk_off: {tool.value} blocked"],
                    policy_set_version="system",
                )
            return None  # Risk-off tools allowed

        return None

    # ------------------------------------------------------------------
    # System state management
    # ------------------------------------------------------------------

    def set_system_state(self, key: str, value: str) -> None:
        """Set a system state value (e.g., degraded_mode)."""
        self._system_state[key] = value

    def get_system_state(self, key: str) -> str:
        """Get a system state value."""
        return self._system_state.get(key, "")

    @property
    def degraded_mode(self) -> str:
        """Current degraded mode level."""
        return self._system_state.get("degraded_mode", DegradedMode.NORMAL.value)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def registered_policy_sets(self) -> list[str]:
        """List registered policy set IDs."""
        return self._engine.registered_sets

    @property
    def tier_overrides(self) -> dict[ToolName, ApprovalTier]:
        """Current tier overrides."""
        return dict(self._tier_overrides)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACTION_SEVERITY: dict[GovernanceAction, int] = {
    GovernanceAction.ALLOW: 0,
    GovernanceAction.REDUCE_SIZE: 1,
    GovernanceAction.DEMOTE: 2,
    GovernanceAction.BLOCK: 3,
    GovernanceAction.PAUSE: 4,
    GovernanceAction.KILL: 5,
}


def _action_severity(action: GovernanceAction) -> int:
    return _ACTION_SEVERITY.get(action, 0)


def build_default_evaluator(
    policy_engine: PolicyEngine | None = None,
) -> CPPolicyEvaluator:
    """Build a CPPolicyEvaluator with sensible defaults.

    - Protective tools (cancel, TP/SL) → T0 fast-path
    - All registered policy sets evaluated
    - Default tier T0_AUTONOMOUS
    """
    engine = policy_engine or PolicyEngine()

    tier_overrides = {
        ToolName.CANCEL_ORDER: ApprovalTier.T0_AUTONOMOUS,
        ToolName.CANCEL_ALL_ORDERS: ApprovalTier.T0_AUTONOMOUS,
        ToolName.SET_TRADING_STOP: ApprovalTier.T0_AUTONOMOUS,
    }

    return CPPolicyEvaluator(
        policy_engine=engine,
        tier_overrides=tier_overrides,
        default_tier=ApprovalTier.T0_AUTONOMOUS,
    )
