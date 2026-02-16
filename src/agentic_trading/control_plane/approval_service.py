"""Tiered approval workflow service for the institutional control plane.

Wraps the existing :class:`ApprovalManager` and provides the
:class:`IApprovalService` interface expected by :class:`ToolGateway`.

Responsibilities:
    1. Map CPPolicyDecision.tier → approval workflow action.
    2. T0_AUTONOMOUS → auto-approve (no human, no hold).
    3. T1_NOTIFY → execute immediately, post-hoc notification.
    4. T2_APPROVE → hold and create approval request, return pending.
    5. T3_DUAL_APPROVE → hold, require 2 approvals, return pending.
    6. Emit audit events for every decision.

The service is FAIL-CLOSED: any error results in a rejected decision.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.governance.approval_manager import ApprovalManager
from agentic_trading.governance.approval_models import (
    ApprovalTrigger,
    EscalationLevel,
)

from .action_types import (
    ApprovalDecision as CPApprovalDecision,
    ApprovalTier,
    AuditEntry,
    CPPolicyDecision,
    ProposedAction,
    ToolName,
)
from .audit_log import AuditLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier → EscalationLevel mapping
# ---------------------------------------------------------------------------

_TIER_TO_ESCALATION: dict[ApprovalTier, EscalationLevel] = {
    ApprovalTier.T0_AUTONOMOUS: EscalationLevel.L1_AUTO,
    ApprovalTier.T1_NOTIFY: EscalationLevel.L1_AUTO,
    ApprovalTier.T2_APPROVE: EscalationLevel.L2_OPERATOR,
    ApprovalTier.T3_DUAL_APPROVE: EscalationLevel.L3_RISK,
}

# ---------------------------------------------------------------------------
# Trigger inference
# ---------------------------------------------------------------------------

_REASON_TO_TRIGGER: dict[str, ApprovalTrigger] = {
    "max_notional": ApprovalTrigger.SIZE_THRESHOLD,
    "max_leverage": ApprovalTrigger.SIZE_THRESHOLD,
    "max_position": ApprovalTrigger.SIZE_THRESHOLD,
    "policy_violation": ApprovalTrigger.POLICY_VIOLATION,
    "maturity": ApprovalTrigger.MATURITY_GATE,
}


def _infer_trigger(reasons: list[str]) -> ApprovalTrigger:
    """Infer the approval trigger from policy decision reasons."""
    reason_text = " ".join(reasons).lower()
    for keyword, trigger in _REASON_TO_TRIGGER.items():
        if keyword in reason_text:
            return trigger
    return ApprovalTrigger.HIGH_IMPACT


class CPApprovalService:
    """Tiered approval service implementing IApprovalService.

    Construction:
        - approval_manager: The existing ApprovalManager instance.
        - audit_log: Optional AuditLog for recording approval decisions.
        - event_bus: Optional event bus for approval events.

    The service does NOT modify the ApprovalManager's rules;
    it only uses its workflow methods.
    """

    def __init__(
        self,
        approval_manager: ApprovalManager,
        audit_log: AuditLog | None = None,
        event_bus: Any = None,
    ) -> None:
        self._manager = approval_manager
        self._audit = audit_log
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # IApprovalService interface
    # ------------------------------------------------------------------

    async def request(
        self,
        policy_decision: CPPolicyDecision,
        proposed: ProposedAction,
    ) -> CPApprovalDecision:
        """Process an approval request based on the policy decision tier.

        Behavior by tier:
            T0_AUTONOMOUS: Auto-approved immediately.
            T1_NOTIFY: Auto-approved, notification emitted.
            T2_APPROVE: Held for single human approval.
            T3_DUAL_APPROVE: Held for dual human approval.

        Never raises. Errors are caught and result in rejection (fail-closed).
        """
        try:
            return await self._do_request(policy_decision, proposed)
        except Exception as exc:
            logger.error(
                "CPApprovalService: unhandled error — REJECTING (fail-closed): %s",
                exc,
                exc_info=True,
            )
            return CPApprovalDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                approved=False,
                tier=policy_decision.tier,
                reason=f"approval_service_internal_error: {exc}",
            )

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    async def _do_request(
        self,
        policy_decision: CPPolicyDecision,
        proposed: ProposedAction,
    ) -> CPApprovalDecision:
        """Core approval logic by tier."""
        tier = policy_decision.tier

        # T0: Auto-approve, no further action
        if tier == ApprovalTier.T0_AUTONOMOUS:
            decision = CPApprovalDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                approved=True,
                tier=tier,
                decided_by=["system:autonomous"],
                reason="T0_autonomous",
            )
            await self._audit_decision(proposed, decision)
            return decision

        # T1: Auto-approve, but emit notification
        if tier == ApprovalTier.T1_NOTIFY:
            decision = CPApprovalDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                approved=True,
                tier=tier,
                decided_by=["system:notify"],
                reason="T1_notify_auto_approved",
            )
            await self._audit_decision(proposed, decision)
            await self._emit_notification(proposed, policy_decision)
            return decision

        # T2/T3: Create approval request, return pending
        escalation = _TIER_TO_ESCALATION.get(tier, EscalationLevel.L2_OPERATOR)
        trigger = _infer_trigger(policy_decision.reasons)

        # Extract notional from context if available
        notional = policy_decision.context_snapshot.get(
            "order_notional_usd", 0.0,
        )
        if isinstance(notional, str):
            try:
                notional = float(notional)
            except (ValueError, TypeError):
                notional = 0.0

        approval_request = await self._manager.request_approval(
            strategy_id=proposed.scope.strategy_id,
            symbol=proposed.scope.symbol,
            trigger=trigger,
            escalation_level=escalation,
            notional_usd=float(notional),
            impact_tier=self._impact_from_tier(tier),
            reason="; ".join(policy_decision.reasons),
            context={
                "action_id": proposed.action_id,
                "tool_name": proposed.tool_name.value,
                "policy_decision_id": policy_decision.decision_id,
                "failed_rules": policy_decision.failed_rules,
            },
            order_intent_data=proposed.request_params,
            ttl_seconds=self._ttl_for_tier(tier),
        )

        # If auto-approved (L1_AUTO level in manager), return approved
        if approval_request.status.value == "approved":
            decision = CPApprovalDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                approved=True,
                tier=tier,
                decided_by=[approval_request.decided_by],
                reason=approval_request.decision_reason,
            )
            await self._audit_decision(proposed, decision)
            return decision

        # Otherwise, return pending
        decision = CPApprovalDecision(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            approved=False,
            tier=tier,
            reason=f"awaiting_{tier.value}_approval",
            pending_request_id=approval_request.request_id,
        )
        await self._audit_decision(proposed, decision)
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _impact_from_tier(tier: ApprovalTier) -> str:
        """Map tier to impact string for ApprovalManager."""
        if tier == ApprovalTier.T3_DUAL_APPROVE:
            return "critical"
        if tier == ApprovalTier.T2_APPROVE:
            return "high"
        if tier == ApprovalTier.T1_NOTIFY:
            return "medium"
        return "low"

    @staticmethod
    def _ttl_for_tier(tier: ApprovalTier) -> int:
        """Determine TTL in seconds based on tier.

        Higher tiers get longer TTLs since they may need more approvers.
        """
        if tier == ApprovalTier.T3_DUAL_APPROVE:
            return 600  # 10 minutes
        if tier == ApprovalTier.T2_APPROVE:
            return 300  # 5 minutes
        return 60  # 1 minute

    async def _audit_decision(
        self,
        proposed: ProposedAction,
        decision: CPApprovalDecision,
    ) -> None:
        """Record the approval decision in the audit log."""
        if self._audit is None:
            return
        try:
            await self._audit.append(AuditEntry(
                correlation_id=proposed.correlation_id,
                causation_id=proposed.action_id,
                actor=proposed.scope.actor,
                scope=proposed.scope,
                event_type="approval_decision",
                payload={
                    "action_id": proposed.action_id,
                    "approved": decision.approved,
                    "tier": decision.tier.value,
                    "reason": decision.reason,
                    "decided_by": decision.decided_by,
                    "pending_request_id": decision.pending_request_id,
                },
            ))
        except Exception:
            logger.warning("Failed to audit approval decision", exc_info=True)

    async def _emit_notification(
        self,
        proposed: ProposedAction,
        policy_decision: CPPolicyDecision,
    ) -> None:
        """Emit a T1 notification event."""
        if self._event_bus is None:
            return
        try:
            from agentic_trading.core.events import SystemHealth

            event = SystemHealth(
                component="control_plane.approval_notify",
                healthy=True,
                message=(
                    f"T1_NOTIFY: {proposed.tool_name.value} executed with "
                    f"policy violations: {'; '.join(policy_decision.reasons)}"
                ),
                details={
                    "action_id": proposed.action_id,
                    "tool_name": proposed.tool_name.value,
                    "tier": ApprovalTier.T1_NOTIFY.value,
                    "failed_rules": policy_decision.failed_rules,
                },
            )
            await self._event_bus.publish("system", event)
        except Exception:
            logger.warning("Failed to emit T1 notification", exc_info=True)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_pending_approvals(
        self, strategy_id: str | None = None,
    ) -> list[Any]:
        """Get pending approval requests."""
        return self._manager.get_pending(strategy_id)

    def has_pending(self, strategy_id: str) -> bool:
        """Check if a strategy has pending approvals."""
        return self._manager.has_pending(strategy_id)
