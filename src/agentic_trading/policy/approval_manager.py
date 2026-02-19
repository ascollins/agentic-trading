"""Approval workflow manager for the governance framework.

Manages the lifecycle of approval requests:
1. **check_approval_required()** — evaluates rules to see if approval is needed.
2. **request_approval()** — creates a pending request and publishes event.
3. **approve() / reject()** — record decision and publish resolution event.
4. **escalate()** — bump to a higher escalation level.
5. **expire_stale()** — background sweep for timed-out requests.

The ApprovalManager integrates with the GovernanceGate: when the gate
detects a high-impact or policy-gated action, it asks the manager
whether approval is required before allowing execution.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from .approval_models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRule,
    ApprovalStatus,
    ApprovalSummary,
    ApprovalTrigger,
    EscalationLevel,
)

logger = logging.getLogger(__name__)


class ApprovalManager:
    """Manages the approval request lifecycle.

    Thread-safe within a single asyncio event loop (no locks needed).

    Args:
        rules: List of :class:`ApprovalRule` that determine when approval
               is required.
        event_bus: Optional event bus for publishing approval events.
        auto_approve_l1: If True, L1_AUTO escalation requests are
                         automatically approved (logged, not held).
    """

    def __init__(
        self,
        rules: list[ApprovalRule] | None = None,
        event_bus: Any = None,
        auto_approve_l1: bool = True,
    ) -> None:
        self._rules: list[ApprovalRule] = rules or []
        self._event_bus = event_bus
        self._auto_approve_l1 = auto_approve_l1

        # Request store: request_id → ApprovalRequest
        self._requests: dict[str, ApprovalRequest] = {}

        # Index: strategy_id → list of pending request_ids
        self._pending_by_strategy: dict[str, list[str]] = {}

        # Metrics
        self._total_auto_approved: int = 0
        self._decision_times: list[float] = []

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: ApprovalRule) -> None:
        """Add an approval rule."""
        self._rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID. Returns True if found and removed."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        return len(self._rules) < before

    @property
    def rules(self) -> list[ApprovalRule]:
        """Active rules."""
        return list(self._rules)

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------

    def check_approval_required(
        self, context: dict[str, Any],
    ) -> ApprovalRule | None:
        """Check if any approval rule matches the given context.

        Returns the *first* matching rule (highest priority wins), or
        None if no rule matches and the action can proceed directly.
        """
        for rule in self._rules:
            if rule.matches(context):
                return rule
        return None

    async def request_approval(
        self,
        strategy_id: str,
        symbol: str,
        trigger: ApprovalTrigger,
        escalation_level: EscalationLevel = EscalationLevel.L2_OPERATOR,
        notional_usd: float = 0.0,
        impact_tier: str = "low",
        reason: str = "",
        context: dict[str, Any] | None = None,
        order_intent_data: dict[str, Any] | None = None,
        ttl_seconds: int = 300,
        action_type: str = "order",
    ) -> ApprovalRequest:
        """Create and store an approval request.

        If escalation_level is L1_AUTO and auto_approve_l1 is True,
        the request is immediately auto-approved.

        Returns the ApprovalRequest (which may already be approved).
        """
        request = ApprovalRequest(
            strategy_id=strategy_id,
            symbol=symbol,
            action_type=action_type,
            trigger=trigger,
            escalation_level=escalation_level,
            notional_usd=notional_usd,
            impact_tier=impact_tier,
            reason=reason,
            context=context or {},
            order_intent_data=order_intent_data,
            ttl_seconds=ttl_seconds,
        )

        self._requests[request.request_id] = request

        # Index by strategy
        pending_list = self._pending_by_strategy.setdefault(strategy_id, [])
        pending_list.append(request.request_id)

        logger.info(
            "Approval requested: id=%s strategy=%s symbol=%s trigger=%s "
            "escalation=%s notional=%.2f",
            request.request_id,
            strategy_id,
            symbol,
            trigger.value,
            escalation_level.value,
            notional_usd,
        )

        # Publish event
        await self._publish_requested(request)

        # Auto-approve L1
        if (
            self._auto_approve_l1
            and escalation_level == EscalationLevel.L1_AUTO
        ):
            await self.approve(
                request.request_id,
                decided_by="system_auto",
                reason="L1_auto_approve",
            )
            self._total_auto_approved += 1

        return request

    async def approve(
        self,
        request_id: str,
        decided_by: str,
        reason: str = "",
        conditions: dict[str, Any] | None = None,
    ) -> ApprovalDecision | None:
        """Approve a pending request.

        Returns the decision, or None if request not found or already terminal.
        """
        request = self._requests.get(request_id)
        if request is None:
            logger.warning("Approval not found: %s", request_id)
            return None

        if request.is_terminal:
            logger.warning(
                "Cannot approve terminal request %s (status=%s)",
                request_id,
                request.status.value,
            )
            return None

        # Check expiry
        if request.is_expired:
            request.status = ApprovalStatus.EXPIRED
            await self._publish_resolved(request)
            return None

        now = datetime.now(UTC)
        request.status = ApprovalStatus.APPROVED
        request.decided_at = now
        request.decided_by = decided_by
        request.decision_reason = reason

        decision = ApprovalDecision(
            request_id=request_id,
            status=ApprovalStatus.APPROVED,
            decided_by=decided_by,
            decided_at=now,
            reason=reason,
            conditions=conditions or {},
        )

        # Track decision time
        dt = (now - request.created_at).total_seconds()
        self._decision_times.append(dt)

        # Remove from pending index
        self._remove_from_pending(request.strategy_id, request_id)

        logger.info(
            "Approval granted: id=%s by=%s reason=%s (%.1fs)",
            request_id,
            decided_by,
            reason,
            dt,
        )

        await self._publish_resolved(request)
        return decision

    async def reject(
        self,
        request_id: str,
        decided_by: str,
        reason: str = "",
    ) -> ApprovalDecision | None:
        """Reject a pending request.

        Returns the decision, or None if request not found or already terminal.
        """
        request = self._requests.get(request_id)
        if request is None:
            logger.warning("Approval not found: %s", request_id)
            return None

        if request.is_terminal:
            logger.warning(
                "Cannot reject terminal request %s (status=%s)",
                request_id,
                request.status.value,
            )
            return None

        now = datetime.now(UTC)
        request.status = ApprovalStatus.REJECTED
        request.decided_at = now
        request.decided_by = decided_by
        request.decision_reason = reason

        decision = ApprovalDecision(
            request_id=request_id,
            status=ApprovalStatus.REJECTED,
            decided_by=decided_by,
            decided_at=now,
            reason=reason,
        )

        dt = (now - request.created_at).total_seconds()
        self._decision_times.append(dt)

        self._remove_from_pending(request.strategy_id, request_id)

        logger.info(
            "Approval rejected: id=%s by=%s reason=%s (%.1fs)",
            request_id,
            decided_by,
            reason,
            dt,
        )

        await self._publish_resolved(request)
        return decision

    async def escalate(
        self,
        request_id: str,
        new_level: EscalationLevel,
        reason: str = "",
    ) -> bool:
        """Escalate a pending request to a higher level.

        Returns True if escalated, False if not found or already terminal.
        """
        request = self._requests.get(request_id)
        if request is None or request.is_terminal:
            return False

        old_level = request.escalation_level
        request.escalation_level = new_level
        request.status = ApprovalStatus.ESCALATED
        # Reset to pending so it can be approved at new level
        request.status = ApprovalStatus.PENDING

        logger.info(
            "Approval escalated: id=%s from=%s to=%s reason=%s",
            request_id,
            old_level.value,
            new_level.value,
            reason,
        )

        await self._publish_resolved(request, status_override="escalated")
        return True

    async def cancel(self, request_id: str, reason: str = "") -> bool:
        """Cancel a pending request. Returns True if cancelled."""
        request = self._requests.get(request_id)
        if request is None or request.is_terminal:
            return False

        request.status = ApprovalStatus.CANCELLED
        request.decided_at = datetime.now(UTC)
        request.decision_reason = reason

        self._remove_from_pending(request.strategy_id, request_id)

        logger.info("Approval cancelled: id=%s reason=%s", request_id, reason)
        await self._publish_resolved(request)
        return True

    # ------------------------------------------------------------------
    # Expiry management
    # ------------------------------------------------------------------

    async def expire_stale(self) -> list[str]:
        """Expire all pending requests that have exceeded their TTL.

        Returns list of expired request IDs. This should be called
        periodically (e.g., every 30 seconds) by a background agent.
        """
        expired_ids: list[str] = []

        for request_id, request in list(self._requests.items()):
            if request.status == ApprovalStatus.PENDING and request.is_expired:
                request.status = ApprovalStatus.EXPIRED
                request.decided_at = datetime.now(UTC)
                request.decision_reason = "ttl_expired"

                self._remove_from_pending(request.strategy_id, request_id)
                expired_ids.append(request_id)

                logger.info(
                    "Approval expired: id=%s strategy=%s ttl=%ds",
                    request_id,
                    request.strategy_id,
                    request.ttl_seconds,
                )
                await self._publish_resolved(request)

        return expired_ids

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get a request by ID."""
        return self._requests.get(request_id)

    def get_pending(
        self, strategy_id: str | None = None,
    ) -> list[ApprovalRequest]:
        """Get all pending requests, optionally filtered by strategy."""
        if strategy_id is not None:
            ids = self._pending_by_strategy.get(strategy_id, [])
            return [
                self._requests[rid]
                for rid in ids
                if rid in self._requests
                and self._requests[rid].status == ApprovalStatus.PENDING
            ]

        return [
            r for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        ]

    def has_pending(self, strategy_id: str) -> bool:
        """Check if a strategy has any pending approval requests."""
        return bool(self.get_pending(strategy_id))

    def get_summary(self) -> ApprovalSummary:
        """Get summary statistics."""
        counts: dict[str, int] = {s.value: 0 for s in ApprovalStatus}
        for r in self._requests.values():
            counts[r.status.value] = counts.get(r.status.value, 0) + 1

        avg_dt = 0.0
        if self._decision_times:
            avg_dt = sum(self._decision_times) / len(self._decision_times)

        return ApprovalSummary(
            total_requests=len(self._requests),
            pending=counts.get("pending", 0),
            approved=counts.get("approved", 0),
            rejected=counts.get("rejected", 0),
            expired=counts.get("expired", 0),
            escalated=counts.get("escalated", 0),
            cancelled=counts.get("cancelled", 0),
            auto_approved=self._total_auto_approved,
            avg_decision_time_seconds=round(avg_dt, 2),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _remove_from_pending(self, strategy_id: str, request_id: str) -> None:
        """Remove a request from the pending-by-strategy index."""
        pending = self._pending_by_strategy.get(strategy_id, [])
        if request_id in pending:
            pending.remove(request_id)

    async def _publish_requested(self, request: ApprovalRequest) -> None:
        """Publish an ApprovalRequested event."""
        if self._event_bus is None:
            return
        try:
            from agentic_trading.core.events import ApprovalRequested

            event = ApprovalRequested(
                request_id=request.request_id,
                strategy_id=request.strategy_id,
                symbol=request.symbol,
                action_type=request.action_type,
                trigger=request.trigger.value,
                escalation_level=request.escalation_level.value,
                notional_usd=request.notional_usd,
                impact_tier=request.impact_tier,
                reason=request.reason,
                ttl_seconds=request.ttl_seconds,
            )
            await self._event_bus.publish("governance.approval", event)
        except Exception:
            logger.warning("Failed to publish ApprovalRequested", exc_info=True)

    async def _publish_resolved(
        self,
        request: ApprovalRequest,
        status_override: str | None = None,
    ) -> None:
        """Publish an ApprovalResolved event."""
        if self._event_bus is None:
            return
        try:
            from agentic_trading.core.events import ApprovalResolved

            dt = 0.0
            if request.decided_at:
                dt = (request.decided_at - request.created_at).total_seconds()

            event = ApprovalResolved(
                request_id=request.request_id,
                strategy_id=request.strategy_id,
                symbol=request.symbol,
                status=status_override or request.status.value,
                decided_by=request.decided_by,
                reason=request.decision_reason,
                decision_time_seconds=round(dt, 2),
            )
            await self._event_bus.publish("governance.approval", event)
        except Exception:
            logger.warning("Failed to publish ApprovalResolved", exc_info=True)
