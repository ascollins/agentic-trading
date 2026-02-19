"""Approval workflow models for the governance framework.

Provides declarative approval request/response lifecycle for
high-impact or policy-gated actions.  Inspired by Soteria's
multi-level approval gates where critical agent actions require
human or senior-agent sign-off before execution.

Key concepts:
- **ApprovalRequest**: A pending action that needs sign-off.
- **ApprovalStatus**: Lifecycle states (pending → approved/rejected/expired/escalated).
- **ApprovalRule**: Declarative rule that decides *when* approval is required.
- **ApprovalDecision**: The recorded approve/reject decision with metadata.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ApprovalStatus(str, Enum):
    """Lifecycle state of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


class ApprovalTrigger(str, Enum):
    """What triggered the approval requirement."""

    HIGH_IMPACT = "high_impact"
    CRITICAL_IMPACT = "critical_impact"
    POLICY_VIOLATION = "policy_violation"
    MATURITY_GATE = "maturity_gate"
    MANUAL_HOLD = "manual_hold"
    STRATEGY_OVERRIDE = "strategy_override"
    SIZE_THRESHOLD = "size_threshold"
    NEW_SYMBOL = "new_symbol"


class EscalationLevel(str, Enum):
    """Escalation tiers for approval routing."""

    L1_AUTO = "L1_auto"          # Auto-approve (logged only)
    L2_OPERATOR = "L2_operator"  # Human operator approval
    L3_RISK = "L3_risk"          # Risk team approval
    L4_ADMIN = "L4_admin"        # Admin / CTO approval


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class ApprovalRequest(BaseModel):
    """A pending action that requires approval before execution.

    Created by the governance gate when an action hits an approval
    trigger (e.g., high-impact trade, policy violation in gated mode).
    """

    request_id: str = Field(default_factory=_uuid)
    created_at: datetime = Field(default_factory=_now)
    expires_at: datetime | None = None
    ttl_seconds: int = 300  # 5 minutes default

    # What needs approval
    strategy_id: str
    symbol: str
    action_type: str = "order"  # "order", "parameter_change", "promotion"
    trigger: ApprovalTrigger = ApprovalTrigger.HIGH_IMPACT
    escalation_level: EscalationLevel = EscalationLevel.L2_OPERATOR

    # Context for the approver
    notional_usd: float = 0.0
    impact_tier: str = "low"
    sizing_multiplier: float = 1.0
    context: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    # Lifecycle
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_at: datetime | None = None
    decided_by: str = ""
    decision_reason: str = ""

    # Original order intent (serialised for replay after approval)
    order_intent_data: dict[str, Any] | None = None

    @property
    def is_expired(self) -> bool:
        """Check if the request has exceeded its TTL."""
        elapsed = (datetime.now(UTC) - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    @property
    def is_terminal(self) -> bool:
        """Whether the request has reached a final state."""
        return self.status in (
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
            ApprovalStatus.EXPIRED,
            ApprovalStatus.CANCELLED,
        )


class ApprovalRule(BaseModel):
    """Declarative rule that determines when approval is required.

    Rules are evaluated against trade context. If any rule matches,
    the action is held for approval rather than executed immediately.
    """

    rule_id: str
    name: str
    description: str = ""
    enabled: bool = True

    # Matching conditions
    trigger: ApprovalTrigger
    escalation_level: EscalationLevel = EscalationLevel.L2_OPERATOR

    # Thresholds (any that apply)
    min_notional_usd: float | None = None
    impact_tiers: list[str] | None = None  # ["high", "critical"]
    strategy_ids: list[str] | None = None
    symbols: list[str] | None = None
    maturity_levels: list[str] | None = None  # ["L2_gated"]

    # Auto-approve at L1 (log-only, no human needed)
    auto_approve: bool = False
    ttl_seconds: int = 300

    def matches(self, context: dict[str, Any]) -> bool:
        """Check if this rule matches the given trade context.

        All specified conditions must be met (AND logic).
        Returns False if the rule is disabled.
        """
        if not self.enabled:
            return False

        # Check notional threshold
        if self.min_notional_usd is not None:
            notional = context.get("notional_usd", 0.0)
            if notional < self.min_notional_usd:
                return False

        # Check impact tier
        if self.impact_tiers is not None:
            impact = context.get("impact_tier", "low")
            if impact not in self.impact_tiers:
                return False

        # Check strategy scope
        if self.strategy_ids is not None:
            strategy = context.get("strategy_id", "")
            if strategy not in self.strategy_ids:
                return False

        # Check symbol scope
        if self.symbols is not None:
            symbol = context.get("symbol", "")
            if symbol not in self.symbols:
                return False

        # Check maturity level
        if self.maturity_levels is not None:
            maturity = context.get("maturity_level", "")
            if maturity not in self.maturity_levels:
                return False

        return True


class ApprovalDecision(BaseModel):
    """Recorded decision on an approval request."""

    request_id: str
    status: ApprovalStatus
    decided_by: str
    decided_at: datetime = Field(default_factory=_now)
    reason: str = ""
    conditions: dict[str, Any] = Field(default_factory=dict)


class ApprovalSummary(BaseModel):
    """Summary statistics for approval workflow health."""

    total_requests: int = 0
    pending: int = 0
    approved: int = 0
    rejected: int = 0
    expired: int = 0
    escalated: int = 0
    cancelled: int = 0
    auto_approved: int = 0
    avg_decision_time_seconds: float = 0.0
