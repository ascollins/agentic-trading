"""Core type definitions for the institutional control plane.

All side effects flow through typed actions:
    ProposedAction -> PolicyDecision -> ApprovalDecision -> ToolCallResult

Every type is immutable (Pydantic model) and includes hashes for
audit integrity.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Tool allowlist
# ---------------------------------------------------------------------------


class ToolName(str, Enum):
    """Allowlisted tool names. No other tools exist."""

    # Mutating (require policy + approval + audit)
    SUBMIT_ORDER = "submit_order"
    CANCEL_ORDER = "cancel_order"
    CANCEL_ALL_ORDERS = "cancel_all_orders"
    AMEND_ORDER = "amend_order"
    BATCH_SUBMIT_ORDERS = "batch_submit_orders"
    SET_TRADING_STOP = "set_trading_stop"
    SET_LEVERAGE = "set_leverage"
    SET_POSITION_MODE = "set_position_mode"

    # Read-only (audited, no policy required)
    GET_POSITIONS = "get_positions"
    GET_BALANCES = "get_balances"
    GET_OPEN_ORDERS = "get_open_orders"
    GET_INSTRUMENT = "get_instrument"
    GET_FUNDING_RATE = "get_funding_rate"
    GET_CLOSED_PNL = "get_closed_pnl"


MUTATING_TOOLS: frozenset[ToolName] = frozenset({
    ToolName.SUBMIT_ORDER,
    ToolName.CANCEL_ORDER,
    ToolName.CANCEL_ALL_ORDERS,
    ToolName.AMEND_ORDER,
    ToolName.BATCH_SUBMIT_ORDERS,
    ToolName.SET_TRADING_STOP,
    ToolName.SET_LEVERAGE,
    ToolName.SET_POSITION_MODE,
})


# ---------------------------------------------------------------------------
# Approval tiers
# ---------------------------------------------------------------------------


class ApprovalTier(str, Enum):
    """Approval tiers for side effects."""

    T0_AUTONOMOUS = "T0_autonomous"       # No human needed
    T1_NOTIFY = "T1_notify"               # Execute + notify
    T2_APPROVE = "T2_approve"             # Hold until 1 approval
    T3_DUAL_APPROVE = "T3_dual_approve"   # Hold until 2 approvals


TIER_RANK: dict[ApprovalTier, int] = {
    ApprovalTier.T0_AUTONOMOUS: 0,
    ApprovalTier.T1_NOTIFY: 1,
    ApprovalTier.T2_APPROVE: 2,
    ApprovalTier.T3_DUAL_APPROVE: 3,
}


# ---------------------------------------------------------------------------
# Degraded modes
# ---------------------------------------------------------------------------


class DegradedMode(str, Enum):
    """System degraded mode levels."""

    NORMAL = "normal"
    RISK_OFF_ONLY = "risk_off_only"   # Cancel/reduce only
    READ_ONLY = "read_only"           # No mutations at all
    FULL_STOP = "full_stop"           # Nothing (not even reads)


# ---------------------------------------------------------------------------
# Action scope
# ---------------------------------------------------------------------------


class ActionScope(BaseModel):
    """Scope of a proposed action. Used for policy scoping and audit."""

    strategy_id: str = ""
    symbol: str = ""
    exchange: str = "bybit"
    actor: str = ""  # agent_id that proposed this


# ---------------------------------------------------------------------------
# ProposedAction
# ---------------------------------------------------------------------------


class ProposedAction(BaseModel):
    """An agent's request to perform a side effect. Immutable after creation."""

    action_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    correlation_id: str = Field(default_factory=_uuid)
    causation_id: str = ""  # event_id that caused this action

    tool_name: ToolName
    scope: ActionScope = Field(default_factory=ActionScope)
    request_params: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""

    # Computed at creation
    request_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.request_hash:
            payload = (
                f"{self.tool_name.value}:{self.idempotency_key}:"
                f"{self.scope.model_dump_json()}"
            )
            self.request_hash = hashlib.sha256(
                payload.encode()
            ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------


class CPPolicyDecision(BaseModel):
    """Result of deterministic policy evaluation. Immutable.

    Named CPPolicyDecision to avoid collision with existing
    governance.policy_models.PolicyDecision during migration.
    """

    decision_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    action_id: str = ""  # references ProposedAction.action_id
    correlation_id: str = ""

    allowed: bool
    tier: ApprovalTier = ApprovalTier.T0_AUTONOMOUS
    sizing_multiplier: float = 1.0
    reasons: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    shadow_violations: list[str] = Field(default_factory=list)

    policy_set_version: str = ""
    snapshot_hash: str = ""  # hash of the context snapshot used

    # For replay: exact inputs used
    context_snapshot: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ApprovalDecision
# ---------------------------------------------------------------------------


class ApprovalDecision(BaseModel):
    """Result of the approval service check."""

    approval_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    action_id: str = ""
    correlation_id: str = ""

    approved: bool
    tier: ApprovalTier = ApprovalTier.T0_AUTONOMOUS
    decided_by: list[str] = Field(default_factory=list)
    reason: str = ""

    # If pending, this is the request ID to poll
    pending_request_id: str | None = None
    expires_at: datetime | None = None


# ---------------------------------------------------------------------------
# ToolCallResult
# ---------------------------------------------------------------------------


class ToolCallResult(BaseModel):
    """Result of a ToolGateway.call() invocation."""

    result_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    action_id: str = ""
    correlation_id: str = ""

    tool_name: ToolName
    success: bool
    response: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    response_hash: str = ""

    # Timing
    latency_ms: float = 0.0

    # Idempotency: was this a replay of a previous call?
    was_idempotent_replay: bool = False


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """Single entry in the append-only audit log."""

    entry_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    correlation_id: str = ""
    causation_id: str = ""
    actor: str = ""
    scope: ActionScope | None = None

    event_type: str  # "proposed_action", "policy_evaluated", etc.
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.payload_hash:
            raw = json.dumps(self.payload, sort_keys=True, default=str)
            self.payload_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
