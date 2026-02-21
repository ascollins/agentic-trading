"""Validation framework models.

Defines the core data structures for multi-layer validation of LLM
agent outputs:

- ValidationSeverity: Issue severity levels
- ValidationLayer: Which validator produced an issue
- ValidationIssue: A single validation problem
- ClaimAnnotation: Links a factual claim to evidence or marks it assumption
- CritiqueResult: Output from second-model review
- RemediationAction / RemediationState: Remediation state machine
- ValidationResult: Aggregate result from the full pipeline
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.ids import new_id as _uuid
from agentic_trading.core.ids import utc_now as _now

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ValidationSeverity(str, Enum):
    """Severity of a validation issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationLayer(str, Enum):
    """Which validation layer produced an issue."""

    SCHEMA = "schema"
    EVIDENCE = "evidence"
    BUSINESS_RULE = "business_rule"
    CRITIQUE = "critique"


class ClaimType(str, Enum):
    """Classification of a factual claim in agent output."""

    CITED = "cited"
    ASSUMPTION = "assumption"
    UNCITED = "uncited"
    DERIVED = "derived"


class RemediationAction(str, Enum):
    """Actions the remediation engine can recommend."""

    RETRY = "retry"
    RE_RETRIEVE = "re_retrieve"
    ESCALATE = "escalate"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ACCEPT_WITH_WARNINGS = "accept_with_warnings"


class RemediationState(str, Enum):
    """State in the remediation lifecycle."""

    PENDING = "pending"
    RETRYING = "retrying"
    RE_RETRIEVING = "re_retrieving"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    EXHAUSTED = "exhausted"


# ---------------------------------------------------------------------------
# Issue model
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    """A single validation problem detected by any layer."""

    issue_id: str = Field(default_factory=_uuid)
    layer: ValidationLayer
    severity: ValidationSeverity
    code: str
    message: str
    field_path: str = ""
    expected: Any = None
    actual: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Claim annotation model
# ---------------------------------------------------------------------------


class ClaimAnnotation(BaseModel):
    """Links a factual claim in agent output to its evidence basis."""

    claim_id: str = Field(default_factory=_uuid)
    field_path: str
    claim_text: str
    claim_type: ClaimType
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    validator_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Critique result
# ---------------------------------------------------------------------------


class CritiqueResult(BaseModel):
    """Output from the second-model critique layer."""

    critique_id: str = Field(default_factory=_uuid)
    model_used: str = ""
    triggered_by: str = ""
    overall_score: float = 0.0
    issues_found: list[ValidationIssue] = Field(default_factory=list)
    reasoning: str = ""
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Remediation record
# ---------------------------------------------------------------------------


class RemediationRecord(BaseModel):
    """Tracks the remediation lifecycle for a failed validation."""

    remediation_id: str = Field(default_factory=_uuid)
    validation_id: str
    state: RemediationState = RemediationState.PENDING
    actions_taken: list[dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2
    started_at: datetime = Field(default_factory=_now)
    resolved_at: datetime | None = None
    resolution: str = ""


# ---------------------------------------------------------------------------
# Aggregate validation result
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Aggregate result from the full validation pipeline.

    This is the primary output.  It captures all layer results,
    claims analysis, optional critique, and the final disposition.
    """

    validation_id: str = Field(default_factory=_uuid)
    trace_id: str = ""
    envelope_id: str = ""
    agent_id: str = ""
    agent_type: str = ""
    output_type: str = ""
    timestamp: datetime = Field(default_factory=_now)

    # Per-layer results
    schema_passed: bool = False
    evidence_passed: bool = False
    business_rules_passed: bool = False
    critique_passed: bool | None = None

    # Aggregate
    overall_passed: bool = False
    quality_score: float = 0.0
    issues: list[ValidationIssue] = Field(default_factory=list)
    claims: list[ClaimAnnotation] = Field(default_factory=list)
    critique: CritiqueResult | None = None

    # Remediation
    remediation: RemediationRecord | None = None
    recommended_action: RemediationAction = RemediationAction.ACCEPT_WITH_WARNINGS

    # Timing
    latency_ms: float = 0.0

    # Integrity
    output_hash: str = ""

    @property
    def error_count(self) -> int:
        """Count of ERROR and CRITICAL issues."""
        return sum(
            1
            for i in self.issues
            if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        """Count of WARNING issues."""
        return sum(
            1 for i in self.issues if i.severity == ValidationSeverity.WARNING
        )

    @property
    def uncited_claims(self) -> list[ClaimAnnotation]:
        """Claims classified as UNCITED."""
        return [c for c in self.claims if c.claim_type == ClaimType.UNCITED]
