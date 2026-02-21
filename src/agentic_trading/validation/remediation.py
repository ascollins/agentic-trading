"""Remediation engine — what to do when validation fails.

State machine::

    PENDING → RETRYING → (success: RESOLVED)
                       → RE_RETRIEVING → RETRYING → RESOLVED
                                                  → ESCALATED
                                                  → EXHAUSTED

The engine is a pure decision function: it does NOT execute retries.
It returns the recommended action and an updated record.  The caller
(pipeline orchestrator or agent code) is responsible for executing.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from agentic_trading.core.ids import utc_now as _now

from .models import (
    RemediationAction,
    RemediationRecord,
    RemediationState,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class RemediationPolicy(BaseModel):
    """Per-output-type remediation configuration."""

    output_type: str = "*"
    max_retries: int = 2
    max_re_retrievals: int = 1
    auto_escalate_on_critical: bool = True
    min_severity_to_remediate: ValidationSeverity = ValidationSeverity.ERROR


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RemediationEngine:
    """Decides the next remediation action based on validation results.

    Does NOT execute the action (retry, re-retrieve, escalate).
    It only decides and records.
    """

    def __init__(
        self,
        policies: list[RemediationPolicy] | None = None,
    ) -> None:
        self._policies: dict[str, RemediationPolicy] = {}
        self._default_policy = RemediationPolicy()
        for p in policies or []:
            self._policies[p.output_type] = p

    def get_policy(self, output_type: str) -> RemediationPolicy:
        """Retrieve the remediation policy for an output type."""
        return self._policies.get(output_type, self._default_policy)

    def decide(
        self,
        validation_result: ValidationResult,
        current_record: RemediationRecord | None = None,
    ) -> tuple[RemediationAction, RemediationRecord]:
        """Decide the next remediation action.

        Returns
        -------
        tuple[RemediationAction, RemediationRecord]
            The recommended action and the updated record.
        """
        policy = self.get_policy(validation_result.output_type)
        record = current_record or RemediationRecord(
            validation_id=validation_result.validation_id,
            max_retries=policy.max_retries,
        )

        worst_severity = self._worst_severity(validation_result)

        # Already passed — accept with any warnings
        if validation_result.overall_passed:
            record.state = RemediationState.RESOLVED
            record.resolved_at = _now()
            record.resolution = "passed"
            return RemediationAction.ACCEPT_WITH_WARNINGS, record

        # Critical issues with auto-escalate
        if (
            worst_severity == ValidationSeverity.CRITICAL
            and policy.auto_escalate_on_critical
        ):
            record.state = RemediationState.ESCALATED
            record.actions_taken.append(
                {
                    "action": "escalate",
                    "reason": "critical_severity",
                    "timestamp": _now().isoformat(),
                }
            )
            return RemediationAction.ESCALATE, record

        # Retry path
        if record.retry_count < policy.max_retries:
            record.state = RemediationState.RETRYING
            record.retry_count += 1
            record.actions_taken.append(
                {
                    "action": "retry",
                    "attempt": record.retry_count,
                    "timestamp": _now().isoformat(),
                }
            )
            return RemediationAction.RETRY, record

        # Re-retrieve path (after retries exhausted)
        re_retrieval_count = sum(
            1
            for a in record.actions_taken
            if a.get("action") == "re_retrieve"
        )
        if re_retrieval_count < policy.max_re_retrievals:
            record.state = RemediationState.RE_RETRIEVING
            record.retry_count = 0  # Reset for next round
            record.actions_taken.append(
                {
                    "action": "re_retrieve",
                    "timestamp": _now().isoformat(),
                }
            )
            return RemediationAction.RE_RETRIEVE, record

        # All remediation exhausted
        has_evidence_issues = any(
            i.layer.value == "evidence"
            for i in validation_result.issues
            if i.severity
            in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        )
        if has_evidence_issues:
            record.state = RemediationState.EXHAUSTED
            record.resolved_at = _now()
            record.resolution = "insufficient_evidence"
            return RemediationAction.INSUFFICIENT_EVIDENCE, record

        # Default: escalate
        record.state = RemediationState.ESCALATED
        record.actions_taken.append(
            {
                "action": "escalate",
                "reason": "remediation_exhausted",
                "timestamp": _now().isoformat(),
            }
        )
        return RemediationAction.ESCALATE, record

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _worst_severity(
        result: ValidationResult,
    ) -> ValidationSeverity:
        """Return the worst severity across all issues."""
        severity_rank = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.ERROR: 2,
            ValidationSeverity.CRITICAL: 3,
        }
        worst = ValidationSeverity.INFO
        for issue in result.issues:
            if severity_rank.get(issue.severity, 0) > severity_rank.get(
                worst, 0
            ):
                worst = issue.severity
        return worst
