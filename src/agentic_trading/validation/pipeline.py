"""ValidationPipeline — composes validators and produces ValidationResult.

This is the main entry point.  Usage::

    pipeline = ValidationPipeline(
        validators=[SchemaValidator(), EvidenceValidator(), BusinessRuleValidator()],
        critique_validator=CritiqueValidator(config, call_llm),
        remediation_engine=RemediationEngine(),
    )
    result = pipeline.run(parsed_output, envelope, llm_result)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentic_trading.core.ids import content_hash as _content_hash
from agentic_trading.llm.envelope import LLMEnvelope, LLMResult

from .critique_validator import CritiqueValidator
from .evidence_validator import EvidenceValidator
from .models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from .protocol import IValidator
from .remediation import RemediationEngine

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Orchestrates multi-layer validation of LLM agent outputs.

    Runs validators in order.  If schema validation fails (hard fail),
    subsequent layers are skipped.  Evidence and business rules always
    run.  Critique runs conditionally.

    The pipeline computes:

    - Per-layer pass/fail
    - Aggregate quality_score (0.0 to 1.0)
    - Recommended remediation action
    """

    def __init__(
        self,
        validators: list[IValidator] | None = None,
        critique_validator: CritiqueValidator | None = None,
        remediation_engine: RemediationEngine | None = None,
    ) -> None:
        self._validators: list[IValidator] = validators or []
        self._critique = critique_validator
        self._remediation = remediation_engine or RemediationEngine()

    def run(
        self,
        parsed_output: dict[str, Any],
        envelope: LLMEnvelope,
        llm_result: LLMResult,
    ) -> ValidationResult:
        """Run the full validation pipeline.

        Also populates ``llm_result.validation_passed`` and
        ``validation_errors`` for backward compatibility.
        """
        t_start = time.monotonic()
        all_issues: list[ValidationIssue] = []
        schema_passed = True
        evidence_passed = True
        business_passed = True

        # Run validators in order
        for validator in self._validators:
            issues = validator.validate(
                parsed_output, envelope, llm_result
            )
            all_issues.extend(issues)

            has_errors = any(
                i.severity
                in (
                    ValidationSeverity.ERROR,
                    ValidationSeverity.CRITICAL,
                )
                for i in issues
            )

            if validator.layer_name == "schema":
                schema_passed = not has_errors
                if not schema_passed:
                    logger.warning(
                        "Schema validation failed; "
                        "skipping remaining layers"
                    )
                    break
            elif validator.layer_name == "evidence":
                evidence_passed = not has_errors
            elif validator.layer_name == "business_rule":
                business_passed = not has_errors

        # Compute interim quality score
        quality_score = self._compute_quality_score(
            all_issues, schema_passed
        )

        # Run critique if applicable
        critique_passed: bool | None = None
        if self._critique and schema_passed:
            critique_issues = self._critique.validate(
                parsed_output,
                envelope,
                llm_result,
                prior_quality_score=quality_score,
            )
            if critique_issues:
                all_issues.extend(critique_issues)
                critique_passed = not any(
                    i.severity
                    in (
                        ValidationSeverity.ERROR,
                        ValidationSeverity.CRITICAL,
                    )
                    for i in critique_issues
                )
            # If critique was not triggered, stays None

        # Final quality score (post-critique)
        quality_score = self._compute_quality_score(
            all_issues, schema_passed
        )

        # Overall pass/fail
        overall_passed = (
            schema_passed and evidence_passed and business_passed
        )
        if critique_passed is not None:
            overall_passed = overall_passed and critique_passed

        # Extract claims from evidence validator
        claims = []
        for validator in self._validators:
            if isinstance(validator, EvidenceValidator):
                available_sources = {
                    e.source for e in envelope.retrieved_evidence
                }
                output_type = parsed_output.get("_output_type", "")
                claims = validator.extract_claims(
                    parsed_output, output_type, available_sources
                )
                break

        result = ValidationResult(
            trace_id=envelope.trace_id,
            envelope_id=envelope.envelope_id,
            agent_id=envelope.agent_id,
            agent_type=envelope.agent_type,
            output_type=parsed_output.get("_output_type", ""),
            schema_passed=schema_passed,
            evidence_passed=evidence_passed,
            business_rules_passed=business_passed,
            critique_passed=critique_passed,
            overall_passed=overall_passed,
            quality_score=quality_score,
            issues=all_issues,
            claims=claims,
            latency_ms=(time.monotonic() - t_start) * 1000,
            output_hash=_content_hash(str(parsed_output)),
        )

        # Compute remediation recommendation
        if not overall_passed:
            action, record = self._remediation.decide(result)
            result.recommended_action = action
            result.remediation = record

        # Sync back to LLMResult for backward compatibility
        llm_result.validation_passed = overall_passed
        llm_result.validation_errors = [
            f"[{i.layer.value}:{i.severity.value}] {i.message}"
            for i in all_issues
            if i.severity
            in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        ]

        return result

    # ------------------------------------------------------------------
    # Quality score
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_quality_score(
        issues: list[ValidationIssue],
        schema_passed: bool,
    ) -> float:
        """Compute a 0.0–1.0 quality score from issues.

        Scoring:
        - Start at 1.0
        - Schema failure: 0.0 immediately
        - Each CRITICAL: −0.3
        - Each ERROR: −0.15
        - Each WARNING: −0.05
        - Floor at 0.0
        """
        if not schema_passed:
            return 0.0

        score = 1.0
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                score -= 0.3
            elif issue.severity == ValidationSeverity.ERROR:
                score -= 0.15
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 0.05

        return max(0.0, min(1.0, score))
