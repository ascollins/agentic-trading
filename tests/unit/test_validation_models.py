"""Tests for validation framework models."""

from __future__ import annotations

from agentic_trading.validation.models import (
    ClaimAnnotation,
    ClaimType,
    CritiqueResult,
    RemediationAction,
    RemediationRecord,
    RemediationState,
    ValidationIssue,
    ValidationLayer,
    ValidationResult,
    ValidationSeverity,
)

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestValidationSeverity:
    def test_has_four_members(self) -> None:
        assert len(ValidationSeverity) == 4

    def test_expected_values(self) -> None:
        assert ValidationSeverity.INFO == "info"
        assert ValidationSeverity.WARNING == "warning"
        assert ValidationSeverity.ERROR == "error"
        assert ValidationSeverity.CRITICAL == "critical"


class TestValidationLayer:
    def test_has_four_members(self) -> None:
        assert len(ValidationLayer) == 4

    def test_expected_values(self) -> None:
        assert ValidationLayer.SCHEMA == "schema"
        assert ValidationLayer.EVIDENCE == "evidence"
        assert ValidationLayer.BUSINESS_RULE == "business_rule"
        assert ValidationLayer.CRITIQUE == "critique"


class TestClaimType:
    def test_has_four_members(self) -> None:
        assert len(ClaimType) == 4

    def test_expected_values(self) -> None:
        assert ClaimType.CITED == "cited"
        assert ClaimType.ASSUMPTION == "assumption"
        assert ClaimType.UNCITED == "uncited"
        assert ClaimType.DERIVED == "derived"


class TestRemediationAction:
    def test_has_five_members(self) -> None:
        assert len(RemediationAction) == 5

    def test_expected_values(self) -> None:
        assert RemediationAction.RETRY == "retry"
        assert RemediationAction.RE_RETRIEVE == "re_retrieve"
        assert RemediationAction.ESCALATE == "escalate"
        assert RemediationAction.INSUFFICIENT_EVIDENCE == "insufficient_evidence"
        assert RemediationAction.ACCEPT_WITH_WARNINGS == "accept_with_warnings"


class TestRemediationState:
    def test_has_six_members(self) -> None:
        assert len(RemediationState) == 6


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestValidationIssue:
    def test_defaults(self) -> None:
        issue = ValidationIssue(
            layer=ValidationLayer.SCHEMA,
            severity=ValidationSeverity.ERROR,
            code="schema.missing_field",
            message="Field 'x' is required",
        )
        assert issue.issue_id != ""
        assert issue.field_path == ""
        assert issue.expected is None
        assert issue.actual is None
        assert issue.metadata == {}

    def test_full_construction(self) -> None:
        issue = ValidationIssue(
            layer=ValidationLayer.BUSINESS_RULE,
            severity=ValidationSeverity.WARNING,
            code="business_rule.range.confidence",
            message="Confidence out of range",
            field_path="confidence",
            expected=[0.0, 1.0],
            actual=1.5,
            metadata={"rule_id": "conf_range"},
        )
        assert issue.field_path == "confidence"
        assert issue.actual == 1.5

    def test_serialization_roundtrip(self) -> None:
        issue = ValidationIssue(
            layer=ValidationLayer.EVIDENCE,
            severity=ValidationSeverity.INFO,
            code="evidence.unused",
            message="test",
        )
        data = issue.model_dump_json()
        restored = ValidationIssue.model_validate_json(data)
        assert restored.layer == issue.layer
        assert restored.code == issue.code


class TestClaimAnnotation:
    def test_defaults(self) -> None:
        claim = ClaimAnnotation(
            field_path="thesis",
            claim_text="BTC is bullish",
            claim_type=ClaimType.UNCITED,
        )
        assert claim.claim_id != ""
        assert claim.evidence_ids == []
        assert claim.confidence == 1.0
        assert claim.validator_confidence == 0.0

    def test_cited_claim(self) -> None:
        claim = ClaimAnnotation(
            field_path="stop_loss",
            claim_text="42000",
            claim_type=ClaimType.CITED,
            evidence_ids=["candle_history", "indicator_values"],
        )
        assert len(claim.evidence_ids) == 2

    def test_serialization_roundtrip(self) -> None:
        claim = ClaimAnnotation(
            field_path="thesis",
            claim_text="test",
            claim_type=ClaimType.ASSUMPTION,
        )
        data = claim.model_dump_json()
        restored = ClaimAnnotation.model_validate_json(data)
        assert restored.claim_type == ClaimType.ASSUMPTION


class TestCritiqueResult:
    def test_defaults(self) -> None:
        cr = CritiqueResult()
        assert cr.critique_id != ""
        assert cr.model_used == ""
        assert cr.overall_score == 0.0
        assert cr.issues_found == []
        assert cr.reasoning == ""
        assert cr.cost_usd == 0.0

    def test_with_issues(self) -> None:
        issue = ValidationIssue(
            layer=ValidationLayer.CRITIQUE,
            severity=ValidationSeverity.WARNING,
            code="critique.logic",
            message="Contradiction found",
        )
        cr = CritiqueResult(
            model_used="claude-haiku",
            overall_score=0.4,
            issues_found=[issue],
            reasoning="Logic error detected",
        )
        assert len(cr.issues_found) == 1
        assert cr.overall_score == 0.4


class TestRemediationRecord:
    def test_defaults(self) -> None:
        record = RemediationRecord(validation_id="val-123")
        assert record.remediation_id != ""
        assert record.state == RemediationState.PENDING
        assert record.actions_taken == []
        assert record.retry_count == 0
        assert record.max_retries == 2
        assert record.resolved_at is None

    def test_serialization_roundtrip(self) -> None:
        record = RemediationRecord(
            validation_id="val-456",
            state=RemediationState.RETRYING,
            retry_count=1,
        )
        data = record.model_dump_json()
        restored = RemediationRecord.model_validate_json(data)
        assert restored.state == RemediationState.RETRYING
        assert restored.retry_count == 1


# ---------------------------------------------------------------------------
# ValidationResult tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_defaults(self) -> None:
        result = ValidationResult()
        assert result.validation_id != ""
        assert result.overall_passed is False
        assert result.quality_score == 0.0
        assert result.issues == []
        assert result.claims == []
        assert result.critique is None
        assert result.remediation is None
        assert result.recommended_action == RemediationAction.ACCEPT_WITH_WARNINGS

    def test_error_count(self) -> None:
        result = ValidationResult(
            issues=[
                ValidationIssue(
                    layer=ValidationLayer.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    code="e1",
                    message="err",
                ),
                ValidationIssue(
                    layer=ValidationLayer.EVIDENCE,
                    severity=ValidationSeverity.WARNING,
                    code="w1",
                    message="warn",
                ),
                ValidationIssue(
                    layer=ValidationLayer.BUSINESS_RULE,
                    severity=ValidationSeverity.CRITICAL,
                    code="c1",
                    message="crit",
                ),
                ValidationIssue(
                    layer=ValidationLayer.EVIDENCE,
                    severity=ValidationSeverity.INFO,
                    code="i1",
                    message="info",
                ),
            ]
        )
        assert result.error_count == 2  # ERROR + CRITICAL
        assert result.warning_count == 1

    def test_uncited_claims(self) -> None:
        result = ValidationResult(
            claims=[
                ClaimAnnotation(
                    field_path="a",
                    claim_text="cited",
                    claim_type=ClaimType.CITED,
                ),
                ClaimAnnotation(
                    field_path="b",
                    claim_text="uncited one",
                    claim_type=ClaimType.UNCITED,
                ),
                ClaimAnnotation(
                    field_path="c",
                    claim_text="uncited two",
                    claim_type=ClaimType.UNCITED,
                ),
            ]
        )
        assert len(result.uncited_claims) == 2

    def test_serialization_roundtrip(self) -> None:
        result = ValidationResult(
            overall_passed=True,
            quality_score=0.85,
            output_type="Signal",
        )
        data = result.model_dump_json()
        restored = ValidationResult.model_validate_json(data)
        assert restored.overall_passed is True
        assert restored.quality_score == 0.85
        assert restored.output_type == "Signal"
