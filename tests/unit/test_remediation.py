"""Tests for RemediationEngine."""

from __future__ import annotations

from agentic_trading.validation.models import (
    RemediationAction,
    RemediationState,
    ValidationIssue,
    ValidationLayer,
    ValidationResult,
    ValidationSeverity,
)
from agentic_trading.validation.remediation import (
    RemediationEngine,
    RemediationPolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    passed: bool = False,
    issues: list[ValidationIssue] | None = None,
) -> ValidationResult:
    return ValidationResult(
        overall_passed=passed,
        issues=issues or [],
        output_type="Signal",
    )


def _make_issue(
    severity: ValidationSeverity = ValidationSeverity.ERROR,
    layer: ValidationLayer = ValidationLayer.SCHEMA,
) -> ValidationIssue:
    return ValidationIssue(
        layer=layer,
        severity=severity,
        code="test.issue",
        message="test issue",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRemediationEngineLayerName:
    def test_get_default_policy(self) -> None:
        engine = RemediationEngine()
        policy = engine.get_policy("Signal")
        assert policy.max_retries == 2
        assert policy.max_re_retrievals == 1

    def test_get_custom_policy(self) -> None:
        engine = RemediationEngine(
            policies=[
                RemediationPolicy(
                    output_type="Signal", max_retries=5
                )
            ]
        )
        policy = engine.get_policy("Signal")
        assert policy.max_retries == 5

    def test_fallback_to_default_for_unknown_type(self) -> None:
        engine = RemediationEngine(
            policies=[
                RemediationPolicy(
                    output_type="Signal", max_retries=5
                )
            ]
        )
        policy = engine.get_policy("UnknownType")
        assert policy.max_retries == 2  # default


class TestPassedResult:
    def test_passed_result_resolves(self) -> None:
        engine = RemediationEngine()
        result = _make_result(passed=True)
        action, record = engine.decide(result)
        assert action == RemediationAction.ACCEPT_WITH_WARNINGS
        assert record.state == RemediationState.RESOLVED
        assert record.resolution == "passed"


class TestCriticalEscalation:
    def test_critical_auto_escalates(self) -> None:
        engine = RemediationEngine()
        result = _make_result(
            issues=[_make_issue(ValidationSeverity.CRITICAL)]
        )
        action, record = engine.decide(result)
        assert action == RemediationAction.ESCALATE
        assert record.state == RemediationState.ESCALATED

    def test_critical_no_escalate_when_disabled(self) -> None:
        engine = RemediationEngine(
            policies=[
                RemediationPolicy(
                    output_type="Signal",
                    auto_escalate_on_critical=False,
                )
            ]
        )
        result = _make_result(
            issues=[_make_issue(ValidationSeverity.CRITICAL)]
        )
        action, record = engine.decide(result)
        # Should try retry first instead
        assert action == RemediationAction.RETRY


class TestRetryPath:
    def test_first_retry(self) -> None:
        engine = RemediationEngine()
        result = _make_result(
            issues=[_make_issue(ValidationSeverity.ERROR)]
        )
        action, record = engine.decide(result)
        assert action == RemediationAction.RETRY
        assert record.state == RemediationState.RETRYING
        assert record.retry_count == 1

    def test_second_retry(self) -> None:
        engine = RemediationEngine()
        result = _make_result(
            issues=[_make_issue(ValidationSeverity.ERROR)]
        )
        # First retry
        _, record = engine.decide(result)
        # Second retry
        action, record = engine.decide(result, current_record=record)
        assert action == RemediationAction.RETRY
        assert record.retry_count == 2

    def test_retries_exhausted_leads_to_re_retrieve(self) -> None:
        engine = RemediationEngine()
        result = _make_result(
            issues=[_make_issue(ValidationSeverity.ERROR)]
        )
        # Exhaust retries
        _, record = engine.decide(result)
        _, record = engine.decide(result, current_record=record)
        # Third attempt -> re-retrieve
        action, record = engine.decide(result, current_record=record)
        assert action == RemediationAction.RE_RETRIEVE
        assert record.state == RemediationState.RE_RETRIEVING
        assert record.retry_count == 0  # Reset


class TestReRetrievePath:
    def test_re_retrieve_resets_retry_counter(self) -> None:
        engine = RemediationEngine()
        result = _make_result(
            issues=[_make_issue(ValidationSeverity.ERROR)]
        )
        # Exhaust retries → re-retrieve
        _, record = engine.decide(result)
        _, record = engine.decide(result, current_record=record)
        _, record = engine.decide(result, current_record=record)
        assert record.retry_count == 0
        # Now can retry again
        action, record = engine.decide(result, current_record=record)
        assert action == RemediationAction.RETRY
        assert record.retry_count == 1


class TestExhaustion:
    def test_all_exhausted_with_evidence_issues(self) -> None:
        engine = RemediationEngine(
            policies=[
                RemediationPolicy(
                    output_type="Signal",
                    max_retries=1,
                    max_re_retrievals=0,
                )
            ]
        )
        evidence_issue = _make_issue(
            ValidationSeverity.ERROR,
            ValidationLayer.EVIDENCE,
        )
        result = _make_result(issues=[evidence_issue])
        # Exhaust single retry
        _, record = engine.decide(result)
        # No re-retrievals -> insufficient evidence
        action, record = engine.decide(result, current_record=record)
        assert action == RemediationAction.INSUFFICIENT_EVIDENCE
        assert record.state == RemediationState.EXHAUSTED

    def test_all_exhausted_without_evidence_escalates(self) -> None:
        engine = RemediationEngine(
            policies=[
                RemediationPolicy(
                    output_type="Signal",
                    max_retries=1,
                    max_re_retrievals=0,
                )
            ]
        )
        schema_issue = _make_issue(
            ValidationSeverity.ERROR,
            ValidationLayer.SCHEMA,
        )
        result = _make_result(issues=[schema_issue])
        _, record = engine.decide(result)
        action, record = engine.decide(result, current_record=record)
        assert action == RemediationAction.ESCALATE
        assert record.state == RemediationState.ESCALATED


class TestActionTracking:
    def test_actions_tracked_in_record(self) -> None:
        engine = RemediationEngine()
        result = _make_result(
            issues=[_make_issue(ValidationSeverity.ERROR)]
        )
        _, record = engine.decide(result)
        assert len(record.actions_taken) == 1
        assert record.actions_taken[0]["action"] == "retry"

        _, record = engine.decide(result, current_record=record)
        assert len(record.actions_taken) == 2

        _, record = engine.decide(result, current_record=record)
        assert len(record.actions_taken) == 3
        assert record.actions_taken[2]["action"] == "re_retrieve"


class TestRemediationPolicy:
    def test_defaults(self) -> None:
        policy = RemediationPolicy()
        assert policy.output_type == "*"
        assert policy.max_retries == 2
        assert policy.max_re_retrievals == 1
        assert policy.auto_escalate_on_critical is True
        assert policy.min_severity_to_remediate == ValidationSeverity.ERROR
