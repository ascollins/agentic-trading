"""Tests for ValidationPipeline orchestrator."""

from __future__ import annotations

from typing import Any

from agentic_trading.llm.envelope import EvidenceItem, LLMEnvelope, LLMResult
from agentic_trading.policy.models import Operator
from agentic_trading.validation.business_rules import (
    BusinessRule,
    BusinessRuleSet,
    BusinessRuleType,
    BusinessRuleValidator,
)
from agentic_trading.validation.critique_validator import (
    CritiqueTriggerConfig,
    CritiqueValidator,
)
from agentic_trading.validation.evidence_validator import EvidenceValidator
from agentic_trading.validation.models import (
    RemediationAction,
    ValidationIssue,
    ValidationLayer,
    ValidationSeverity,
)
from agentic_trading.validation.pipeline import ValidationPipeline
from agentic_trading.validation.remediation import (
    RemediationEngine,
)
from agentic_trading.validation.schema_validator import SchemaValidator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    evidence: list[EvidenceItem] | None = None,
    context: dict[str, Any] | None = None,
) -> LLMEnvelope:
    return LLMEnvelope(
        instructions="test",
        retrieved_evidence=evidence or [],
        context=context or {},
    )


def _make_result() -> LLMResult:
    return LLMResult(envelope_id="test-envelope")


def _make_evidence(source: str) -> EvidenceItem:
    return EvidenceItem(
        source=source,
        content={"data": "value"},
        relevance=0.9,
    )


class _StubValidator:
    """Stub validator that returns configurable issues."""

    def __init__(
        self,
        name: str,
        issues: list[ValidationIssue] | None = None,
    ) -> None:
        self._name = name
        self._issues = issues or []

    @property
    def layer_name(self) -> str:
        return self._name

    def validate(
        self,
        parsed_output: dict[str, Any],
        envelope: LLMEnvelope,
        result: LLMResult,
    ) -> list[ValidationIssue]:
        return list(self._issues)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineCleanPass:
    def test_all_pass_when_no_validators(self) -> None:
        pipeline = ValidationPipeline()
        llm_result = _make_result()
        result = pipeline.run({}, _make_envelope(), llm_result)
        assert result.overall_passed is True
        assert result.quality_score == 1.0
        assert result.issues == []
        assert llm_result.validation_passed is True
        assert llm_result.validation_errors == []

    def test_all_pass_with_clean_validators(self) -> None:
        pipeline = ValidationPipeline(
            validators=[
                _StubValidator("schema"),
                _StubValidator("evidence"),
                _StubValidator("business_rule"),
            ]
        )
        result = pipeline.run({}, _make_envelope(), _make_result())
        assert result.overall_passed is True
        assert result.schema_passed is True
        assert result.evidence_passed is True
        assert result.business_rules_passed is True


class TestSchemaHardFail:
    def test_schema_failure_skips_other_layers(self) -> None:
        schema_fail = _StubValidator(
            "schema",
            [
                ValidationIssue(
                    layer=ValidationLayer.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    code="schema.missing",
                    message="required field missing",
                )
            ],
        )
        evidence_ok = _StubValidator("evidence")
        biz_ok = _StubValidator("business_rule")

        pipeline = ValidationPipeline(
            validators=[schema_fail, evidence_ok, biz_ok]
        )
        result = pipeline.run({}, _make_envelope(), _make_result())

        assert result.overall_passed is False
        assert result.schema_passed is False
        assert result.quality_score == 0.0
        assert len(result.issues) == 1


class TestQualityScore:
    def test_schema_failure_gives_zero(self) -> None:
        assert (
            ValidationPipeline._compute_quality_score(
                [], schema_passed=False
            )
            == 0.0
        )

    def test_no_issues_gives_one(self) -> None:
        assert (
            ValidationPipeline._compute_quality_score(
                [], schema_passed=True
            )
            == 1.0
        )

    def test_warning_deduction(self) -> None:
        issues = [
            ValidationIssue(
                layer=ValidationLayer.EVIDENCE,
                severity=ValidationSeverity.WARNING,
                code="test",
                message="warn",
            )
        ]
        score = ValidationPipeline._compute_quality_score(
            issues, schema_passed=True
        )
        assert score == 0.95

    def test_error_deduction(self) -> None:
        issues = [
            ValidationIssue(
                layer=ValidationLayer.BUSINESS_RULE,
                severity=ValidationSeverity.ERROR,
                code="test",
                message="err",
            )
        ]
        score = ValidationPipeline._compute_quality_score(
            issues, schema_passed=True
        )
        assert score == 0.85

    def test_critical_deduction(self) -> None:
        issues = [
            ValidationIssue(
                layer=ValidationLayer.CRITIQUE,
                severity=ValidationSeverity.CRITICAL,
                code="test",
                message="crit",
            )
        ]
        score = ValidationPipeline._compute_quality_score(
            issues, schema_passed=True
        )
        assert score == 0.7

    def test_score_floors_at_zero(self) -> None:
        issues = [
            ValidationIssue(
                layer=ValidationLayer.SCHEMA,
                severity=ValidationSeverity.CRITICAL,
                code="test",
                message="crit",
            )
            for _ in range(10)
        ]
        score = ValidationPipeline._compute_quality_score(
            issues, schema_passed=True
        )
        assert score == 0.0


class TestBackwardCompatibility:
    def test_llm_result_updated_on_pass(self) -> None:
        pipeline = ValidationPipeline()
        llm_result = _make_result()
        pipeline.run({}, _make_envelope(), llm_result)
        assert llm_result.validation_passed is True
        assert llm_result.validation_errors == []

    def test_llm_result_updated_on_fail(self) -> None:
        pipeline = ValidationPipeline(
            validators=[
                _StubValidator(
                    "schema",
                    [
                        ValidationIssue(
                            layer=ValidationLayer.SCHEMA,
                            severity=ValidationSeverity.ERROR,
                            code="schema.test",
                            message="test error message",
                        )
                    ],
                )
            ]
        )
        llm_result = _make_result()
        pipeline.run({}, _make_envelope(), llm_result)
        assert llm_result.validation_passed is False
        assert len(llm_result.validation_errors) == 1
        assert "[schema:error]" in llm_result.validation_errors[0]


class TestRemediationIntegration:
    def test_failed_result_has_remediation(self) -> None:
        pipeline = ValidationPipeline(
            validators=[
                _StubValidator(
                    "evidence",
                    [
                        ValidationIssue(
                            layer=ValidationLayer.EVIDENCE,
                            severity=ValidationSeverity.ERROR,
                            code="evidence.test",
                            message="uncited",
                        )
                    ],
                )
            ],
            remediation_engine=RemediationEngine(),
        )
        result = pipeline.run({}, _make_envelope(), _make_result())
        assert result.overall_passed is False
        assert result.remediation is not None
        assert result.recommended_action == RemediationAction.RETRY

    def test_passed_result_no_remediation(self) -> None:
        pipeline = ValidationPipeline()
        result = pipeline.run({}, _make_envelope(), _make_result())
        assert result.overall_passed is True
        assert result.remediation is None


class TestCritiqueIntegration:
    def test_critique_not_triggered_when_absent(self) -> None:
        pipeline = ValidationPipeline(
            validators=[_StubValidator("schema")]
        )
        result = pipeline.run({}, _make_envelope(), _make_result())
        assert result.critique_passed is None

    def test_critique_skipped_after_schema_failure(self) -> None:
        pipeline = ValidationPipeline(
            validators=[
                _StubValidator(
                    "schema",
                    [
                        ValidationIssue(
                            layer=ValidationLayer.SCHEMA,
                            severity=ValidationSeverity.ERROR,
                            code="schema.fail",
                            message="fail",
                        )
                    ],
                )
            ],
            critique_validator=CritiqueValidator(
                trigger_config=CritiqueTriggerConfig(
                    always_critique_types=["Signal"]
                ),
                call_llm=lambda **kw: {
                    "overall_score": 0.9,
                    "issues": [],
                    "reasoning": "ok",
                },
            ),
        )
        result = pipeline.run(
            {"_output_type": "Signal"}, _make_envelope(), _make_result()
        )
        assert result.schema_passed is False
        assert result.critique_passed is None


class TestClaimExtraction:
    def test_claims_extracted_when_evidence_validator_present(self) -> None:
        evidence = [_make_evidence("candle_history")]
        envelope = _make_envelope(evidence)
        output = {
            "_output_type": "Signal",
            "rationale": "BTC breaking above resistance with high volume confirmation",
            "evidence_refs": {"rationale": ["candle_history"]},
            "confidence": 0.85,
        }
        pipeline = ValidationPipeline(
            validators=[
                SchemaValidator(),
                EvidenceValidator(),
            ]
        )
        result = pipeline.run(output, envelope, _make_result())
        assert len(result.claims) > 0


class TestOutputHash:
    def test_output_hash_populated(self) -> None:
        pipeline = ValidationPipeline()
        result = pipeline.run(
            {"key": "value"}, _make_envelope(), _make_result()
        )
        assert result.output_hash != ""
        assert len(result.output_hash) == 16


class TestFullPipeline:
    def test_full_pipeline_with_all_layers(self) -> None:
        """Integration test with all real validators."""
        # Setup
        evidence = [_make_evidence("candle_history")]
        envelope = _make_envelope(evidence)

        biz_validator = BusinessRuleValidator()
        biz_validator.register(
            BusinessRuleSet(
                set_id="test",
                name="Test",
                rules=[
                    BusinessRule(
                        rule_id="conf_range",
                        name="Confidence range",
                        rule_type=BusinessRuleType.RANGE,
                        field="confidence",
                        operator=Operator.BETWEEN,
                        threshold=[0.0, 1.0],
                    ),
                ],
            )
        )

        pipeline = ValidationPipeline(
            validators=[
                SchemaValidator(),
                EvidenceValidator(),
                biz_validator,
            ],
            remediation_engine=RemediationEngine(),
        )

        # Valid output
        output = {
            "_output_type": "Signal",
            "rationale": "BTC bullish breakout above key resistance level",
            "evidence_refs": {"rationale": ["candle_history"]},
            "confidence": 0.85,
        }

        llm_result = _make_result()
        result = pipeline.run(output, envelope, llm_result)

        assert result.schema_passed is True
        assert result.business_rules_passed is True
        assert result.quality_score > 0.5
        assert result.output_hash != ""
        assert result.latency_ms >= 0
