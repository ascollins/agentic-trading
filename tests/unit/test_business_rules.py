"""Tests for Layer 3: BusinessRuleValidator."""

from __future__ import annotations

from agentic_trading.llm.envelope import LLMEnvelope, LLMResult
from agentic_trading.policy.models import Operator
from agentic_trading.validation.business_rules import (
    BusinessRule,
    BusinessRuleSet,
    BusinessRuleType,
    BusinessRuleValidator,
    build_cmt_rules,
    build_signal_rules,
)
from agentic_trading.validation.models import ValidationSeverity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope() -> LLMEnvelope:
    return LLMEnvelope(instructions="test")


def _make_result() -> LLMResult:
    return LLMResult(envelope_id="test-envelope")


def _make_validator(rules: list[BusinessRule]) -> BusinessRuleValidator:
    v = BusinessRuleValidator()
    v.register(
        BusinessRuleSet(
            set_id="test_rules",
            name="Test Rules",
            rules=rules,
        )
    )
    return v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBusinessRuleValidatorLayerName:
    def test_layer_name(self) -> None:
        v = BusinessRuleValidator()
        assert v.layer_name == "business_rule"


class TestRequiredRules:
    def test_required_field_present_passes(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Rationale required",
                    rule_type=BusinessRuleType.REQUIRED,
                    field="rationale",
                    operator=Operator.NE,
                    threshold="",
                )
            ]
        )
        output = {"rationale": "some analysis"}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 0

    def test_required_field_missing_fails(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Rationale required",
                    rule_type=BusinessRuleType.REQUIRED,
                    field="rationale",
                    operator=Operator.NE,
                    threshold="",
                )
            ]
        )
        output = {"confidence": 0.5}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 1
        assert issues[0].code == "business_rule.required.r1"

    def test_required_field_empty_string_fails(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Rationale required",
                    rule_type=BusinessRuleType.REQUIRED,
                    field="rationale",
                    operator=Operator.NE,
                    threshold="",
                )
            ]
        )
        output = {"rationale": ""}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 1

    def test_required_field_empty_list_fails(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Layers required",
                    rule_type=BusinessRuleType.REQUIRED,
                    field="layers",
                    operator=Operator.NE,
                    threshold=[],
                )
            ]
        )
        output = {"layers": []}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 1


class TestRangeRules:
    def test_in_range_passes(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Confidence range",
                    rule_type=BusinessRuleType.RANGE,
                    field="confidence",
                    operator=Operator.BETWEEN,
                    threshold=[0.0, 1.0],
                )
            ]
        )
        output = {"confidence": 0.7}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 0

    def test_out_of_range_fails(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Confidence range",
                    rule_type=BusinessRuleType.RANGE,
                    field="confidence",
                    operator=Operator.BETWEEN,
                    threshold=[0.0, 1.0],
                )
            ]
        )
        output = {"confidence": 1.5}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR

    def test_in_set_passes(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Direction valid",
                    rule_type=BusinessRuleType.RANGE,
                    field="direction",
                    operator=Operator.IN,
                    threshold=["LONG", "SHORT", "FLAT"],
                )
            ]
        )
        output = {"direction": "LONG"}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 0

    def test_not_in_set_fails(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Direction valid",
                    rule_type=BusinessRuleType.RANGE,
                    field="direction",
                    operator=Operator.IN,
                    threshold=["LONG", "SHORT", "FLAT"],
                )
            ]
        )
        output = {"direction": "INVALID"}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 1


class TestInvariantRules:
    def test_invariant_passes(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Stop below entry",
                    rule_type=BusinessRuleType.INVARIANT,
                    field="stop_loss",
                    operator=Operator.LT,
                    threshold=0,
                    compare_field="entry_price",
                )
            ]
        )
        output = {"stop_loss": 41000, "entry_price": 43000}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 0

    def test_invariant_fails(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Stop below entry",
                    rule_type=BusinessRuleType.INVARIANT,
                    field="stop_loss",
                    operator=Operator.LT,
                    threshold=0,
                    compare_field="entry_price",
                )
            ]
        )
        output = {"stop_loss": 45000, "entry_price": 43000}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 1

    def test_invariant_skipped_when_compare_field_missing(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Stop below entry",
                    rule_type=BusinessRuleType.INVARIANT,
                    field="stop_loss",
                    operator=Operator.LT,
                    threshold=0,
                    compare_field="entry_price",
                )
            ]
        )
        output = {"stop_loss": 41000}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 0


class TestOutputTypeScoping:
    def test_scoped_rule_applies_when_matched(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Signal confidence",
                    rule_type=BusinessRuleType.RANGE,
                    field="confidence",
                    operator=Operator.BETWEEN,
                    threshold=[0.0, 1.0],
                    output_types=["Signal"],
                )
            ]
        )
        output = {"_output_type": "Signal", "confidence": 1.5}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 1

    def test_scoped_rule_skipped_when_not_matched(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Signal confidence",
                    rule_type=BusinessRuleType.RANGE,
                    field="confidence",
                    operator=Operator.BETWEEN,
                    threshold=[0.0, 1.0],
                    output_types=["Signal"],
                )
            ]
        )
        output = {
            "_output_type": "CMTAssessmentResponse",
            "confidence": 1.5,
        }
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 0

    def test_unscoped_rule_applies_to_all(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Confidence range",
                    rule_type=BusinessRuleType.RANGE,
                    field="confidence",
                    operator=Operator.BETWEEN,
                    threshold=[0.0, 1.0],
                    output_types=None,
                )
            ]
        )
        output = {
            "_output_type": "AnyType",
            "confidence": 1.5,
        }
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 1


class TestDisabledRules:
    def test_disabled_rule_skipped(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Confidence range",
                    rule_type=BusinessRuleType.RANGE,
                    field="confidence",
                    operator=Operator.BETWEEN,
                    threshold=[0.0, 1.0],
                    enabled=False,
                )
            ]
        )
        output = {"confidence": 1.5}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 0


class TestDotPathResolution:
    def test_nested_field_resolved(self) -> None:
        v = _make_validator(
            [
                BusinessRule(
                    rule_id="r1",
                    name="Nested score",
                    rule_type=BusinessRuleType.RANGE,
                    field="confluence.total",
                    operator=Operator.GE,
                    threshold=0,
                )
            ]
        )
        output = {"confluence": {"total": 5}}
        issues = v.validate(output, _make_envelope(), _make_result())
        assert len(issues) == 0


class TestBusinessRuleSet:
    def test_active_rules_filters_disabled(self) -> None:
        rule_set = BusinessRuleSet(
            set_id="test",
            name="Test",
            rules=[
                BusinessRule(
                    rule_id="r1",
                    name="Active",
                    rule_type=BusinessRuleType.REQUIRED,
                    field="x",
                    operator=Operator.NE,
                    threshold="",
                    enabled=True,
                ),
                BusinessRule(
                    rule_id="r2",
                    name="Disabled",
                    rule_type=BusinessRuleType.REQUIRED,
                    field="y",
                    operator=Operator.NE,
                    threshold="",
                    enabled=False,
                ),
            ],
        )
        assert len(rule_set.active_rules) == 1


class TestBuiltInRuleFactories:
    def test_signal_rules_exist(self) -> None:
        rules = build_signal_rules()
        assert rules.set_id == "signal_validation_v1"
        assert len(rules.rules) >= 2

    def test_cmt_rules_exist(self) -> None:
        rules = build_cmt_rules()
        assert rules.set_id == "cmt_validation_v1"
        assert len(rules.rules) >= 2

    def test_signal_rules_scoped_to_signal(self) -> None:
        rules = build_signal_rules()
        for rule in rules.rules:
            assert rule.output_types is not None
            assert "Signal" in rule.output_types

    def test_cmt_rules_scoped_to_cmt(self) -> None:
        rules = build_cmt_rules()
        for rule in rules.rules:
            assert rule.output_types is not None
            assert "CMTAssessmentResponse" in rule.output_types
