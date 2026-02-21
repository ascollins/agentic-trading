"""Layer 3: Declarative business rule validation.

Mirrors the PolicyEngine pattern: rules are data (field, operator,
threshold), not imperative code.  This makes rules auditable,
versionable, and configurable per output type.

Built-in rule categories:

- RANGE: field must be within [min, max]
- REQUIRED: field must be present and non-empty
- INVARIANT: cross-field consistency (e.g. stop_loss < entry for long)
- CROSS_CHECK: field must match or correlate with another field

Reuses ``Operator`` enum and ``PolicyEngine._resolve_field`` /
``_check_condition`` from ``policy/`` to maintain consistent
operator semantics.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.llm.envelope import LLMEnvelope, LLMResult
from agentic_trading.policy.engine import PolicyEngine
from agentic_trading.policy.models import Operator

from .models import ValidationIssue, ValidationLayer, ValidationSeverity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and models
# ---------------------------------------------------------------------------


class BusinessRuleType(str, Enum):
    """Categories of business validation rules."""

    RANGE = "range"
    REQUIRED = "required"
    INVARIANT = "invariant"
    CROSS_CHECK = "cross_check"


class BusinessRule(BaseModel):
    """A single declarative business rule for agent output validation.

    Follows the same pattern as ``policy.models.PolicyRule``:
    field + operator + threshold, with scoping by output_type.
    """

    rule_id: str
    name: str
    description: str = ""
    rule_type: BusinessRuleType
    field: str
    operator: Operator
    threshold: Any
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True
    output_types: list[str] | None = None
    compare_field: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BusinessRuleSet(BaseModel):
    """Named, versioned collection of business rules."""

    set_id: str
    name: str
    version: int = 1
    rules: list[BusinessRule] = Field(default_factory=list)

    @property
    def active_rules(self) -> list[BusinessRule]:
        """Return only enabled rules."""
        return [r for r in self.rules if r.enabled]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class BusinessRuleValidator:
    """Evaluates declarative business rules against parsed LLM output.

    Uses the same operator evaluation logic as PolicyEngine.
    """

    def __init__(self) -> None:
        self._rule_sets: dict[str, BusinessRuleSet] = {}

    @property
    def layer_name(self) -> str:
        return "business_rule"

    def register(self, rule_set: BusinessRuleSet) -> None:
        """Register (or replace) a rule set."""
        self._rule_sets[rule_set.set_id] = rule_set

    def validate(
        self,
        parsed_output: dict[str, Any],
        envelope: LLMEnvelope,
        result: LLMResult,
    ) -> list[ValidationIssue]:
        """Evaluate all registered rules against the output."""
        issues: list[ValidationIssue] = []
        output_type = parsed_output.get("_output_type", "")

        for rule_set in self._rule_sets.values():
            for rule in rule_set.active_rules:
                # Check scope
                if (
                    rule.output_types is not None
                    and output_type not in rule.output_types
                ):
                    continue

                issue = self._evaluate_rule(rule, parsed_output)
                if issue is not None:
                    issues.append(issue)

        return issues

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_rule(
        rule: BusinessRule,
        data: dict[str, Any],
    ) -> ValidationIssue | None:
        """Evaluate a single business rule.  Returns issue if violated."""
        value = PolicyEngine._resolve_field(rule.field, data)

        # REQUIRED: check presence
        if rule.rule_type == BusinessRuleType.REQUIRED:
            if value is None or value == "" or value == []:
                return ValidationIssue(
                    layer=ValidationLayer.BUSINESS_RULE,
                    severity=rule.severity,
                    code=f"business_rule.required.{rule.rule_id}",
                    message=(
                        f"{rule.name}: required field "
                        f"'{rule.field}' is missing or empty"
                    ),
                    field_path=rule.field,
                )
            return None

        # Non-required field missing — skip (not a violation)
        if value is None:
            return None

        # INVARIANT: compare two fields
        if rule.rule_type == BusinessRuleType.INVARIANT and rule.compare_field:
            compare_value = PolicyEngine._resolve_field(
                rule.compare_field, data
            )
            if compare_value is None:
                return None
            passed = PolicyEngine._check_condition(
                value, rule.operator, compare_value
            )
        else:
            passed = PolicyEngine._check_condition(
                value, rule.operator, rule.threshold
            )

        if not passed:
            return ValidationIssue(
                layer=ValidationLayer.BUSINESS_RULE,
                severity=rule.severity,
                code=(
                    f"business_rule.{rule.rule_type.value}.{rule.rule_id}"
                ),
                message=(
                    f"{rule.name}: {rule.field}={value} "
                    f"violates {rule.operator.value} {rule.threshold}"
                ),
                field_path=rule.field,
                expected=rule.threshold,
                actual=value,
                metadata=rule.metadata,
            )

        return None


# ---------------------------------------------------------------------------
# Built-in rule factories
# ---------------------------------------------------------------------------


def build_signal_rules() -> BusinessRuleSet:
    """Default business rules for Signal outputs."""
    return BusinessRuleSet(
        set_id="signal_validation_v1",
        name="Signal Output Validation",
        rules=[
            BusinessRule(
                rule_id="confidence_range",
                name="Confidence in valid range",
                rule_type=BusinessRuleType.RANGE,
                field="confidence",
                operator=Operator.BETWEEN,
                threshold=[0.0, 1.0],
                output_types=["Signal"],
            ),
            BusinessRule(
                rule_id="direction_valid",
                name="Direction must be valid",
                rule_type=BusinessRuleType.RANGE,
                field="direction",
                operator=Operator.IN,
                threshold=["LONG", "SHORT", "FLAT"],
                output_types=["Signal"],
            ),
            BusinessRule(
                rule_id="rationale_required",
                name="Rationale must be present",
                rule_type=BusinessRuleType.REQUIRED,
                field="rationale",
                operator=Operator.NE,
                threshold="",
                output_types=["Signal"],
            ),
        ],
    )


def build_cmt_rules() -> BusinessRuleSet:
    """Default business rules for CMT assessment outputs."""
    return BusinessRuleSet(
        set_id="cmt_validation_v1",
        name="CMT Assessment Validation",
        rules=[
            BusinessRule(
                rule_id="thesis_required",
                name="Thesis must be present",
                rule_type=BusinessRuleType.REQUIRED,
                field="thesis",
                operator=Operator.NE,
                threshold="",
                output_types=["CMTAssessmentResponse"],
            ),
            BusinessRule(
                rule_id="system_health_valid",
                name="System health must be valid",
                rule_type=BusinessRuleType.RANGE,
                field="system_health",
                operator=Operator.IN,
                threshold=["green", "amber", "red"],
                output_types=["CMTAssessmentResponse"],
            ),
            BusinessRule(
                rule_id="layers_non_empty",
                name="At least one analysis layer required",
                rule_type=BusinessRuleType.REQUIRED,
                field="layers",
                operator=Operator.NE,
                threshold=[],
                output_types=["CMTAssessmentResponse"],
            ),
        ],
    )
