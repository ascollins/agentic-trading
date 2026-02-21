"""Layer 1: Structural schema validation.

Hard fail — if the output doesn't match the expected schema,
subsequent validation layers cannot run meaningfully.

Supports two validation modes:

1. **JSON Schema** (from ``envelope.expected_output_schema``):
   Uses ``jsonschema`` library (Draft 2020-12).

2. **Pydantic model** (registered per output type):
   Attempts ``model_class.model_validate(data)``.

Both produce ``ValidationIssue`` with ``severity=ERROR``.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.llm.envelope import LLMEnvelope, LLMResult

from .models import ValidationIssue, ValidationLayer, ValidationSeverity

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates LLM output against the expected schema.

    Parameters
    ----------
    pydantic_models:
        Registry mapping output_type strings to Pydantic model classes.
        E.g. ``{"CMTAssessmentResponse": CMTAssessmentResponse}``.
    """

    def __init__(
        self,
        *,
        pydantic_models: dict[str, type] | None = None,
    ) -> None:
        self._pydantic_models: dict[str, type] = pydantic_models or {}

    @property
    def layer_name(self) -> str:
        return "schema"

    def register_model(self, output_type: str, model_class: type) -> None:
        """Register a Pydantic model for a given output type."""
        self._pydantic_models[output_type] = model_class

    def validate(
        self,
        parsed_output: dict[str, Any],
        envelope: LLMEnvelope,
        result: LLMResult,
    ) -> list[ValidationIssue]:
        """Run schema validation.  Returns issues (empty = clean)."""
        issues: list[ValidationIssue] = []

        # 1. JSON Schema validation
        if envelope.expected_output_schema:
            issues.extend(
                self._validate_json_schema(
                    parsed_output, envelope.expected_output_schema
                )
            )

        # 2. Pydantic model validation
        output_type = parsed_output.get("_output_type", "")
        if output_type in self._pydantic_models:
            issues.extend(
                self._validate_pydantic(
                    parsed_output, self._pydantic_models[output_type]
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_json_schema(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
    ) -> list[ValidationIssue]:
        """Validate data against a JSON Schema."""
        issues: list[ValidationIssue] = []
        try:
            import jsonschema  # noqa: I001

            validator_cls = jsonschema.Draft202012Validator
            validator = validator_cls(schema)
            for error in validator.iter_errors(data):
                issues.append(
                    ValidationIssue(
                        layer=ValidationLayer.SCHEMA,
                        severity=ValidationSeverity.ERROR,
                        code=f"schema.{error.validator}",
                        message=error.message,
                        field_path=".".join(
                            str(p) for p in error.absolute_path
                        ),
                        expected=error.schema.get("type", error.schema),
                        actual=error.instance
                        if not isinstance(error.instance, dict)
                        else "<object>",
                    )
                )
        except ImportError:
            logger.warning(
                "jsonschema not installed; skipping JSON Schema validation"
            )
        except Exception:
            logger.exception("JSON Schema validation failed unexpectedly")
            issues.append(
                ValidationIssue(
                    layer=ValidationLayer.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    code="schema.internal_error",
                    message="JSON Schema validation raised an exception",
                )
            )
        return issues

    def _validate_pydantic(
        self,
        data: dict[str, Any],
        model_class: type,
    ) -> list[ValidationIssue]:
        """Validate data by attempting Pydantic model construction."""
        issues: list[ValidationIssue] = []
        try:
            model_class.model_validate(data)
        except Exception as exc:
            if hasattr(exc, "errors"):
                for err in exc.errors():  # type: ignore[union-attr]
                    field = ".".join(
                        str(loc) for loc in err.get("loc", [])
                    )
                    issues.append(
                        ValidationIssue(
                            layer=ValidationLayer.SCHEMA,
                            severity=ValidationSeverity.ERROR,
                            code=f"schema.pydantic.{err.get('type', 'unknown')}",
                            message=err.get("msg", str(exc)),
                            field_path=field,
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        layer=ValidationLayer.SCHEMA,
                        severity=ValidationSeverity.ERROR,
                        code="schema.pydantic.unexpected",
                        message=str(exc),
                    )
                )
        return issues
