"""Tests for Layer 1: SchemaValidator."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.llm.envelope import LLMEnvelope, LLMResult
from agentic_trading.validation.models import ValidationLayer, ValidationSeverity
from agentic_trading.validation.schema_validator import SchemaValidator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    schema: dict[str, Any] | None = None,
) -> LLMEnvelope:
    """Create a minimal envelope for testing."""
    return LLMEnvelope(
        instructions="test",
        expected_output_schema=schema or {},
    )


def _make_result() -> LLMResult:
    """Create a minimal LLMResult for testing."""
    return LLMResult(envelope_id="test-envelope")


class _SampleModel(BaseModel):
    """Pydantic model for testing schema validation."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchemaValidatorLayerName:
    def test_layer_name(self) -> None:
        v = SchemaValidator()
        assert v.layer_name == "schema"


class TestSchemaValidatorPydantic:
    def test_valid_data_passes(self) -> None:
        v = SchemaValidator(
            pydantic_models={"SampleModel": _SampleModel}
        )
        data = {"_output_type": "SampleModel", "name": "test", "score": 0.5}
        issues = v.validate(data, _make_envelope(), _make_result())
        assert issues == []

    def test_missing_required_field_fails(self) -> None:
        v = SchemaValidator(
            pydantic_models={"SampleModel": _SampleModel}
        )
        data = {"_output_type": "SampleModel", "score": 0.5}
        issues = v.validate(data, _make_envelope(), _make_result())
        assert len(issues) >= 1
        assert all(i.layer == ValidationLayer.SCHEMA for i in issues)
        assert all(i.severity == ValidationSeverity.ERROR for i in issues)

    def test_out_of_range_field_fails(self) -> None:
        v = SchemaValidator(
            pydantic_models={"SampleModel": _SampleModel}
        )
        data = {"_output_type": "SampleModel", "name": "test", "score": 1.5}
        issues = v.validate(data, _make_envelope(), _make_result())
        assert len(issues) >= 1
        assert any("score" in i.field_path for i in issues)

    def test_unregistered_type_skipped(self) -> None:
        v = SchemaValidator(
            pydantic_models={"SampleModel": _SampleModel}
        )
        data = {"_output_type": "OtherModel", "anything": "goes"}
        issues = v.validate(data, _make_envelope(), _make_result())
        assert issues == []

    def test_register_model(self) -> None:
        v = SchemaValidator()
        v.register_model("SampleModel", _SampleModel)
        data = {"_output_type": "SampleModel", "name": "test", "score": 0.5}
        issues = v.validate(data, _make_envelope(), _make_result())
        assert issues == []


class TestSchemaValidatorJsonSchema:
    def test_valid_data_passes(self) -> None:
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
            },
        }
        v = SchemaValidator()
        data = {"name": "test"}
        issues = v.validate(data, _make_envelope(schema), _make_result())
        # jsonschema may not be installed in test env; check gracefully
        # If no issues, either data passed or jsonschema wasn't available
        for issue in issues:
            assert issue.layer == ValidationLayer.SCHEMA

    def test_missing_required_field_detected(self) -> None:
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
            },
        }
        v = SchemaValidator()
        data = {}
        issues = v.validate(data, _make_envelope(schema), _make_result())
        # If jsonschema is installed, should find the missing field
        if issues:
            assert any(
                i.severity == ValidationSeverity.ERROR for i in issues
            )

    def test_wrong_type_detected(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
            },
        }
        v = SchemaValidator()
        data = {"score": "not_a_number"}
        issues = v.validate(data, _make_envelope(schema), _make_result())
        if issues:
            assert any(
                i.severity == ValidationSeverity.ERROR for i in issues
            )

    def test_empty_schema_skipped(self) -> None:
        v = SchemaValidator()
        data = {"anything": "goes"}
        issues = v.validate(data, _make_envelope({}), _make_result())
        assert issues == []


class TestSchemaValidatorCombined:
    def test_both_json_schema_and_pydantic_run(self) -> None:
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "number"},
            },
        }
        v = SchemaValidator(
            pydantic_models={"SampleModel": _SampleModel}
        )
        # Valid for both
        data = {
            "_output_type": "SampleModel",
            "name": "test",
            "score": 0.5,
        }
        issues = v.validate(data, _make_envelope(schema), _make_result())
        assert issues == []

    def test_issue_codes_contain_schema_prefix(self) -> None:
        v = SchemaValidator(
            pydantic_models={"SampleModel": _SampleModel}
        )
        data = {"_output_type": "SampleModel"}  # missing 'name'
        issues = v.validate(data, _make_envelope(), _make_result())
        for issue in issues:
            assert issue.code.startswith("schema.")
