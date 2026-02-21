"""Tests for Layer 4: CritiqueValidator."""

from __future__ import annotations

from typing import Any

from agentic_trading.llm.envelope import LLMEnvelope, LLMResult
from agentic_trading.validation.critique_validator import (
    CritiqueTriggerConfig,
    CritiqueValidator,
)
from agentic_trading.validation.models import ValidationLayer, ValidationSeverity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    context: dict[str, Any] | None = None,
) -> LLMEnvelope:
    return LLMEnvelope(
        instructions="test",
        context=context or {},
    )


def _make_result() -> LLMResult:
    return LLMResult(envelope_id="test-envelope")


def _mock_llm_pass(**kwargs: Any) -> dict[str, Any]:
    """Mock LLM that returns a passing critique."""
    return {
        "overall_score": 0.9,
        "issues": [],
        "reasoning": "Output looks correct",
    }


def _mock_llm_fail(**kwargs: Any) -> dict[str, Any]:
    """Mock LLM that returns a failing critique."""
    return {
        "overall_score": 0.3,
        "issues": [
            {
                "code": "critique.logic_error",
                "message": "Contradiction detected",
                "severity": "error",
            }
        ],
        "reasoning": "Found logical inconsistency",
    }


def _mock_llm_raise(**kwargs: Any) -> dict[str, Any]:
    """Mock LLM that raises an exception."""
    msg = "LLM service unavailable"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCritiqueValidatorLayerName:
    def test_layer_name(self) -> None:
        v = CritiqueValidator()
        assert v.layer_name == "critique"


class TestShouldTrigger:
    def test_disabled_does_not_trigger(self) -> None:
        config = CritiqueTriggerConfig(enabled=False)
        v = CritiqueValidator(trigger_config=config)
        should, reason = v.should_trigger(
            {"_output_type": "CMTAssessmentResponse"},
            _make_envelope(),
            1.0,
        )
        assert should is False
        assert reason == "disabled"

    def test_always_critique_type_triggers(self) -> None:
        config = CritiqueTriggerConfig(
            always_critique_types=["CMTAssessmentResponse"]
        )
        v = CritiqueValidator(trigger_config=config)
        should, reason = v.should_trigger(
            {"_output_type": "CMTAssessmentResponse"},
            _make_envelope(),
            1.0,
        )
        assert should is True
        assert "always_critique" in reason

    def test_high_notional_triggers(self) -> None:
        config = CritiqueTriggerConfig(
            notional_usd_threshold=50_000,
            always_critique_types=[],
        )
        v = CritiqueValidator(trigger_config=config)
        should, reason = v.should_trigger(
            {"_output_type": "Signal"},
            _make_envelope({"notional_usd": 100_000}),
            1.0,
        )
        assert should is True
        assert "high_notional" in reason

    def test_low_confidence_triggers(self) -> None:
        config = CritiqueTriggerConfig(
            confidence_floor=0.5,
            always_critique_types=[],
        )
        v = CritiqueValidator(trigger_config=config)
        should, reason = v.should_trigger(
            {"_output_type": "Signal"},
            _make_envelope(),
            0.3,
        )
        assert should is True
        assert "low_confidence" in reason

    def test_no_trigger_when_conditions_not_met(self) -> None:
        config = CritiqueTriggerConfig(
            always_critique_types=[],
            notional_usd_threshold=1_000_000,
            confidence_floor=0.1,
        )
        v = CritiqueValidator(trigger_config=config)
        should, reason = v.should_trigger(
            {"_output_type": "Signal"},
            _make_envelope(),
            0.9,
        )
        assert should is False
        assert reason == "not_triggered"


class TestCritiqueValidation:
    def test_no_llm_callable_returns_empty(self) -> None:
        config = CritiqueTriggerConfig(
            always_critique_types=["Signal"]
        )
        v = CritiqueValidator(trigger_config=config, call_llm=None)
        issues = v.validate(
            {"_output_type": "Signal"},
            _make_envelope(),
            _make_result(),
        )
        assert issues == []

    def test_not_triggered_returns_empty(self) -> None:
        config = CritiqueTriggerConfig(
            always_critique_types=[],
            notional_usd_threshold=1_000_000,
            confidence_floor=0.1,
        )
        v = CritiqueValidator(
            trigger_config=config, call_llm=_mock_llm_pass
        )
        issues = v.validate(
            {"_output_type": "Signal"},
            _make_envelope(),
            _make_result(),
            prior_quality_score=0.9,
        )
        assert issues == []

    def test_passing_critique_no_errors(self) -> None:
        config = CritiqueTriggerConfig(
            always_critique_types=["Signal"]
        )
        v = CritiqueValidator(
            trigger_config=config, call_llm=_mock_llm_pass
        )
        issues = v.validate(
            {"_output_type": "Signal"},
            _make_envelope(),
            _make_result(),
        )
        assert len(issues) == 0

    def test_failing_critique_produces_error(self) -> None:
        config = CritiqueTriggerConfig(
            always_critique_types=["Signal"],
            acceptance_threshold=0.6,
        )
        v = CritiqueValidator(
            trigger_config=config, call_llm=_mock_llm_fail
        )
        issues = v.validate(
            {"_output_type": "Signal"},
            _make_envelope(),
            _make_result(),
        )
        error_issues = [
            i
            for i in issues
            if i.severity == ValidationSeverity.ERROR
        ]
        assert len(error_issues) >= 1
        assert any("low_score" in i.code for i in error_issues)

    def test_failing_critique_includes_critique_issues(self) -> None:
        config = CritiqueTriggerConfig(
            always_critique_types=["Signal"],
            acceptance_threshold=0.6,
        )
        v = CritiqueValidator(
            trigger_config=config, call_llm=_mock_llm_fail
        )
        issues = v.validate(
            {"_output_type": "Signal"},
            _make_envelope(),
            _make_result(),
        )
        critique_issues = [
            i for i in issues if i.code == "critique.logic_error"
        ]
        assert len(critique_issues) == 1

    def test_llm_exception_handled_gracefully(self) -> None:
        config = CritiqueTriggerConfig(
            always_critique_types=["Signal"]
        )
        v = CritiqueValidator(
            trigger_config=config, call_llm=_mock_llm_raise
        )
        issues = v.validate(
            {"_output_type": "Signal"},
            _make_envelope(),
            _make_result(),
        )
        assert len(issues) == 1
        assert issues[0].code == "critique.call_failed"
        assert issues[0].severity == ValidationSeverity.WARNING

    def test_all_issues_tagged_with_critique_layer(self) -> None:
        config = CritiqueTriggerConfig(
            always_critique_types=["Signal"]
        )
        v = CritiqueValidator(
            trigger_config=config, call_llm=_mock_llm_fail
        )
        issues = v.validate(
            {"_output_type": "Signal"},
            _make_envelope(),
            _make_result(),
        )
        for issue in issues:
            assert issue.layer == ValidationLayer.CRITIQUE


class TestCritiqueTriggerConfig:
    def test_defaults(self) -> None:
        config = CritiqueTriggerConfig()
        assert config.notional_usd_threshold == 50_000.0
        assert config.confidence_floor == 0.5
        assert config.acceptance_threshold == 0.6
        assert config.enabled is True
        assert "CMTAssessmentResponse" in config.always_critique_types
