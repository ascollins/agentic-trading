"""Unit tests for EnvelopeBuilder."""

from __future__ import annotations

import pytest

from agentic_trading.llm.envelope import (
    EnvelopeWorkflow,
    LLMEnvelope,
    LLMProvider,
    ResponseFormat,
)
from agentic_trading.llm.envelope_builder import EnvelopeBuilder
from agentic_trading.llm.errors import EnvelopeValidationError

# ---------------------------------------------------------------------------
# Workflow presets
# ---------------------------------------------------------------------------


class TestBuilderPresets:
    """Workflow preset tests."""

    def test_for_analysis_defaults(self):
        envelope = (
            EnvelopeBuilder()
            .for_analysis()
            .with_instructions("Analyse market data")
            .build()
        )
        assert envelope.workflow == EnvelopeWorkflow.ANALYSIS
        assert envelope.temperature == 0.0
        assert envelope.budget.max_output_tokens == 4096
        assert envelope.budget.thinking_budget_tokens == 8000
        assert envelope.safety_constraints.require_json_output is True
        assert envelope.safety_constraints.require_deterministic is False
        assert envelope.response_format == ResponseFormat.JSON

    def test_for_planning_defaults(self):
        envelope = (
            EnvelopeBuilder()
            .for_planning()
            .with_instructions("Recommend parameters")
            .build()
        )
        assert envelope.workflow == EnvelopeWorkflow.PLANNING
        assert envelope.temperature == 0.0
        assert envelope.budget.max_output_tokens == 8192
        assert envelope.budget.thinking_budget_tokens is None
        assert envelope.safety_constraints.require_json_output is True
        assert envelope.safety_constraints.require_deterministic is False

    def test_for_execution_defaults(self):
        envelope = (
            EnvelopeBuilder()
            .for_execution()
            .with_instructions("Select fill strategy")
            .build()
        )
        assert envelope.workflow == EnvelopeWorkflow.EXECUTION
        assert envelope.temperature == 0.0
        assert envelope.budget.max_output_tokens == 2048
        assert envelope.safety_constraints.require_json_output is True
        assert envelope.safety_constraints.require_deterministic is True

    def test_execution_rejects_nonzero_temperature(self):
        builder = (
            EnvelopeBuilder()
            .for_execution()
            .with_instructions("Select fill strategy")
            .with_temperature(0.5)
        )
        with pytest.raises(
            EnvelopeValidationError,
            match="temperature must be 0.0",
        ):
            builder.build()


# ---------------------------------------------------------------------------
# Fluent API
# ---------------------------------------------------------------------------


class TestBuilderFluent:
    """Fluent chaining tests."""

    def test_all_setters_return_self(self):
        builder = EnvelopeBuilder()
        assert builder.for_analysis() is builder
        assert builder.with_instructions("test") is builder
        assert builder.with_context({"k": "v"}) is builder
        assert builder.add_evidence("src", {}) is builder
        assert builder.allow_tools(["tool1"]) is builder
        assert builder.with_budget(max_output_tokens=4096) is builder
        assert builder.with_output_schema({"type": "object"}) is builder
        assert builder.with_safety(require_json=True) is builder
        assert builder.with_retry(max_retries=5) is builder
        assert builder.with_provider(LLMProvider.OPENAI, "gpt-4") is builder
        assert builder.with_temperature(0.0) is builder
        assert builder.with_trace("trace-1", "cause-1") is builder
        assert builder.with_agent("agent-1", "analyst") is builder
        assert builder.deterministic() is builder

    def test_chained_build_produces_valid_envelope(self):
        envelope = (
            EnvelopeBuilder()
            .for_analysis()
            .with_instructions("Full chain test")
            .with_context({"symbol": "BTCUSDT"})
            .add_evidence("candles", {"1h": {}})
            .add_evidence("indicators", {"rsi": 55.0})
            .allow_tools(["calculate"])
            .with_output_schema({"type": "object"})
            .with_provider(LLMProvider.ANTHROPIC, "claude-sonnet-4-5-20250929")
            .with_trace("trace-abc", "cause-xyz")
            .with_agent("analyst-01", "cmt_analyst")
            .build()
        )
        assert isinstance(envelope, LLMEnvelope)
        assert envelope.instructions == "Full chain test"
        assert envelope.context == {"symbol": "BTCUSDT"}
        assert len(envelope.retrieved_evidence) == 2
        assert envelope.tools_allowed == ["calculate"]
        assert envelope.model == "claude-sonnet-4-5-20250929"
        assert envelope.trace_id == "trace-abc"
        assert envelope.causation_id == "cause-xyz"
        assert envelope.agent_id == "analyst-01"
        assert envelope.agent_type == "cmt_analyst"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestBuilderValidation:
    """Builder validation tests."""

    def test_missing_instructions_raises(self):
        with pytest.raises(
            EnvelopeValidationError,
            match="instructions must not be empty",
        ):
            EnvelopeBuilder().build()

    def test_empty_instructions_raises(self):
        with pytest.raises(
            EnvelopeValidationError,
            match="instructions must not be empty",
        ):
            EnvelopeBuilder().with_instructions("").build()

    def test_deterministic_with_nonzero_temperature_raises(self):
        with pytest.raises(
            EnvelopeValidationError,
            match="temperature must be 0.0",
        ):
            (
                EnvelopeBuilder()
                .with_instructions("Test")
                .deterministic()
                .with_temperature(0.7)
                .build()
            )

    def test_deterministic_sugar_sets_both_flags(self):
        envelope = (
            EnvelopeBuilder()
            .with_instructions("Test")
            .deterministic()
            .build()
        )
        assert envelope.temperature == 0.0
        assert envelope.safety_constraints.require_deterministic is True


# ---------------------------------------------------------------------------
# Evidence accumulation
# ---------------------------------------------------------------------------


class TestBuilderEvidence:
    """Evidence accumulation tests."""

    def test_add_evidence_accumulates(self):
        envelope = (
            EnvelopeBuilder()
            .with_instructions("Test")
            .add_evidence("source_a", {"a": 1})
            .add_evidence("source_b", {"b": 2})
            .add_evidence("source_c", {"c": 3}, relevance=0.5)
            .build()
        )
        assert len(envelope.retrieved_evidence) == 3
        assert envelope.retrieved_evidence[0].source == "source_a"
        assert envelope.retrieved_evidence[1].source == "source_b"
        assert envelope.retrieved_evidence[2].relevance == 0.5


# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------


class TestBuilderBuild:
    """Build output tests."""

    def test_builds_valid_llm_envelope(self):
        envelope = (
            EnvelopeBuilder()
            .with_instructions("Test prompt")
            .build()
        )
        assert isinstance(envelope, LLMEnvelope)
        assert envelope.envelope_id != ""
        assert envelope.envelope_hash != ""

    def test_hash_is_stable_for_same_inputs(self):
        def _build():
            return (
                EnvelopeBuilder()
                .for_analysis()
                .with_instructions("Same prompt")
                .with_context({"k": "v"})
                .with_provider(LLMProvider.ANTHROPIC, "claude-sonnet-4-5-20250929")
                .build()
            )

        e1 = _build()
        e2 = _build()
        assert e1.envelope_hash == e2.envelope_hash

    def test_trace_id_not_set_when_omitted(self):
        """When no trace is set, a random trace_id is generated."""
        envelope = (
            EnvelopeBuilder()
            .with_instructions("Test")
            .build()
        )
        assert envelope.trace_id != ""

    def test_trace_id_preserved_when_set(self):
        envelope = (
            EnvelopeBuilder()
            .with_instructions("Test")
            .with_trace("my-trace-id")
            .build()
        )
        assert envelope.trace_id == "my-trace-id"

    def test_provider_defaults_to_anthropic(self):
        envelope = (
            EnvelopeBuilder()
            .with_instructions("Test")
            .build()
        )
        assert envelope.provider == LLMProvider.ANTHROPIC

    def test_custom_budget(self):
        envelope = (
            EnvelopeBuilder()
            .with_instructions("Test")
            .with_budget(
                max_output_tokens=16000,
                thinking_budget=10000,
                max_cost_usd=1.0,
            )
            .build()
        )
        assert envelope.budget.max_output_tokens == 16000
        assert envelope.budget.thinking_budget_tokens == 10000
        assert envelope.budget.max_cost_usd == 1.0

    def test_custom_safety(self):
        envelope = (
            EnvelopeBuilder()
            .with_instructions("Test")
            .with_safety(
                require_json=True,
                pii_filter=False,
                deterministic=True,
            )
            .build()
        )
        assert envelope.safety_constraints.require_json_output is True
        assert envelope.safety_constraints.pii_filter is False
        assert envelope.safety_constraints.require_deterministic is True

    def test_custom_retry(self):
        envelope = (
            EnvelopeBuilder()
            .with_instructions("Test")
            .with_retry(max_retries=5, backoff_base=2.0)
            .build()
        )
        assert envelope.retry_policy.max_retries == 5
        assert envelope.retry_policy.backoff_base_seconds == 2.0
