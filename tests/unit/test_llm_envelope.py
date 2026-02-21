"""Unit tests for LLM Interaction Envelope models and enums."""

from __future__ import annotations

from datetime import timezone  # noqa: UP017

from agentic_trading.llm.envelope import (
    EnvelopeWorkflow,
    EvidenceItem,
    LLMBudget,
    LLMEnvelope,
    LLMInteraction,
    LLMProvider,
    LLMResult,
    ResponseFormat,
    RetryPolicy,
    SafetyConstraints,
)

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestLLMProvider:
    """LLMProvider enum tests."""

    def test_has_three_members(self):
        assert len(LLMProvider) == 3

    def test_values_are_lowercase(self):
        for member in LLMProvider:
            assert member.value == member.value.lower()

    def test_expected_members(self):
        expected = {"anthropic", "openai", "local"}
        actual = {m.value for m in LLMProvider}
        assert actual == expected


class TestResponseFormat:
    """ResponseFormat enum tests."""

    def test_has_three_members(self):
        assert len(ResponseFormat) == 3

    def test_expected_members(self):
        expected = {"json", "text", "structured"}
        actual = {m.value for m in ResponseFormat}
        assert actual == expected


class TestEnvelopeWorkflow:
    """EnvelopeWorkflow enum tests."""

    def test_has_four_members(self):
        assert len(EnvelopeWorkflow) == 4

    def test_expected_members(self):
        expected = {"analysis", "planning", "execution", "general"}
        actual = {m.value for m in EnvelopeWorkflow}
        assert actual == expected


# ---------------------------------------------------------------------------
# Sub-model tests
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    """RetryPolicy model tests."""

    def test_defaults(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_base_seconds == 1.0
        assert policy.backoff_max_seconds == 30.0
        assert "rate_limit" in policy.retryable_errors
        assert "timeout" in policy.retryable_errors
        assert "server_error" in policy.retryable_errors

    def test_custom_values(self):
        policy = RetryPolicy(
            max_retries=5,
            backoff_base_seconds=2.0,
            backoff_max_seconds=60.0,
            retryable_errors=["rate_limit"],
        )
        assert policy.max_retries == 5
        assert policy.backoff_base_seconds == 2.0
        assert policy.backoff_max_seconds == 60.0
        assert policy.retryable_errors == ["rate_limit"]


class TestLLMBudget:
    """LLMBudget model tests."""

    def test_defaults(self):
        budget = LLMBudget()
        assert budget.max_tokens == 4096
        assert budget.max_input_tokens is None
        assert budget.max_output_tokens == 4096
        assert budget.max_cost_usd is None
        assert budget.thinking_budget_tokens is None

    def test_with_thinking_budget(self):
        budget = LLMBudget(thinking_budget_tokens=8000)
        assert budget.thinking_budget_tokens == 8000

    def test_with_cost_ceiling(self):
        budget = LLMBudget(max_cost_usd=0.50)
        assert budget.max_cost_usd == 0.50


class TestSafetyConstraints:
    """SafetyConstraints model tests."""

    def test_defaults(self):
        safety = SafetyConstraints()
        assert safety.banned_topics == []
        assert safety.max_output_length == 50_000
        assert safety.require_json_output is False
        assert safety.pii_filter is True
        assert safety.require_deterministic is False

    def test_deterministic_flag(self):
        safety = SafetyConstraints(require_deterministic=True)
        assert safety.require_deterministic is True

    def test_custom_banned_topics(self):
        safety = SafetyConstraints(banned_topics=["politics", "religion"])
        assert len(safety.banned_topics) == 2


class TestEvidenceItem:
    """EvidenceItem model tests."""

    def test_creation(self):
        item = EvidenceItem(source="candle_history", content={"1h": {}})
        assert item.source == "candle_history"
        assert item.content == {"1h": {}}
        assert item.relevance == 1.0
        assert item.retrieved_at.tzinfo is not None

    def test_custom_relevance(self):
        item = EvidenceItem(source="fact_table", relevance=0.75)
        assert item.relevance == 0.75


# ---------------------------------------------------------------------------
# LLMEnvelope tests
# ---------------------------------------------------------------------------


class TestLLMEnvelope:
    """LLMEnvelope model tests."""

    def test_creation_with_defaults(self):
        envelope = LLMEnvelope(instructions="Test prompt")
        assert envelope.envelope_id != ""
        assert envelope.trace_id != ""
        assert envelope.tenant_id == "default"
        assert envelope.workflow == EnvelopeWorkflow.GENERAL
        assert envelope.instructions == "Test prompt"
        assert envelope.context == {}
        assert envelope.retrieved_evidence == []
        assert envelope.tools_allowed == []
        assert envelope.response_format == ResponseFormat.JSON
        assert envelope.provider == LLMProvider.ANTHROPIC
        assert envelope.temperature == 0.0

    def test_unique_envelope_ids(self):
        e1 = LLMEnvelope(instructions="a")
        e2 = LLMEnvelope(instructions="b")
        assert e1.envelope_id != e2.envelope_id

    def test_unique_trace_ids(self):
        e1 = LLMEnvelope(instructions="a")
        e2 = LLMEnvelope(instructions="b")
        assert e1.trace_id != e2.trace_id

    def test_envelope_hash_computed(self):
        envelope = LLMEnvelope(instructions="Test prompt")
        assert envelope.envelope_hash != ""
        assert len(envelope.envelope_hash) == 16

    def test_envelope_hash_deterministic(self):
        """Same inputs produce same hash."""
        e1 = LLMEnvelope(
            instructions="Exactly the same prompt",
            context={"k": "v"},
            model="claude-sonnet-4-5-20250929",
            workflow=EnvelopeWorkflow.ANALYSIS,
        )
        e2 = LLMEnvelope(
            instructions="Exactly the same prompt",
            context={"k": "v"},
            model="claude-sonnet-4-5-20250929",
            workflow=EnvelopeWorkflow.ANALYSIS,
        )
        assert e1.envelope_hash == e2.envelope_hash

    def test_envelope_hash_differs_for_different_instructions(self):
        e1 = LLMEnvelope(instructions="Prompt A")
        e2 = LLMEnvelope(instructions="Prompt B")
        assert e1.envelope_hash != e2.envelope_hash

    def test_timestamp_is_utc(self):
        envelope = LLMEnvelope(instructions="Test")
        assert envelope.created_at.tzinfo is not None
        assert envelope.created_at.tzinfo == timezone.utc  # noqa: UP017

    def test_serialization_roundtrip(self):
        envelope = LLMEnvelope(
            instructions="Test prompt",
            context={"symbol": "BTCUSDT"},
            workflow=EnvelopeWorkflow.ANALYSIS,
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-5-20250929",
            agent_id="test-agent",
            agent_type="cmt_analyst",
        )
        json_str = envelope.model_dump_json()
        restored = LLMEnvelope.model_validate_json(json_str)

        assert restored.envelope_id == envelope.envelope_id
        assert restored.instructions == "Test prompt"
        assert restored.context == {"symbol": "BTCUSDT"}
        assert restored.workflow == EnvelopeWorkflow.ANALYSIS
        assert restored.provider == LLMProvider.ANTHROPIC
        assert restored.model == "claude-sonnet-4-5-20250929"
        assert restored.agent_id == "test-agent"
        assert restored.envelope_hash == envelope.envelope_hash

    def test_model_dump_includes_all_fields(self):
        envelope = LLMEnvelope(instructions="Test")
        dumped = envelope.model_dump()
        expected_fields = {
            "envelope_id", "trace_id", "causation_id", "tenant_id",
            "created_at", "workflow", "agent_id", "agent_type",
            "instructions", "context", "retrieved_evidence",
            "tools_allowed", "budget", "expected_output_schema",
            "safety_constraints", "response_format", "provider",
            "model", "temperature", "retry_policy", "envelope_hash",
        }
        assert set(dumped.keys()) == expected_fields

    def test_with_evidence(self):
        evidence = EvidenceItem(
            source="candle_history",
            content={"1h": {"open": 42500}},
        )
        envelope = LLMEnvelope(
            instructions="Analyse",
            retrieved_evidence=[evidence],
        )
        assert len(envelope.retrieved_evidence) == 1
        assert envelope.retrieved_evidence[0].source == "candle_history"


# ---------------------------------------------------------------------------
# LLMResult tests
# ---------------------------------------------------------------------------


class TestLLMResult:
    """LLMResult model tests."""

    def test_creation_with_defaults(self):
        result = LLMResult(envelope_id="env-123")
        assert result.result_id != ""
        assert result.envelope_id == "env-123"
        assert result.raw_output == ""
        assert result.parsed_output == {}
        assert result.validation_passed is False
        assert result.validation_errors == []
        assert result.latency_ms == 0.0
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.thinking_tokens == 0
        assert result.cost_usd == 0.0
        assert result.attempt_number == 1
        assert result.error is None
        assert result.success is True

    def test_output_hash_computed_from_raw_output(self):
        result = LLMResult(
            envelope_id="env-123",
            raw_output='{"score": 0.85}',
        )
        assert result.output_hash != ""
        assert len(result.output_hash) == 16

    def test_output_hash_empty_when_no_raw_output(self):
        result = LLMResult(envelope_id="env-123")
        assert result.output_hash == ""

    def test_output_hash_deterministic(self):
        r1 = LLMResult(
            envelope_id="env-1",
            raw_output="same output",
        )
        r2 = LLMResult(
            envelope_id="env-2",
            raw_output="same output",
        )
        assert r1.output_hash == r2.output_hash

    def test_output_hash_differs_for_different_output(self):
        r1 = LLMResult(envelope_id="env-1", raw_output="output A")
        r2 = LLMResult(envelope_id="env-1", raw_output="output B")
        assert r1.output_hash != r2.output_hash

    def test_attempt_number_tracks_retries(self):
        result = LLMResult(
            envelope_id="env-123",
            attempt_number=3,
            success=False,
            error="rate_limit",
        )
        assert result.attempt_number == 3
        assert result.success is False
        assert result.error == "rate_limit"

    def test_serialization_roundtrip(self):
        result = LLMResult(
            envelope_id="env-123",
            raw_output='{"score": 0.85}',
            parsed_output={"score": 0.85},
            validation_passed=True,
            latency_ms=150.5,
            input_tokens=1200,
            output_tokens=350,
            cost_usd=0.012,
        )
        json_str = result.model_dump_json()
        restored = LLMResult.model_validate_json(json_str)

        assert restored.result_id == result.result_id
        assert restored.envelope_id == "env-123"
        assert restored.parsed_output == {"score": 0.85}
        assert restored.validation_passed is True
        assert restored.output_hash == result.output_hash


# ---------------------------------------------------------------------------
# LLMInteraction tests
# ---------------------------------------------------------------------------


class TestLLMInteraction:
    """LLMInteraction model tests."""

    def test_combines_envelope_and_result(self):
        envelope = LLMEnvelope(instructions="Test")
        result = LLMResult(
            envelope_id=envelope.envelope_id,
            raw_output="response",
        )
        interaction = LLMInteraction(envelope=envelope, result=result)

        assert interaction.interaction_id != ""
        assert interaction.envelope.instructions == "Test"
        assert interaction.result.raw_output == "response"
        assert interaction.result.envelope_id == envelope.envelope_id
        assert interaction.stored_at.tzinfo is not None

    def test_serialization_roundtrip(self):
        envelope = LLMEnvelope(instructions="Test")
        result = LLMResult(
            envelope_id=envelope.envelope_id,
            raw_output='{"ok": true}',
            validation_passed=True,
        )
        interaction = LLMInteraction(envelope=envelope, result=result)

        json_str = interaction.model_dump_json()
        restored = LLMInteraction.model_validate_json(json_str)

        assert restored.interaction_id == interaction.interaction_id
        assert restored.envelope.envelope_id == envelope.envelope_id
        assert restored.result.validation_passed is True
