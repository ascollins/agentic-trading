"""Unit tests for SpineEvent model and taxonomy subclasses."""

from __future__ import annotations

from datetime import timezone

from agentic_trading.telemetry.models import (
    AgentStepCompletedEvent,
    AgentStepStartedEvent,
    CostMetricEvent,
    DecisionEvent,
    EventTaxonomy,
    ExceptionEvent,
    RetrievalEvent,
    SpineEvent,
    ToolCallEvent,
    UserActionEvent,
    ValidationEvent,
)


class TestEventTaxonomy:
    """EventTaxonomy enum tests."""

    def test_has_exactly_nine_members(self):
        assert len(EventTaxonomy) == 9

    def test_values_are_snake_case_strings(self):
        for member in EventTaxonomy:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    def test_all_expected_members_present(self):
        expected = {
            "user_action",
            "agent_step_started",
            "agent_step_completed",
            "tool_call",
            "retrieval",
            "decision",
            "validation",
            "exception",
            "cost_metric",
        }
        actual = {m.value for m in EventTaxonomy}
        assert actual == expected


class TestSpineEvent:
    """SpineEvent base model tests."""

    def test_creation_with_defaults(self):
        event = SpineEvent()
        assert event.event_id != ""
        assert event.trace_id != ""
        assert event.span_id != ""
        assert event.tenant_id == "default"
        assert event.actor == "system"
        assert event.component == ""
        assert event.schema_version == 1
        assert event.event_type == EventTaxonomy.AGENT_STEP_STARTED
        assert event.causation_id == ""
        assert event.payload == {}
        assert event.latency_ms is None
        assert event.error is None

    def test_unique_event_ids(self):
        e1 = SpineEvent()
        e2 = SpineEvent()
        assert e1.event_id != e2.event_id

    def test_unique_span_ids(self):
        e1 = SpineEvent()
        e2 = SpineEvent()
        assert e1.span_id != e2.span_id

    def test_timestamp_is_utc_aware(self):
        event = SpineEvent()
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == timezone.utc  # noqa: UP017

    def test_serialization_roundtrip(self):
        event = SpineEvent(
            actor="test_agent",
            component="test_module",
            payload={"key": "value", "nested": {"a": 1}},
            latency_ms=42.5,
            error="test error",
        )
        json_str = event.model_dump_json()
        restored = SpineEvent.model_validate_json(json_str)

        assert restored.event_id == event.event_id
        assert restored.trace_id == event.trace_id
        assert restored.actor == "test_agent"
        assert restored.component == "test_module"
        assert restored.payload == {"key": "value", "nested": {"a": 1}}
        assert restored.latency_ms == 42.5
        assert restored.error == "test error"

    def test_tenant_id_default(self):
        event = SpineEvent()
        assert event.tenant_id == "default"

    def test_tenant_id_custom(self):
        event = SpineEvent(tenant_id="acme_corp")
        assert event.tenant_id == "acme_corp"

    def test_model_dump_includes_all_fields(self):
        event = SpineEvent()
        dumped = event.model_dump()
        expected_fields = {
            "event_id", "trace_id", "span_id", "tenant_id", "actor",
            "component", "timestamp", "input_hash", "output_hash",
            "schema_version", "event_type", "causation_id", "payload",
            "latency_ms", "error",
        }
        assert set(dumped.keys()) == expected_fields


class TestTaxonomySubclasses:
    """Each subclass pins event_type to its taxonomy value."""

    def test_user_action_event_type(self):
        event = UserActionEvent()
        assert event.event_type == EventTaxonomy.USER_ACTION

    def test_agent_step_started_event_type(self):
        event = AgentStepStartedEvent()
        assert event.event_type == EventTaxonomy.AGENT_STEP_STARTED

    def test_agent_step_completed_event_type(self):
        event = AgentStepCompletedEvent()
        assert event.event_type == EventTaxonomy.AGENT_STEP_COMPLETED

    def test_tool_call_event_type(self):
        event = ToolCallEvent()
        assert event.event_type == EventTaxonomy.TOOL_CALL

    def test_retrieval_event_type(self):
        event = RetrievalEvent()
        assert event.event_type == EventTaxonomy.RETRIEVAL

    def test_decision_event_type(self):
        event = DecisionEvent()
        assert event.event_type == EventTaxonomy.DECISION

    def test_validation_event_type(self):
        event = ValidationEvent()
        assert event.event_type == EventTaxonomy.VALIDATION

    def test_exception_event_type(self):
        event = ExceptionEvent()
        assert event.event_type == EventTaxonomy.EXCEPTION

    def test_cost_metric_event_type(self):
        event = CostMetricEvent()
        assert event.event_type == EventTaxonomy.COST_METRIC

    def test_all_subclasses_inherit_spine_event(self):
        subclasses = [
            UserActionEvent,
            AgentStepStartedEvent,
            AgentStepCompletedEvent,
            ToolCallEvent,
            RetrievalEvent,
            DecisionEvent,
            ValidationEvent,
            ExceptionEvent,
            CostMetricEvent,
        ]
        for cls in subclasses:
            assert issubclass(cls, SpineEvent), f"{cls.__name__} not a SpineEvent subclass"

    def test_subclass_serialization_preserves_event_type(self):
        event = ToolCallEvent(
            component="control_plane.tool_gateway",
            payload={"tool_name": "submit_order"},
        )
        json_str = event.model_dump_json()
        restored = ToolCallEvent.model_validate_json(json_str)
        assert restored.event_type == EventTaxonomy.TOOL_CALL
