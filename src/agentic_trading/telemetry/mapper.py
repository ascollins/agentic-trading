"""Map existing BaseEvent subclasses to SpineEvent taxonomy types.

This mapper bridges the existing event bus (40+ BaseEvent types) to the
parallel spine telemetry layer without coupling the two.

Usage::

    from agentic_trading.telemetry.mapper import map_base_event_to_spine

    spine_event = map_base_event_to_spine(base_event, topic="execution")
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.core.events import BaseEvent
from agentic_trading.core.ids import payload_hash as _payload_hash
from agentic_trading.telemetry.models import (
    AgentStepCompletedEvent,
    AgentStepStartedEvent,
    DecisionEvent,
    EventTaxonomy,
    ExceptionEvent,
    RetrievalEvent,
    SpineEvent,
    ToolCallEvent,
    ValidationEvent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping table: BaseEvent class name -> (EventTaxonomy, SpineEvent subclass)
# ---------------------------------------------------------------------------

_CLASS_MAP: dict[str, tuple[EventTaxonomy, type[SpineEvent]]] = {
    # Tool calls
    "ToolCallRecorded": (EventTaxonomy.TOOL_CALL, ToolCallEvent),
    # Decisions
    "RiskCheckResult": (EventTaxonomy.DECISION, DecisionEvent),
    "GovernanceDecision": (EventTaxonomy.DECISION, DecisionEvent),
    # Agent step completed (produced an output)
    "Signal": (EventTaxonomy.AGENT_STEP_COMPLETED, AgentStepCompletedEvent),
    "OrderIntent": (EventTaxonomy.AGENT_STEP_COMPLETED, AgentStepCompletedEvent),
    "FillEvent": (EventTaxonomy.AGENT_STEP_COMPLETED, AgentStepCompletedEvent),
    "FeatureVector": (EventTaxonomy.AGENT_STEP_COMPLETED, AgentStepCompletedEvent),
    "CMTAssessment": (EventTaxonomy.AGENT_STEP_COMPLETED, AgentStepCompletedEvent),
    "OrderAck": (EventTaxonomy.AGENT_STEP_COMPLETED, AgentStepCompletedEvent),
    "TargetPosition": (EventTaxonomy.AGENT_STEP_COMPLETED, AgentStepCompletedEvent),
    # Retrieval
    "CandleEvent": (EventTaxonomy.RETRIEVAL, RetrievalEvent),
    "TickEvent": (EventTaxonomy.RETRIEVAL, RetrievalEvent),
    "OrderBookSnapshot": (EventTaxonomy.RETRIEVAL, RetrievalEvent),
    # Exceptions / circuit breakers
    "CircuitBreakerEvent": (EventTaxonomy.EXCEPTION, ExceptionEvent),
    "KillSwitchEvent": (EventTaxonomy.EXCEPTION, ExceptionEvent),
    "IncidentDeclared": (EventTaxonomy.EXCEPTION, ExceptionEvent),
    "IncidentCreated": (EventTaxonomy.EXCEPTION, ExceptionEvent),
    # Validation
    "ReconciliationResult": (EventTaxonomy.VALIDATION, ValidationEvent),
}

# Default for unmapped event types
_DEFAULT_TAXONOMY = EventTaxonomy.AGENT_STEP_STARTED
_DEFAULT_CLASS = AgentStepStartedEvent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_base_event_to_spine(
    event: BaseEvent,
    *,
    topic: str = "",
    tenant_id: str = "default",
) -> SpineEvent:
    """Convert a ``BaseEvent`` to the appropriate ``SpineEvent`` subclass.

    Parameters
    ----------
    event:
        Any BaseEvent subclass instance.
    topic:
        Optional event bus topic (included in payload for context).
    tenant_id:
        Tenant isolation key.

    Returns
    -------
    SpineEvent:
        The taxonomy-classified spine event with fields copied from *event*.
    """
    class_name = type(event).__name__
    taxonomy, spine_cls = _CLASS_MAP.get(
        class_name, (_DEFAULT_TAXONOMY, _DEFAULT_CLASS),
    )

    # Build payload from the event's full dump
    event_data = event.model_dump(mode="json")
    payload: dict[str, Any] = {
        "source_event_type": class_name,
        "topic": topic,
    }
    # Include all event-specific fields (exclude BaseEvent fields)
    _BASE_FIELDS = {
        "event_id", "timestamp", "trace_id", "source_module",
        "schema_version", "causation_id",
    }
    for key, value in event_data.items():
        if key not in _BASE_FIELDS:
            payload[key] = value

    # Extract hashes from events that carry them
    input_hash = ""
    output_hash = ""
    if hasattr(event, "request_hash") and event.request_hash:
        input_hash = event.request_hash
    if hasattr(event, "response_hash") and event.response_hash:
        output_hash = event.response_hash

    # Compute fallback hashes if not available from the event
    if not input_hash:
        input_hash = _payload_hash(payload)
    if not output_hash:
        output_hash = _payload_hash(payload)

    # Extract latency if available
    latency_ms: float | None = None
    if hasattr(event, "latency_ms") and event.latency_ms is not None:
        latency_ms = float(event.latency_ms)

    # Extract error if available
    error_msg: str | None = None
    if hasattr(event, "error") and event.error is not None:
        error_msg = str(event.error)

    return spine_cls(
        # Copy identity fields from the source event
        trace_id=event.trace_id,
        span_id=event.event_id,  # Each event is its own span for now
        causation_id=event.causation_id,
        timestamp=event.timestamp,
        schema_version=event.schema_version,
        # Spine-specific fields
        tenant_id=tenant_id,
        actor=event.source_module or "system",
        component=event.source_module,
        event_type=taxonomy,
        input_hash=input_hash,
        output_hash=output_hash,
        payload=payload,
        latency_ms=latency_ms,
        error=error_msg,
    )
