"""Telemetry spine -- parallel append-only event log.

Public API
----------
::

    from agentic_trading.telemetry import (
        EventWriter,
        SpineEvent,
        EventTaxonomy,
        SpanContext,
        MemorySpineStorage,
        PostgresSpineStorage,
        map_base_event_to_spine,
    )
"""

from __future__ import annotations

from agentic_trading.telemetry.event_writer import EventWriter
from agentic_trading.telemetry.mapper import map_base_event_to_spine
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
from agentic_trading.telemetry.spans import SpanContext
from agentic_trading.telemetry.storage import (
    ISpineStorage,
    MemorySpineStorage,
    PostgresSpineStorage,
)

__all__ = [
    # Core
    "EventWriter",
    "SpineEvent",
    "EventTaxonomy",
    "SpanContext",
    # Storage
    "ISpineStorage",
    "MemorySpineStorage",
    "PostgresSpineStorage",
    # Mapper
    "map_base_event_to_spine",
    # Taxonomy subclasses
    "AgentStepCompletedEvent",
    "AgentStepStartedEvent",
    "CostMetricEvent",
    "DecisionEvent",
    "ExceptionEvent",
    "RetrievalEvent",
    "ToolCallEvent",
    "UserActionEvent",
    "ValidationEvent",
]
