"""SpineEvent model and taxonomy subclasses.

The spine is a parallel append-only event log that sits alongside the
existing event bus.  SpineEvent is *not* a subclass of BaseEvent -- the
mapper (``mapper.py``) bridges the two.

Every spine event carries 15 mandatory fields designed for
cross-component trace queries.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.ids import (
    new_id as _uuid,
)
from agentic_trading.core.ids import (
    utc_now as _now,
)

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class EventTaxonomy(str, Enum):
    """Nine-type classification for every spine event."""

    USER_ACTION = "user_action"
    AGENT_STEP_STARTED = "agent_step_started"
    AGENT_STEP_COMPLETED = "agent_step_completed"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    DECISION = "decision"
    VALIDATION = "validation"
    EXCEPTION = "exception"
    COST_METRIC = "cost_metric"


# ---------------------------------------------------------------------------
# SpineEvent (base)
# ---------------------------------------------------------------------------


class SpineEvent(BaseModel):
    """Base spine event.  All 15 mandatory fields live here."""

    event_id: str = Field(default_factory=_uuid)
    trace_id: str = Field(default_factory=_uuid)
    span_id: str = Field(default_factory=_uuid)
    tenant_id: str = "default"
    actor: str = "system"
    component: str = ""
    timestamp: datetime = Field(default_factory=_now)
    input_hash: str = ""
    output_hash: str = ""
    schema_version: int = 1
    event_type: EventTaxonomy = EventTaxonomy.AGENT_STEP_STARTED
    causation_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Taxonomy subclasses -- each pins ``event_type`` to its taxonomy value.
# Taxonomy-specific data goes in ``payload``.
# ---------------------------------------------------------------------------


class UserActionEvent(SpineEvent):
    """An action initiated by a human user (CLI, API, UI)."""

    event_type: EventTaxonomy = EventTaxonomy.USER_ACTION


class AgentStepStartedEvent(SpineEvent):
    """An agent step has begun (signal generation, risk check, etc.)."""

    event_type: EventTaxonomy = EventTaxonomy.AGENT_STEP_STARTED


class AgentStepCompletedEvent(SpineEvent):
    """An agent step has completed with output."""

    event_type: EventTaxonomy = EventTaxonomy.AGENT_STEP_COMPLETED


class ToolCallEvent(SpineEvent):
    """A tool/exchange call was made via the ToolGateway."""

    event_type: EventTaxonomy = EventTaxonomy.TOOL_CALL


class RetrievalEvent(SpineEvent):
    """A data retrieval operation (candle fetch, fact table query, etc.)."""

    event_type: EventTaxonomy = EventTaxonomy.RETRIEVAL


class DecisionEvent(SpineEvent):
    """A deterministic decision (risk check, policy evaluation, governance gate)."""

    event_type: EventTaxonomy = EventTaxonomy.DECISION


class ValidationEvent(SpineEvent):
    """A schema or data validation check."""

    event_type: EventTaxonomy = EventTaxonomy.VALIDATION


class ExceptionEvent(SpineEvent):
    """An error or circuit-breaker trip."""

    event_type: EventTaxonomy = EventTaxonomy.EXCEPTION


class CostMetricEvent(SpineEvent):
    """LLM token usage or other cost metric."""

    event_type: EventTaxonomy = EventTaxonomy.COST_METRIC
