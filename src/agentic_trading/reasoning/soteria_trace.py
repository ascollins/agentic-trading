"""Soteria Trace — extended reasoning capture with inter-agent context.

Extends the base ``ReasoningTrace`` concept with:

- ``StepType`` enum (superset of ``ReasoningPhase`` — adds CONTEXT_LOAD,
  HANDOFF, VETO, OUTCOME phases for inter-agent workflows)
- ``SoteriaStep`` — includes context_used dict and messages_sent list
- ``SoteriaTrace`` — full agent trace with thinking_raw, trigger,
  final_output, and cross-references to AgentMessages

These co-exist alongside the original ``ReasoningTrace``/``ReasoningStep``
in ``reasoning/models.py``.  The originals are for pipeline-level
single-agent traces; Soteria traces are for multi-agent conversations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .agent_message import AgentRole


# ---------------------------------------------------------------------------
# StepType  (superset of ReasoningPhase)
# ---------------------------------------------------------------------------


class StepType(str, Enum):
    """Phases in structured Soteria agent reasoning.

    Superset of ``ReasoningPhase`` — adds CONTEXT_LOAD, HANDOFF,
    VETO, and OUTCOME for inter-agent conversation flows.
    """

    PERCEPTION = "perception"
    CONTEXT_LOAD = "context_load"
    HYPOTHESIS = "hypothesis"
    EVALUATION = "evaluation"
    DECISION = "decision"
    ACTION = "action"
    HANDOFF = "handoff"
    VETO = "veto"
    OUTCOME = "outcome"

    @property
    def display_label(self) -> str:
        """Human-readable label with emoji prefix."""
        _labels = {
            "perception": "👁️ Perception",
            "context_load": "📚 Context Load",
            "hypothesis": "💡 Hypothesis",
            "evaluation": "⚖️ Evaluation",
            "decision": "🎯 Decision",
            "action": "⚡ Action",
            "handoff": "🤝 Handoff",
            "veto": "🚫 Veto",
            "outcome": "📊 Outcome",
        }
        return _labels.get(self.value, self.value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# SoteriaStep
# ---------------------------------------------------------------------------


class SoteriaStep(BaseModel):
    """A single reasoning step in a Soteria trace.

    Extends the concept from ``ReasoningStep`` with:
    - ``context_used``: snapshot of what the agent read before this step
    - ``messages_sent``: IDs of AgentMessages emitted during this step
    """

    step_id: str = Field(default_factory=_uuid)
    step_type: StepType
    content: str = ""
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=_now)
    evidence: dict[str, Any] = Field(default_factory=dict)
    context_used: dict[str, Any] = Field(default_factory=dict)
    messages_sent: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SoteriaTrace
# ---------------------------------------------------------------------------


class SoteriaTrace(BaseModel):
    """Complete Soteria reasoning trace for one agent in a conversation.

    Captures the full reasoning chain including:
    - Extended thinking from Claude API (``thinking_raw``)
    - Context loaded from FactTable/MemoryStore (``context_used``)
    - Messages sent/received during reasoning
    - Trigger event that started this trace
    - Final output produced

    Parameters
    ----------
    trace_id:
        Unique trace identifier.
    conversation_id:
        The conversation this trace belongs to.
    agent_role:
        Desk role of the agent.
    agent_id:
        Infrastructure agent identifier.
    symbol:
        Instrument being analyzed (if any).
    trigger:
        What initiated this reasoning cycle (e.g. "candle.BTC/USDT.15m").
    """

    trace_id: str = Field(default_factory=_uuid)
    conversation_id: str = ""
    agent_role: AgentRole
    agent_id: str = ""
    symbol: str = ""
    trigger: str = ""

    started_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None

    steps: list[SoteriaStep] = Field(default_factory=list)

    thinking_raw: str = ""
    context_used: dict[str, Any] = Field(default_factory=dict)
    messages_sent: list[str] = Field(default_factory=list)
    messages_received: list[str] = Field(default_factory=list)

    final_output: dict[str, Any] = Field(default_factory=dict)
    outcome: str = ""

    @property
    def duration_ms(self) -> float:
        """Duration of the trace in milliseconds."""
        if self.completed_at and self.started_at:
            return (
                self.completed_at - self.started_at
            ).total_seconds() * 1000
        return 0.0

    def add_step(
        self,
        step_type: StepType,
        content: str,
        confidence: float = 0.0,
        *,
        evidence: dict[str, Any] | None = None,
        context_used: dict[str, Any] | None = None,
        messages_sent: list[str] | None = None,
        **metadata: Any,
    ) -> SoteriaStep:
        """Add a reasoning step and return it."""
        step = SoteriaStep(
            step_type=step_type,
            content=content,
            confidence=confidence,
            evidence=evidence or {},
            context_used=context_used or {},
            messages_sent=messages_sent or [],
            metadata=metadata,
        )
        self.steps.append(step)
        return step

    def complete(self, outcome: str, final_output: dict[str, Any] | None = None) -> None:
        """Mark the trace as complete."""
        self.completed_at = _now()
        self.outcome = outcome
        if final_output is not None:
            self.final_output = final_output

    def format_trace(self) -> str:
        """Human-readable rendering of this trace."""
        lines = [
            f"=== {self.agent_role.display_name} ({self.agent_id[:8] if self.agent_id else '?'}) ===",
            f"  Symbol: {self.symbol or 'N/A'}",
            f"  Trigger: {self.trigger or 'N/A'}",
            f"  Duration: {self.duration_ms:.0f}ms",
            f"  Outcome: {self.outcome or 'in_progress'}",
        ]

        for step in self.steps:
            conf = f" [{step.confidence:.0%}]" if step.confidence > 0 else ""
            lines.append(
                f"  {step.step_type.display_label}{conf}: {step.content}"
            )

        if self.thinking_raw:
            preview = self.thinking_raw[:200]
            if len(self.thinking_raw) > 200:
                preview += "..."
            lines.append(f"  [extended_thinking] {preview}")

        return "\n".join(lines)
