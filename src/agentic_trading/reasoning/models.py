"""Reasoning models — structured capture of agent thinking.

Every agent's reasoning is captured as a ``ReasoningTrace`` containing
ordered ``ReasoningStep`` instances that follow the pipeline:
perception -> hypothesis -> evaluation -> decision -> action.

For Claude API calls (e.g. CMT analyst), the extended thinking block
is captured separately as ``raw_thinking``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.enums import ReasoningPhase


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReasoningStep(BaseModel):
    """A single step in an agent's reasoning chain."""

    step_id: str = Field(default_factory=_uuid)
    phase: ReasoningPhase
    content: str
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReasoningTrace(BaseModel):
    """Complete reasoning trace for one agent in one pipeline run.

    Captures perception through action, plus optional extended
    thinking content from Claude API calls.
    """

    trace_id: str = Field(default_factory=_uuid)
    pipeline_id: str = ""
    agent_id: str = ""
    agent_type: str = ""
    symbol: str = ""
    started_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None
    steps: list[ReasoningStep] = Field(default_factory=list)
    outcome: str = ""
    raw_thinking: str = ""

    @property
    def duration_ms(self) -> float:
        """Duration of the trace in milliseconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def add_step(
        self,
        phase: ReasoningPhase,
        content: str,
        confidence: float = 0.0,
        *,
        evidence: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> ReasoningStep:
        """Add a reasoning step and return it."""
        step = ReasoningStep(
            phase=phase,
            content=content,
            confidence=confidence,
            metadata=metadata,
            evidence=evidence or {},
        )
        self.steps.append(step)
        return step

    def complete(self, outcome: str) -> None:
        """Mark the trace as complete."""
        self.completed_at = _now()
        self.outcome = outcome
