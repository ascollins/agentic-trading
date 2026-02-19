"""Pipeline Result — the inspectable output of every orchestrator pipeline run.

Contains all reasoning traces, context snapshots, and the final outcome.
Call ``.print_chain_of_thought()`` for a human-readable rendering.
Call ``.to_json()`` for serialization.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.enums import PipelineOutcome


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PipelineResult(BaseModel):
    """Complete result from a single pipeline run.

    Contains all reasoning traces, context snapshots, and the final
    outcome. Serializable to JSON for logging and replay.
    """

    pipeline_id: str = Field(default_factory=_uuid)
    started_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None

    # Trigger
    trigger_event_type: str = ""
    trigger_symbol: str = ""
    trigger_timeframe: str = ""

    # Context snapshots
    context_at_start: dict[str, Any] = Field(default_factory=dict)
    context_at_end: dict[str, Any] = Field(default_factory=dict)

    # Reasoning chain (serialized traces)
    reasoning_traces: list[dict[str, Any]] = Field(default_factory=list)

    # Outcome
    outcome: PipelineOutcome = PipelineOutcome.NO_SIGNAL
    outcome_details: dict[str, Any] = Field(default_factory=dict)

    # Signals and intents produced
    signals: list[dict[str, Any]] = Field(default_factory=list)
    intents: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Duration of the pipeline run in milliseconds."""
        if self.completed_at and self.started_at:
            return (
                self.completed_at - self.started_at
            ).total_seconds() * 1000
        return 0.0

    def print_chain_of_thought(self) -> str:
        """Render the full reasoning chain as human-readable text."""
        lines = [
            f"Pipeline {self.pipeline_id}",
            f"Trigger: {self.trigger_event_type} {self.trigger_symbol} "
            f"{self.trigger_timeframe}",
            f"Started: {self.started_at.isoformat()}",
            f"Duration: {self.duration_ms:.0f}ms",
            f"Outcome: {self.outcome.value}",
            "",
        ]

        for trace_data in self.reasoning_traces:
            agent_type = trace_data.get("agent_type", "unknown")
            agent_id = str(trace_data.get("agent_id", ""))[:8]
            lines.append(f"--- {agent_type}: {agent_id} ---")

            symbol = trace_data.get("symbol", "")
            if symbol:
                lines.append(f"  Symbol: {symbol}")

            outcome = trace_data.get("outcome", "")
            if outcome:
                lines.append(f"  Outcome: {outcome}")

            for step in trace_data.get("steps", []):
                phase = step.get("phase", "")
                content = step.get("content", "")
                confidence = step.get("confidence", 0)
                conf_str = (
                    f" [{confidence:.0%}]" if confidence > 0 else ""
                )
                lines.append(f"  [{phase}]{conf_str} {content}")

            raw = trace_data.get("raw_thinking", "")
            if raw:
                preview = raw[:300]
                if len(raw) > 300:
                    preview += "..."
                lines.append(f"  [extended_thinking] {preview}")

            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    def finalize(self) -> None:
        """Mark the pipeline as complete."""
        self.completed_at = _now()
