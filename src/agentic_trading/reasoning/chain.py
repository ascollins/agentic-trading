"""Reasoning Chain — collects traces from all agents in a pipeline run.

The ``ReasoningChain`` is created per pipeline invocation and accumulates
``ReasoningTrace`` instances from every participating agent. It can
render the full chain of thought as human-readable text.
"""

from __future__ import annotations

from typing import Any

from .models import ReasoningTrace


class ReasoningChain:
    """Collects ReasoningTraces from all agents in a pipeline run."""

    def __init__(self, pipeline_id: str) -> None:
        self._pipeline_id = pipeline_id
        self._traces: list[ReasoningTrace] = []

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def traces(self) -> list[ReasoningTrace]:
        return list(self._traces)

    def create_trace(
        self,
        agent_id: str,
        agent_type: str,
        symbol: str = "",
    ) -> ReasoningTrace:
        """Create and register a new trace for an agent."""
        trace = ReasoningTrace(
            pipeline_id=self._pipeline_id,
            agent_id=agent_id,
            agent_type=agent_type,
            symbol=symbol,
        )
        self._traces.append(trace)
        return trace

    def add_trace(self, trace: ReasoningTrace) -> None:
        """Add a pre-built trace to the chain."""
        self._traces.append(trace)

    def format_chain_of_thought(self) -> str:
        """Render the full chain as human-readable text."""
        lines: list[str] = []
        lines.append(f"=== Pipeline {self._pipeline_id} ===")

        for trace in self._traces:
            lines.append(
                f"\n--- {trace.agent_type}: {trace.agent_id[:8]} ---"
            )
            if trace.symbol:
                lines.append(f"    Symbol: {trace.symbol}")
            lines.append(f"    Outcome: {trace.outcome}")
            lines.append(f"    Duration: {trace.duration_ms:.0f}ms")

            for step in trace.steps:
                conf_str = (
                    f" [{step.confidence:.0%}]" if step.confidence > 0 else ""
                )
                lines.append(
                    f"    [{step.phase.value}]{conf_str} {step.content}"
                )

            if trace.raw_thinking:
                preview = trace.raw_thinking[:200]
                if len(trace.raw_thinking) > 200:
                    preview += "..."
                lines.append(f"    [extended_thinking] {preview}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "pipeline_id": self._pipeline_id,
            "traces": [
                t.model_dump(mode="json") for t in self._traces
            ],
        }
