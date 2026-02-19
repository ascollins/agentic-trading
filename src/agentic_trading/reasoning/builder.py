"""Reasoning Builder — convenience helper for agents.

Provides a simple API for agents to create reasoning traces without
importing models directly. Can be used as a utility or mixed into
agent classes.
"""

from __future__ import annotations

from .models import ReasoningTrace


class ReasoningBuilder:
    """Helper for agents to create reasoning traces.

    Usage::

        builder = ReasoningBuilder("agent-123", "market_intelligence")
        trace = builder.start(symbol="BTC/USDT", pipeline_id="pipe-456")
        trace.add_step(ReasoningPhase.PERCEPTION, "Price at 65000, uptrend")
        trace.add_step(ReasoningPhase.DECISION, "Go long", confidence=0.8)
        builder.complete(trace, "signal_emitted")
    """

    def __init__(self, agent_id: str, agent_type: str) -> None:
        self._agent_id = agent_id
        self._agent_type = agent_type

    def start(
        self,
        symbol: str = "",
        pipeline_id: str = "",
    ) -> ReasoningTrace:
        """Create a new reasoning trace for this agent."""
        return ReasoningTrace(
            agent_id=self._agent_id,
            agent_type=self._agent_type,
            symbol=symbol,
            pipeline_id=pipeline_id,
        )

    @staticmethod
    def complete(trace: ReasoningTrace, outcome: str) -> None:
        """Mark a trace as complete with the given outcome."""
        trace.complete(outcome)
