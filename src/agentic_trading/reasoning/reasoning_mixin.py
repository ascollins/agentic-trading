"""Reasoning Mixin — opt-in Soteria reasoning for existing strategies.

Any strategy (or agent) can mix in ``ReasoningMixin`` to gain:
- Auto-creation of SoteriaTraces per reasoning cycle
- Convenience methods for posting messages to the ReasoningMessageBus
- Context loading from ContextManager
- Extended thinking capture from Claude API calls

Usage:
```python
class MySMCStrategy(BaseAgent, ReasoningMixin):
    def __init__(self, ...):
        BaseAgent.__init__(self, ...)
        ReasoningMixin.__init__(self, role=AgentRole.SMC_ANALYST)
```

This is *additive only* — no BaseAgent or BaseStrategy modifications.
Strategies without the mixin work exactly as before.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .agent_conversation import ConversationOutcome
from .agent_message import AgentMessage, AgentRole, MessageType
from .message_bus import ReasoningMessageBus
from .soteria_trace import SoteriaStep, SoteriaTrace, StepType

logger = logging.getLogger(__name__)


class ReasoningMixin:
    """Opt-in mixin that adds Soteria reasoning to any agent/strategy.

    Provides convenience methods for trace creation, message posting,
    and context loading without modifying the base class hierarchy.

    Attributes
    ----------
    _reasoning_role:
        The agent's desk role for message routing.
    _reasoning_bus:
        The reasoning message bus (injected via ``set_reasoning_bus``).
    _current_trace:
        The trace being built for the current reasoning cycle.
    _current_conversation_id:
        Active conversation ID.
    """

    def __init__(self, *, role: AgentRole) -> None:
        self._reasoning_role = role
        self._reasoning_bus: ReasoningMessageBus | None = None
        self._current_trace: SoteriaTrace | None = None
        self._current_conversation_id: str = ""

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def set_reasoning_bus(self, bus: ReasoningMessageBus) -> None:
        """Inject the reasoning message bus."""
        self._reasoning_bus = bus

    @property
    def reasoning_role(self) -> AgentRole:
        return self._reasoning_role

    @property
    def reasoning_bus(self) -> ReasoningMessageBus | None:
        return self._reasoning_bus

    @property
    def has_reasoning(self) -> bool:
        """True if reasoning infrastructure is available."""
        return self._reasoning_bus is not None

    # ------------------------------------------------------------------
    # Trace lifecycle
    # ------------------------------------------------------------------

    def begin_reasoning(
        self,
        *,
        conversation_id: str = "",
        symbol: str = "",
        trigger: str = "",
        context_used: dict[str, Any] | None = None,
    ) -> SoteriaTrace:
        """Start a new reasoning trace for this agent.

        Parameters
        ----------
        conversation_id:
            Link to the parent conversation.
        symbol:
            Instrument being analyzed.
        trigger:
            What triggered this reasoning (e.g. "candle.BTC/USDT.15m").
        context_used:
            Snapshot of context loaded from FactTable/MemoryStore.

        Returns
        -------
        SoteriaTrace
            The trace object. Call ``add_reasoning_step()`` to populate.
        """
        agent_id = getattr(self, "_agent_id", getattr(self, "agent_id", ""))
        trace = SoteriaTrace(
            conversation_id=conversation_id,
            agent_role=self._reasoning_role,
            agent_id=agent_id,
            symbol=symbol,
            trigger=trigger,
            context_used=context_used or {},
        )
        self._current_trace = trace
        self._current_conversation_id = conversation_id

        # Register with bus conversation if available
        if self._reasoning_bus and conversation_id:
            conv = self._reasoning_bus.get_conversation(conversation_id)
            if conv is not None:
                conv.add_trace(trace)

        logger.debug(
            "Reasoning started: role=%s symbol=%s trace=%s",
            self._reasoning_role.value,
            symbol,
            trace.trace_id[:8],
        )
        return trace

    def add_reasoning_step(
        self,
        step_type: StepType,
        content: str,
        confidence: float = 0.0,
        *,
        evidence: dict[str, Any] | None = None,
        context_used: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> SoteriaStep | None:
        """Add a step to the current trace.

        Returns None if no trace is active.
        """
        if self._current_trace is None:
            return None
        return self._current_trace.add_step(
            step_type=step_type,
            content=content,
            confidence=confidence,
            evidence=evidence,
            context_used=context_used,
            **metadata,
        )

    def end_reasoning(
        self,
        outcome: str,
        final_output: dict[str, Any] | None = None,
    ) -> SoteriaTrace | None:
        """Complete the current reasoning trace.

        Returns the completed trace, or None if no trace was active.
        """
        if self._current_trace is None:
            return None
        self._current_trace.complete(outcome, final_output)
        trace = self._current_trace
        self._current_trace = None
        logger.debug(
            "Reasoning complete: role=%s outcome=%s duration=%dms",
            self._reasoning_role.value,
            outcome,
            trace.duration_ms,
        )
        return trace

    # ------------------------------------------------------------------
    # Message posting
    # ------------------------------------------------------------------

    def post_message(
        self,
        message_type: MessageType,
        content: str,
        *,
        recipients: list[AgentRole] | None = None,
        structured_data: dict[str, Any] | None = None,
        confidence: float = 0.0,
        references: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage | None:
        """Post a message to the reasoning bus.

        Returns the message if posted, None if no bus available.
        """
        if self._reasoning_bus is None:
            return None

        msg = AgentMessage(
            conversation_id=self._current_conversation_id,
            sender=self._reasoning_role,
            recipients=recipients or [],
            message_type=message_type,
            content=content,
            structured_data=structured_data or {},
            confidence=confidence,
            references=references or [],
            metadata=metadata or {},
        )

        # Track in current trace
        if self._current_trace is not None:
            self._current_trace.messages_sent.append(msg.message_id)

        self._reasoning_bus.post(msg)
        return msg

    def post_analysis(
        self,
        content: str,
        *,
        confidence: float = 0.0,
        structured_data: dict[str, Any] | None = None,
    ) -> AgentMessage | None:
        """Shorthand for posting an ANALYSIS message (broadcast)."""
        return self.post_message(
            MessageType.ANALYSIS,
            content,
            confidence=confidence,
            structured_data=structured_data,
        )

    def post_signal(
        self,
        content: str,
        *,
        confidence: float = 0.0,
        structured_data: dict[str, Any] | None = None,
    ) -> AgentMessage | None:
        """Shorthand for posting a SIGNAL message (broadcast)."""
        return self.post_message(
            MessageType.SIGNAL,
            content,
            confidence=confidence,
            structured_data=structured_data,
        )

    def post_veto(
        self,
        content: str,
        *,
        recipients: list[AgentRole] | None = None,
        structured_data: dict[str, Any] | None = None,
    ) -> AgentMessage | None:
        """Post a VETO message."""
        return self.post_message(
            MessageType.VETO,
            content,
            recipients=recipients or [AgentRole.ORCHESTRATOR],
            structured_data=structured_data,
            confidence=1.0,  # Vetoes are always high-confidence
        )

    def post_challenge(
        self,
        content: str,
        *,
        references: list[str] | None = None,
        confidence: float = 0.0,
    ) -> AgentMessage | None:
        """Post a CHALLENGE / DISAGREEMENT message (broadcast)."""
        return self.post_message(
            MessageType.CHALLENGE,
            content,
            confidence=confidence,
            references=references,
        )

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def load_reasoning_context(
        self, symbol: str | None = None
    ) -> dict[str, Any]:
        """Load context from ContextManager (if available on self).

        Returns a dict snapshot suitable for trace.context_used.
        Falls back to empty dict if no ContextManager.
        """
        cm = getattr(self, "_context_manager", None)
        if cm is None:
            return {}

        try:
            ctx = cm.read_context(symbol=symbol)
            snapshot = ctx.fact_snapshot
            return {
                "portfolio": snapshot.portfolio.model_dump() if snapshot.portfolio else {},
                "risk": snapshot.risk.model_dump() if snapshot.risk else {},
                "prices": {
                    sym: p.model_dump()
                    for sym, p in (snapshot.prices or {}).items()
                },
                "regimes": snapshot.regimes or {},
                "memories": [
                    {"entry_type": m.entry_type.value, "summary": m.summary}
                    for m in ctx.relevant_memories
                ],
            }
        except Exception:
            logger.debug("Could not load reasoning context", exc_info=True)
            return {}
