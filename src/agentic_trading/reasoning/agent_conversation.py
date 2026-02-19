"""Agent Conversation — the top-level container for a reasoning session.

An ``AgentConversation`` groups all ``AgentMessage`` instances and
``SoteriaTrace`` instances for one reasoning cycle (typically triggered
by a candle close or market event).

Three rendering methods:
- ``print_desk_conversation()`` — Bloomberg-terminal-style desk chat
- ``print_chain_of_thought()`` — numbered steps with indentation
- ``explain()`` — narrative summary for non-technical consumers
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .agent_message import AgentMessage, AgentRole, MessageType
from .soteria_trace import SoteriaTrace


# ---------------------------------------------------------------------------
# ConversationOutcome
# ---------------------------------------------------------------------------


class ConversationOutcome(str, Enum):
    """Possible outcomes of a multi-agent reasoning conversation."""

    TRADE_ENTERED = "trade_entered"
    TRADE_EXITED = "trade_exited"
    NO_TRADE = "no_trade"
    VETOED = "vetoed"
    ERROR = "error"
    DEBRIEF = "debrief"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# AgentConversation
# ---------------------------------------------------------------------------


class AgentConversation(BaseModel):
    """Top-level container for one multi-agent reasoning session.

    Attributes
    ----------
    conversation_id:
        Unique identifier.
    symbol:
        Primary instrument under discussion.
    timeframe:
        Trigger timeframe (e.g. "15m").
    trigger_event:
        What started this conversation (e.g. "candle.BTC/USDT.15m").
    strategy_id:
        Strategy that initiated the conversation (if any).
    started_at:
        UTC timestamp when the conversation began.
    completed_at:
        UTC timestamp when the conversation ended.
    messages:
        Ordered list of all inter-agent messages.
    traces:
        Per-agent reasoning traces.
    outcome:
        Final outcome of the conversation.
    outcome_details:
        Extra data about the outcome (signal params, veto reason, etc.).
    context_snapshot:
        Snapshot of FactTable/MemoryStore at conversation start.
    """

    conversation_id: str = Field(default_factory=_uuid)
    symbol: str = ""
    timeframe: str = ""
    trigger_event: str = ""
    strategy_id: str = ""

    started_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None

    messages: list[AgentMessage] = Field(default_factory=list)
    traces: list[SoteriaTrace] = Field(default_factory=list)

    outcome: ConversationOutcome = ConversationOutcome.NO_TRADE
    outcome_details: dict[str, Any] = Field(default_factory=dict)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration of the conversation in milliseconds."""
        if self.completed_at and self.started_at:
            return (
                self.completed_at - self.started_at
            ).total_seconds() * 1000
        return 0.0

    @property
    def has_veto(self) -> bool:
        """True if any message is a VETO."""
        return any(m.is_veto for m in self.messages)

    @property
    def has_disagreement(self) -> bool:
        """True if agents disagreed (CHALLENGE or DISAGREEMENT messages)."""
        return any(m.is_challenge for m in self.messages)

    @property
    def participating_roles(self) -> list[AgentRole]:
        """Unique roles that participated (in order of first appearance)."""
        seen: set[AgentRole] = set()
        roles: list[AgentRole] = []
        for msg in self.messages:
            if msg.sender not in seen:
                seen.add(msg.sender)
                roles.append(msg.sender)
        return roles

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the conversation."""
        if not message.conversation_id:
            message.conversation_id = self.conversation_id
        self.messages.append(message)

    def add_trace(self, trace: SoteriaTrace) -> None:
        """Add an agent trace to the conversation."""
        if not trace.conversation_id:
            trace.conversation_id = self.conversation_id
        self.traces.append(trace)

    def finalize(
        self,
        outcome: ConversationOutcome,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Mark the conversation as complete."""
        self.completed_at = _now()
        self.outcome = outcome
        if details:
            self.outcome_details = details

    # ------------------------------------------------------------------
    # Rendering: Bloomberg desk conversation
    # ------------------------------------------------------------------

    def print_desk_conversation(self) -> str:
        """Render as a Bloomberg-terminal-style desk chat.

        Format:
        ```
        ═══════════════════════════════════════════════════════
        DESK CONVERSATION: BTC/USDT 15m | 2024-01-15 14:00 UTC
        ═══════════════════════════════════════════════════════

        14:00:01 [Market Structure → ALL] ANALYSIS:
          Higher timeframe shows bullish structure...

        14:00:02 [SMC Analyst → ALL] SIGNAL [85%]:
          Order block identified at 42,800...

        14:00:03 [Risk Manager → Orchestrator] VETO:
          Exposure limit exceeded...

        ─────────────────────────────────────────────────────
        OUTCOME: VETOED | Duration: 245ms | Messages: 5
        ─────────────────────────────────────────────────────
        ```
        """
        ts = self.started_at.strftime("%Y-%m-%d %H:%M UTC")
        width = 60

        lines = [
            "═" * width,
            f"DESK CONVERSATION: {self.symbol} {self.timeframe} | {ts}",
            f"Strategy: {self.strategy_id or 'N/A'}",
            "═" * width,
            "",
        ]

        for msg in self.messages:
            time_str = msg.timestamp.strftime("%H:%M:%S")
            recip = (
                "ALL"
                if msg.is_broadcast
                else ",".join(r.display_name for r in msg.recipients)
            )
            conf = f" [{msg.confidence:.0%}]" if msg.confidence > 0 else ""

            lines.append(
                f"{time_str} [{msg.sender.display_name} → {recip}] "
                f"{msg.message_type.value.upper()}{conf}:"
            )

            # Indent message content
            for content_line in msg.content.split("\n"):
                lines.append(f"  {content_line}")

            # Show structured data highlights
            if msg.structured_data:
                highlights = _format_structured_highlights(msg.structured_data)
                if highlights:
                    lines.append(f"  📊 {highlights}")

            lines.append("")

        lines.append("─" * width)
        dur = f"{self.duration_ms:.0f}ms" if self.completed_at else "in progress"
        veto_flag = " 🚫" if self.has_veto else ""
        disagree_flag = " ⚠️" if self.has_disagreement else ""
        lines.append(
            f"OUTCOME: {self.outcome.value.upper()}{veto_flag}{disagree_flag} "
            f"| Duration: {dur} | Messages: {len(self.messages)}"
        )
        lines.append("─" * width)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rendering: Chain of thought
    # ------------------------------------------------------------------

    def print_chain_of_thought(self) -> str:
        """Render the reasoning chain as numbered steps.

        Groups by agent, shows each reasoning step with phase,
        confidence, and content.
        """
        lines = [
            f"CHAIN OF THOUGHT: {self.symbol} {self.timeframe}",
            f"Conversation: {self.conversation_id[:12]}...",
            f"Outcome: {self.outcome.value}",
            "",
        ]

        for i, trace in enumerate(self.traces, 1):
            lines.append(
                f"Agent {i}: {trace.agent_role.display_name} "
                f"({trace.agent_id[:8] if trace.agent_id else '?'})"
            )
            lines.append(f"  Trigger: {trace.trigger or 'N/A'}")
            lines.append(
                f"  Duration: {trace.duration_ms:.0f}ms"
            )
            lines.append(f"  Outcome: {trace.outcome or 'in_progress'}")

            for j, step in enumerate(trace.steps, 1):
                conf = (
                    f" [{step.confidence:.0%}]"
                    if step.confidence > 0
                    else ""
                )
                lines.append(
                    f"  {i}.{j} {step.step_type.display_label}{conf}: "
                    f"{step.content}"
                )

            if trace.thinking_raw:
                preview = trace.thinking_raw[:200]
                if len(trace.thinking_raw) > 200:
                    preview += "..."
                lines.append(f"  [extended_thinking] {preview}")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rendering: Plain-English explanation
    # ------------------------------------------------------------------

    def explain(self) -> str:
        """Generate a narrative explanation of the conversation.

        Suitable for non-technical consumers — reads like a short
        paragraph explaining what happened and why.
        """
        parts: list[str] = []

        # Opening
        ts = self.started_at.strftime("%H:%M UTC on %B %d")
        parts.append(
            f"At {ts}, the desk analyzed {self.symbol} on the "
            f"{self.timeframe} timeframe"
        )
        if self.trigger_event:
            parts.append(f" (triggered by {self.trigger_event})")
        parts.append(".\n\n")

        # Agent contributions
        for trace in self.traces:
            role_name = trace.agent_role.display_name
            if trace.outcome:
                parts.append(
                    f"{role_name}: {trace.outcome}. "
                )
            # Grab the decision step if present
            decision_steps = [
                s for s in trace.steps
                if s.step_type.value in ("decision", "veto")
            ]
            if decision_steps:
                step = decision_steps[0]
                conf = (
                    f" (confidence: {step.confidence:.0%})"
                    if step.confidence > 0
                    else ""
                )
                parts.append(f"{step.content}{conf}. ")

        parts.append("\n\n")

        # Outcome
        outcome_text = self.outcome.value.replace("_", " ").title()
        parts.append(f"Final outcome: {outcome_text}")

        if self.has_veto:
            veto_msgs = [
                m for m in self.messages if m.is_veto
            ]
            if veto_msgs:
                parts.append(
                    f" — vetoed by {veto_msgs[0].sender.display_name}: "
                    f"{veto_msgs[0].content[:100]}"
                )

        if self.has_disagreement:
            parts.append(" (agents disagreed during analysis)")

        dur = self.duration_ms
        parts.append(f". Total reasoning time: {dur:.0f}ms.")

        return "".join(parts)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_structured_highlights(data: dict[str, Any]) -> str:
    """Extract key highlights from structured message data."""
    parts: list[str] = []

    # Common fields to surface
    for key in ("direction", "signal", "side", "action"):
        if key in data:
            parts.append(f"{key}={data[key]}")

    for key in ("confidence", "score", "risk_score"):
        if key in data:
            val = data[key]
            if isinstance(val, float):
                parts.append(f"{key}={val:.2%}")
            else:
                parts.append(f"{key}={val}")

    for key in ("symbol", "entry_price", "stop_loss", "take_profit", "size"):
        if key in data:
            parts.append(f"{key}={data[key]}")

    if not parts:
        # Show first 3 keys as fallback
        for key in list(data.keys())[:3]:
            val = data[key]
            if isinstance(val, (str, int, float, bool)):
                parts.append(f"{key}={val}")

    return " | ".join(parts)
