"""Agent Message — inter-agent communication for Soteria reasoning.

Every message exchanged between agents during a reasoning conversation
is an ``AgentMessage``. Messages carry structured content (analysis,
signals, vetoes, challenges) and are threaded by ``conversation_id``.

The ``AgentRole`` enum maps desk roles (7 specialists), while
``MessageType`` classifies the 13 message categories that flow
between agents.

This is separate from the trading ``EventBus`` — reasoning messages
are about *thinking*, not *trading events*.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.enums import AgentType


# ---------------------------------------------------------------------------
# Enums  (str, Enum pattern — consistent with core/enums.py)
# ---------------------------------------------------------------------------

from enum import Enum


class AgentRole(str, Enum):
    """Desk roles in the Soteria reasoning layer.

    Maps to AgentType for wiring, but represents the *reasoning role*
    an agent plays in a conversation — not the infrastructure type.
    """

    MARKET_STRUCTURE = "market_structure"
    SMC_ANALYST = "smc_analyst"
    CMT_TECHNICIAN = "cmt_technician"
    RISK_MANAGER = "risk_manager"
    EXECUTION = "execution"
    ORCHESTRATOR = "orchestrator"
    BROADCAST = "broadcast"

    @property
    def display_name(self) -> str:
        """Human-readable role name."""
        _names = {
            "market_structure": "Market Structure",
            "smc_analyst": "SMC Analyst",
            "cmt_technician": "CMT Technician",
            "risk_manager": "Risk Manager",
            "execution": "Execution",
            "orchestrator": "Orchestrator",
            "broadcast": "Broadcast",
        }
        return _names.get(self.value, self.value)


class MessageType(str, Enum):
    """Categories of inter-agent messages.

    These flow over the ``ReasoningMessageBus`` during a conversation,
    not over the trading event bus.
    """

    MORNING_BRIEF = "morning_brief"
    MARKET_UPDATE = "market_update"
    ANALYSIS = "analysis"
    SIGNAL = "signal"
    CHALLENGE = "challenge"
    RESPONSE = "response"
    RISK_ASSESSMENT = "risk_assessment"
    VETO = "veto"
    EXECUTION_PLAN = "execution_plan"
    FILL_REPORT = "fill_report"
    DEBRIEF = "debrief"
    SYSTEM = "system"
    DISAGREEMENT = "disagreement"


# ---------------------------------------------------------------------------
# AgentRole <-> AgentType mapping
# ---------------------------------------------------------------------------

_ROLE_TO_AGENT_TYPE: dict[AgentRole, AgentType] = {
    AgentRole.MARKET_STRUCTURE: AgentType.MARKET_INTELLIGENCE,
    AgentRole.SMC_ANALYST: AgentType.STRATEGY,
    AgentRole.CMT_TECHNICIAN: AgentType.CMT_ANALYST,
    AgentRole.RISK_MANAGER: AgentType.RISK_GATE,
    AgentRole.EXECUTION: AgentType.EXECUTION,
    AgentRole.ORCHESTRATOR: AgentType.CUSTOM,
    AgentRole.BROADCAST: AgentType.CUSTOM,
}


def role_to_agent_type(role: AgentRole) -> AgentType:
    """Map a reasoning role to the corresponding infrastructure AgentType."""
    return _ROLE_TO_AGENT_TYPE.get(role, AgentType.CUSTOM)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------


class AgentMessage(BaseModel):
    """A single message in an inter-agent reasoning conversation.

    Attributes
    ----------
    message_id:
        Unique ID for this message.
    conversation_id:
        Groups messages into a conversation thread.
    sender:
        The role of the agent sending this message.
    recipients:
        Target roles. Empty list = broadcast to all.
    message_type:
        Classification of message content.
    content:
        Free-text reasoning / explanation.
    structured_data:
        Machine-readable payload (signal params, risk limits, etc.).
    confidence:
        Sender's confidence in the message content (0.0–1.0).
    references:
        IDs of prior messages this is responding to.
    timestamp:
        UTC timestamp of message creation.
    metadata:
        Arbitrary extra data (strategy_id, symbol, timeframe, etc.).
    """

    message_id: str = Field(default_factory=_uuid)
    conversation_id: str = ""
    sender: AgentRole
    recipients: list[AgentRole] = Field(default_factory=list)
    message_type: MessageType
    content: str = ""
    structured_data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    references: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_broadcast(self) -> bool:
        """True if this message targets all agents."""
        return len(self.recipients) == 0

    @property
    def is_veto(self) -> bool:
        """True if this is a veto message."""
        return self.message_type == MessageType.VETO

    @property
    def is_challenge(self) -> bool:
        """True if this is a challenge/disagreement."""
        return self.message_type in (
            MessageType.CHALLENGE,
            MessageType.DISAGREEMENT,
        )

    def short_summary(self) -> str:
        """One-line summary for logging and display."""
        recip = (
            "ALL"
            if self.is_broadcast
            else ",".join(r.value for r in self.recipients)
        )
        conf = f" [{self.confidence:.0%}]" if self.confidence > 0 else ""
        preview = self.content[:80]
        if len(self.content) > 80:
            preview += "..."
        return (
            f"[{self.sender.display_name} → {recip}] "
            f"{self.message_type.value}{conf}: {preview}"
        )
