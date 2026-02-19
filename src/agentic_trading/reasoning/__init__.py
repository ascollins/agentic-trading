"""Reasoning Chain — structured capture of agent thinking and pipeline results.

Every agent's reasoning is captured as a ReasoningTrace with phases:
perception -> hypothesis -> evaluation -> decision -> action.

Soteria extensions add inter-agent messaging, multi-turn conversations,
and desk-style reasoning capture.
"""

from __future__ import annotations

from .builder import ReasoningBuilder
from .chain import ReasoningChain
from .models import ReasoningStep, ReasoningTrace
from .pipeline_result import PipelineResult

# Soteria inter-agent reasoning
from .agent_message import AgentMessage, AgentRole, MessageType
from .soteria_trace import SoteriaStep, SoteriaTrace, StepType
from .agent_conversation import AgentConversation, ConversationOutcome
from .message_bus import ReasoningMessageBus
from .reasoning_mixin import ReasoningMixin
from .consensus import (
    CMTTechnicianDesk,
    ConsensusGate,
    ConsensusResult,
    ConsensusVerdict,
    DeskOpinion,
    DeskParticipant,
    MarketStructureDesk,
    RiskManagerDesk,
    SMCAnalystDesk,
)
from .conversation_store import InMemoryConversationStore, JsonFileConversationStore

__all__ = [
    # Original reasoning
    "PipelineResult",
    "ReasoningBuilder",
    "ReasoningChain",
    "ReasoningStep",
    "ReasoningTrace",
    # Soteria: messages
    "AgentMessage",
    "AgentRole",
    "MessageType",
    # Soteria: traces
    "SoteriaStep",
    "SoteriaTrace",
    "StepType",
    # Soteria: conversations
    "AgentConversation",
    "ConversationOutcome",
    # Soteria: bus
    "ReasoningMessageBus",
    # Soteria: mixin
    "ReasoningMixin",
    # Soteria: consensus
    "CMTTechnicianDesk",
    "ConsensusGate",
    "ConsensusResult",
    "ConsensusVerdict",
    "DeskOpinion",
    "DeskParticipant",
    "MarketStructureDesk",
    "RiskManagerDesk",
    "SMCAnalystDesk",
    # Soteria: persistence
    "InMemoryConversationStore",
    "JsonFileConversationStore",
]
