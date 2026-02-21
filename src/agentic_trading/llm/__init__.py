"""LLM Interaction Envelope — mandatory contract for every LLM call.

Public API
----------
Models:
    LLMEnvelope, LLMResult, LLMInteraction,
    LLMBudget, SafetyConstraints, EvidenceItem, RetryPolicy

Enums:
    LLMProvider, ResponseFormat, EnvelopeWorkflow

Builder:
    EnvelopeBuilder

Storage:
    IInteractionStore, MemoryInteractionStore, JsonFileInteractionStore

Errors:
    LLMError, EnvelopeValidationError,
    LLMBudgetExhaustedError, LLMResponseValidationError
"""

from agentic_trading.llm.envelope import (
    EnvelopeWorkflow,
    EvidenceItem,
    LLMBudget,
    LLMEnvelope,
    LLMInteraction,
    LLMProvider,
    LLMResult,
    ResponseFormat,
    RetryPolicy,
    SafetyConstraints,
)
from agentic_trading.llm.envelope_builder import EnvelopeBuilder
from agentic_trading.llm.errors import (
    EnvelopeValidationError,
    LLMBudgetExhaustedError,
    LLMError,
    LLMResponseValidationError,
)
from agentic_trading.llm.store import (
    IInteractionStore,
    JsonFileInteractionStore,
    MemoryInteractionStore,
)

__all__ = [
    # Enums
    "EnvelopeWorkflow",
    "LLMProvider",
    "ResponseFormat",
    # Models
    "EvidenceItem",
    "LLMBudget",
    "LLMEnvelope",
    "LLMInteraction",
    "LLMResult",
    "RetryPolicy",
    "SafetyConstraints",
    # Builder
    "EnvelopeBuilder",
    # Storage
    "IInteractionStore",
    "JsonFileInteractionStore",
    "MemoryInteractionStore",
    # Errors
    "EnvelopeValidationError",
    "LLMBudgetExhaustedError",
    "LLMError",
    "LLMResponseValidationError",
]
