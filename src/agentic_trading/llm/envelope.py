"""LLM Interaction Envelope — mandatory contract for every LLM call.

Provider-agnostic models that define what goes into an LLM call
(instructions, context, evidence, budget, safety constraints) and
what comes out (raw output, parsed output, validation results).

The envelope does **not** execute calls.  Execution remains with
the caller.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.ids import (
    content_hash as _content_hash,
)
from agentic_trading.core.ids import (
    new_id as _uuid,
)
from agentic_trading.core.ids import (
    payload_hash as _payload_hash,
)
from agentic_trading.core.ids import (
    utc_now as _now,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"


class ResponseFormat(str, Enum):
    """Expected response format from the LLM."""

    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"


class EnvelopeWorkflow(str, Enum):
    """Classification of the LLM interaction purpose."""

    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class RetryPolicy(BaseModel):
    """Retry rules for LLM calls."""

    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    retryable_errors: list[str] = Field(
        default_factory=lambda: ["rate_limit", "timeout", "server_error"],
    )


class LLMBudget(BaseModel):
    """Token and cost budget for a single LLM call."""

    max_tokens: int = 4096
    max_input_tokens: int | None = None
    max_output_tokens: int = 4096
    max_cost_usd: float | None = None
    thinking_budget_tokens: int | None = None


class SafetyConstraints(BaseModel):
    """Safety guardrails applied to the LLM call."""

    banned_topics: list[str] = Field(default_factory=list)
    max_output_length: int = 50_000
    require_json_output: bool = False
    pii_filter: bool = True
    require_deterministic: bool = False


class EvidenceItem(BaseModel):
    """A piece of retrieved evidence supplied to the LLM."""

    source: str
    content: dict[str, Any] = Field(default_factory=dict)
    relevance: float = 1.0
    retrieved_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# LLMEnvelope — the main contract
# ---------------------------------------------------------------------------


class LLMEnvelope(BaseModel):
    """Mandatory contract for every LLM interaction.

    Captures instructions, context, evidence, budget, safety
    constraints, and output schema in one auditable unit.
    """

    # Identity
    envelope_id: str = Field(default_factory=_uuid)
    trace_id: str = Field(default_factory=_uuid)
    causation_id: str = ""
    tenant_id: str = "default"
    created_at: datetime = Field(default_factory=_now)

    # Workflow classification
    workflow: EnvelopeWorkflow = EnvelopeWorkflow.GENERAL
    agent_id: str = ""
    agent_type: str = ""

    # --- Mandatory envelope fields ---
    instructions: str
    context: dict[str, Any] = Field(default_factory=dict)
    retrieved_evidence: list[EvidenceItem] = Field(default_factory=list)
    tools_allowed: list[str] = Field(default_factory=list)
    budget: LLMBudget = Field(default_factory=LLMBudget)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    safety_constraints: SafetyConstraints = Field(
        default_factory=SafetyConstraints,
    )
    response_format: ResponseFormat = ResponseFormat.JSON

    # Execution rules
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = ""
    temperature: float = 0.0
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)

    # Computed integrity hash
    envelope_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.envelope_hash:
            self.envelope_hash = _payload_hash(
                {
                    "instructions": self.instructions,
                    "context": self.context,
                    "evidence": [
                        e.model_dump(mode="json")
                        for e in self.retrieved_evidence
                    ],
                    "model": self.model,
                    "workflow": self.workflow.value,
                },
            )


# ---------------------------------------------------------------------------
# LLMResult — persisted output
# ---------------------------------------------------------------------------


class LLMResult(BaseModel):
    """Captured output from an LLM call."""

    result_id: str = Field(default_factory=_uuid)
    envelope_id: str
    timestamp: datetime = Field(default_factory=_now)

    # Raw + parsed output
    raw_output: str = ""
    parsed_output: dict[str, Any] = Field(default_factory=dict)

    # Validation
    validation_passed: bool = False
    validation_errors: list[str] = Field(default_factory=list)

    # Metrics
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = ""
    attempt_number: int = 1

    # Error
    error: str | None = None
    success: bool = True

    # Integrity
    output_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.output_hash and self.raw_output:
            self.output_hash = _content_hash(self.raw_output)


# ---------------------------------------------------------------------------
# LLMInteraction — storage unit (envelope + result)
# ---------------------------------------------------------------------------


class LLMInteraction(BaseModel):
    """Complete LLM interaction: envelope + result, ready for persistence."""

    interaction_id: str = Field(default_factory=_uuid)
    envelope: LLMEnvelope
    result: LLMResult
    stored_at: datetime = Field(default_factory=_now)
