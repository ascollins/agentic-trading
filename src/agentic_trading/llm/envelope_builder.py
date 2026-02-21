"""Fluent builder for LLMEnvelope instances.

Provides workflow presets (analysis, planning, execution) with
sensible defaults, and chainable setters for customisation.

Usage::

    from agentic_trading.llm import EnvelopeBuilder, LLMProvider

    envelope = (
        EnvelopeBuilder()
        .for_analysis()
        .with_instructions("Analyse the 9-layer CMT framework...")
        .with_context({"symbol": "BTCUSDT"})
        .add_evidence("candle_history", {"1h": {...}})
        .with_provider(LLMProvider.ANTHROPIC, "claude-sonnet-4-5-20250929")
        .build()
    )
"""

from __future__ import annotations

from typing import Any

from agentic_trading.llm.envelope import (
    EnvelopeWorkflow,
    EvidenceItem,
    LLMBudget,
    LLMEnvelope,
    LLMProvider,
    ResponseFormat,
    RetryPolicy,
    SafetyConstraints,
)


class EnvelopeBuilder:
    """Fluent builder for :class:`LLMEnvelope`."""

    def __init__(self) -> None:
        self._workflow = EnvelopeWorkflow.GENERAL
        self._instructions: str = ""
        self._context: dict[str, Any] = {}
        self._evidence: list[EvidenceItem] = []
        self._tools_allowed: list[str] = []
        self._budget = LLMBudget()
        self._output_schema: dict[str, Any] = {}
        self._safety = SafetyConstraints()
        self._response_format = ResponseFormat.JSON
        self._provider = LLMProvider.ANTHROPIC
        self._model: str = ""
        self._temperature: float = 0.0
        self._retry = RetryPolicy()
        self._trace_id: str = ""
        self._causation_id: str = ""
        self._agent_id: str = ""
        self._agent_type: str = ""
        self._tenant_id: str = "default"

    # -- workflow presets ----------------------------------------------------

    def for_analysis(self) -> EnvelopeBuilder:
        """Preset for analytical workflows (CMT, regime detection)."""
        self._workflow = EnvelopeWorkflow.ANALYSIS
        self._temperature = 0.0
        self._budget = LLMBudget(
            max_output_tokens=4096,
            thinking_budget_tokens=8000,
        )
        self._safety = SafetyConstraints(
            require_json_output=True,
        )
        self._response_format = ResponseFormat.JSON
        return self

    def for_planning(self) -> EnvelopeBuilder:
        """Preset for planning workflows (parameter optimisation)."""
        self._workflow = EnvelopeWorkflow.PLANNING
        self._temperature = 0.0
        self._budget = LLMBudget(
            max_output_tokens=8192,
        )
        self._safety = SafetyConstraints(
            require_json_output=True,
        )
        self._response_format = ResponseFormat.JSON
        return self

    def for_execution(self) -> EnvelopeBuilder:
        """Preset for execution workflows (fill strategy, sizing).

        Strictest preset: deterministic mode forced, smallest budget.
        """
        self._workflow = EnvelopeWorkflow.EXECUTION
        self._temperature = 0.0
        self._budget = LLMBudget(
            max_output_tokens=2048,
        )
        self._safety = SafetyConstraints(
            require_json_output=True,
            require_deterministic=True,
        )
        self._response_format = ResponseFormat.JSON
        return self

    # -- individual setters -------------------------------------------------

    def with_instructions(self, instructions: str) -> EnvelopeBuilder:
        self._instructions = instructions
        return self

    def with_context(self, context: dict[str, Any]) -> EnvelopeBuilder:
        self._context = context
        return self

    def add_evidence(
        self,
        source: str,
        content: dict[str, Any],
        relevance: float = 1.0,
    ) -> EnvelopeBuilder:
        self._evidence.append(
            EvidenceItem(source=source, content=content, relevance=relevance),
        )
        return self

    def allow_tools(self, tools: list[str]) -> EnvelopeBuilder:
        self._tools_allowed = tools
        return self

    def with_budget(
        self,
        *,
        max_output_tokens: int = 4096,
        thinking_budget: int | None = None,
        max_cost_usd: float | None = None,
    ) -> EnvelopeBuilder:
        self._budget = LLMBudget(
            max_output_tokens=max_output_tokens,
            thinking_budget_tokens=thinking_budget,
            max_cost_usd=max_cost_usd,
        )
        return self

    def with_output_schema(
        self,
        schema: dict[str, Any],
    ) -> EnvelopeBuilder:
        self._output_schema = schema
        return self

    def with_safety(
        self,
        *,
        require_json: bool = False,
        pii_filter: bool = True,
        deterministic: bool = False,
    ) -> EnvelopeBuilder:
        self._safety = SafetyConstraints(
            require_json_output=require_json,
            pii_filter=pii_filter,
            require_deterministic=deterministic,
        )
        return self

    def with_retry(
        self,
        *,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> EnvelopeBuilder:
        self._retry = RetryPolicy(
            max_retries=max_retries,
            backoff_base_seconds=backoff_base,
        )
        return self

    def with_provider(
        self,
        provider: LLMProvider,
        model: str,
    ) -> EnvelopeBuilder:
        self._provider = provider
        self._model = model
        return self

    def with_temperature(self, temperature: float) -> EnvelopeBuilder:
        self._temperature = temperature
        return self

    def with_trace(
        self,
        trace_id: str,
        causation_id: str = "",
    ) -> EnvelopeBuilder:
        self._trace_id = trace_id
        self._causation_id = causation_id
        return self

    def with_agent(
        self,
        agent_id: str,
        agent_type: str = "",
    ) -> EnvelopeBuilder:
        self._agent_id = agent_id
        self._agent_type = agent_type
        return self

    def deterministic(self) -> EnvelopeBuilder:
        """Sugar: set temperature=0 and require_deterministic=True."""
        self._temperature = 0.0
        self._safety = self._safety.model_copy(
            update={"require_deterministic": True},
        )
        return self

    # -- build --------------------------------------------------------------

    def build(self) -> LLMEnvelope:
        """Construct and validate the :class:`LLMEnvelope`.

        Raises
        ------
        EnvelopeValidationError
            If required fields are missing or constraints are violated.
        """
        from agentic_trading.llm.errors import EnvelopeValidationError

        if not self._instructions:
            raise EnvelopeValidationError("instructions must not be empty")

        if (
            self._safety.require_deterministic
            and self._temperature != 0.0
        ):
            raise EnvelopeValidationError(
                "temperature must be 0.0 when require_deterministic is True "
                f"(got {self._temperature})"
            )

        kwargs: dict[str, Any] = {
            "workflow": self._workflow,
            "instructions": self._instructions,
            "context": self._context,
            "retrieved_evidence": self._evidence,
            "tools_allowed": self._tools_allowed,
            "budget": self._budget,
            "expected_output_schema": self._output_schema,
            "safety_constraints": self._safety,
            "response_format": self._response_format,
            "provider": self._provider,
            "model": self._model,
            "temperature": self._temperature,
            "retry_policy": self._retry,
            "agent_id": self._agent_id,
            "agent_type": self._agent_type,
            "tenant_id": self._tenant_id,
        }

        if self._trace_id:
            kwargs["trace_id"] = self._trace_id
        if self._causation_id:
            kwargs["causation_id"] = self._causation_id

        return LLMEnvelope(**kwargs)
