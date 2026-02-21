"""Validator protocol and pipeline composition utilities."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agentic_trading.llm.envelope import LLMEnvelope, LLMResult

from .models import ValidationIssue


@runtime_checkable
class IValidator(Protocol):
    """Protocol for a single validation layer.

    Validators are composable: the pipeline runs them in order.
    Each validator receives the parsed output and the envelope
    (which contains retrieved_evidence and expected_output_schema)
    and returns a list of issues.

    A validator does NOT decide pass/fail on its own — the pipeline
    aggregates issues and computes the overall disposition.
    """

    @property
    def layer_name(self) -> str:
        """Human-readable name of this validator layer."""
        ...

    def validate(
        self,
        parsed_output: dict[str, Any],
        envelope: LLMEnvelope,
        result: LLMResult,
    ) -> list[ValidationIssue]:
        """Run validation and return zero or more issues.

        Parameters
        ----------
        parsed_output:
            The parsed JSON output from the LLM.
        envelope:
            The originating LLMEnvelope (for evidence, schema, context).
        result:
            The LLMResult (for raw_output, metrics).

        Returns
        -------
        list[ValidationIssue]
            Issues found.  Empty list = layer passed cleanly.
        """
        ...
