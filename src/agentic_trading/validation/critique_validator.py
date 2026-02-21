"""Layer 4: Second-model critique (cost-gated).

This layer is ONLY triggered when specific conditions are met:

1. The output's notional impact exceeds a threshold, OR
2. The overall confidence from earlier layers is low, OR
3. A policy flag requests it (e.g., new strategy in L0 shadow), OR
4. The output type is in the always-critique list.

When triggered, it delegates to an injected LLM callable to get
an independent critique of the output.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.llm.envelope import LLMEnvelope, LLMResult

from .models import (
    CritiqueResult,
    ValidationIssue,
    ValidationLayer,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class CritiqueTriggerConfig(BaseModel):
    """Configuration for when to trigger second-model critique."""

    notional_usd_threshold: float = 50_000.0
    confidence_floor: float = 0.5
    always_critique_types: list[str] = Field(
        default_factory=lambda: ["CMTAssessmentResponse"]
    )
    max_cost_usd: float = 0.10
    critique_model: str = "claude-haiku-4-5-20250929"
    acceptance_threshold: float = 0.6
    enabled: bool = True


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class CritiqueValidator:
    """Conditionally invokes a second LLM to critique agent output.

    Parameters
    ----------
    trigger_config:
        When to trigger critique and budget limits.
    call_llm:
        Callable that takes (instructions, context, model) and returns
        a dict with ``overall_score``, ``issues``, ``reasoning``.
        Injected to keep the validator decoupled from the LLM client.
    """

    def __init__(
        self,
        trigger_config: CritiqueTriggerConfig | None = None,
        call_llm: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._config = trigger_config or CritiqueTriggerConfig()
        self._call_llm = call_llm

    @property
    def layer_name(self) -> str:
        return "critique"

    def should_trigger(
        self,
        parsed_output: dict[str, Any],
        envelope: LLMEnvelope,
        prior_quality_score: float,
    ) -> tuple[bool, str]:
        """Decide whether to trigger critique.

        Returns
        -------
        tuple[bool, str]
            (should_trigger, reason)
        """
        if not self._config.enabled:
            return False, "disabled"

        output_type = parsed_output.get("_output_type", "")

        if output_type in self._config.always_critique_types:
            return True, f"always_critique:{output_type}"

        notional = envelope.context.get("notional_usd", 0.0)
        if isinstance(notional, (int, float)) and notional >= self._config.notional_usd_threshold:
            return True, f"high_notional:{notional}"

        if prior_quality_score < self._config.confidence_floor:
            return True, f"low_confidence:{prior_quality_score:.2f}"

        return False, "not_triggered"

    def validate(
        self,
        parsed_output: dict[str, Any],
        envelope: LLMEnvelope,
        result: LLMResult,
        *,
        prior_quality_score: float = 1.0,
    ) -> list[ValidationIssue]:
        """Run critique if triggered.  Returns issues found."""
        should, reason = self.should_trigger(
            parsed_output, envelope, prior_quality_score
        )
        if not should:
            return []

        if self._call_llm is None:
            logger.warning(
                "Critique triggered (%s) but no LLM callable configured",
                reason,
            )
            return []

        logger.info("Critique triggered: reason=%s", reason)
        t_start = time.monotonic()

        try:
            critique = self._run_critique(
                parsed_output, envelope, reason
            )
            latency = (time.monotonic() - t_start) * 1000
            critique.latency_ms = latency

            issues: list[ValidationIssue] = []
            if critique.overall_score < self._config.acceptance_threshold:
                issues.append(
                    ValidationIssue(
                        layer=ValidationLayer.CRITIQUE,
                        severity=ValidationSeverity.ERROR,
                        code="critique.low_score",
                        message=(
                            f"Second-model critique score "
                            f"{critique.overall_score:.2f} below threshold "
                            f"{self._config.acceptance_threshold}"
                        ),
                        metadata={
                            "critique_id": critique.critique_id,
                            "reasoning": critique.reasoning[:500],
                        },
                    )
                )
            issues.extend(critique.issues_found)
            return issues

        except Exception:
            logger.exception("Critique call failed")
            return [
                ValidationIssue(
                    layer=ValidationLayer.CRITIQUE,
                    severity=ValidationSeverity.WARNING,
                    code="critique.call_failed",
                    message=(
                        "Second-model critique call failed; "
                        "proceeding without critique"
                    ),
                )
            ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_critique(
        self,
        parsed_output: dict[str, Any],
        envelope: LLMEnvelope,
        trigger_reason: str,
    ) -> CritiqueResult:
        """Construct and execute the critique call."""
        assert self._call_llm is not None  # noqa: S101

        instructions = (
            "You are a validation critic.  Review the following agent "
            "output and its supporting evidence.  Identify:\n"
            "1. Logical errors or contradictions\n"
            "2. Claims not supported by the provided evidence\n"
            "3. Unjustified confidence or conviction levels\n"
            "4. Missing analysis for key factors\n\n"
            "Respond with JSON: {\"overall_score\": 0.0-1.0, "
            "\"issues\": [{\"code\": \"...\", \"message\": \"...\", "
            "\"severity\": \"error|warning\"}], "
            "\"reasoning\": \"...\"}"
        )

        context = {
            "agent_output": parsed_output,
            "evidence_sources": [
                {"source": e.source, "relevance": e.relevance}
                for e in envelope.retrieved_evidence
            ],
            "trigger_reason": trigger_reason,
        }

        raw = self._call_llm(
            instructions=instructions,
            context=context,
            model=self._config.critique_model,
        )

        # Parse response
        score = float(raw.get("overall_score", 0.0))
        reasoning = str(raw.get("reasoning", ""))

        issues_found: list[ValidationIssue] = []
        for issue_data in raw.get("issues", []):
            sev_str = issue_data.get("severity", "warning")
            try:
                sev = ValidationSeverity(sev_str)
            except ValueError:
                sev = ValidationSeverity.WARNING
            issues_found.append(
                ValidationIssue(
                    layer=ValidationLayer.CRITIQUE,
                    severity=sev,
                    code=issue_data.get("code", "critique.issue"),
                    message=issue_data.get("message", ""),
                )
            )

        return CritiqueResult(
            model_used=self._config.critique_model,
            triggered_by=trigger_reason,
            overall_score=score,
            issues_found=issues_found,
            reasoning=reasoning,
        )
