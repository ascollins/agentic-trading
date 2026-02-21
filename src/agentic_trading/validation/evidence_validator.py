"""Layer 2: Evidence validation — claim-to-evidence linking.

Every factual claim in agent output must either:

1. Cite a specific evidence source (from ``envelope.retrieved_evidence``)
2. Be explicitly marked as "assumption"
3. Be derived from cited data (e.g., computed indicators)

Claims that fail all three are flagged as UNCITED.  If uncited claims
exceed a configurable threshold, the output is rejected.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.llm.envelope import LLMEnvelope, LLMResult

from .models import (
    ClaimAnnotation,
    ClaimType,
    ValidationIssue,
    ValidationLayer,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)

# Fields that are considered factual claims requiring evidence.
_DEFAULT_CLAIM_FIELDS: dict[str, list[str]] = {
    "CMTAssessmentResponse": [
        "thesis",
        "trade_plan.entry_price",
        "trade_plan.stop_loss",
        "trade_plan.targets",
        "layers.*.key_findings",
        "confluence.total",
    ],
    "Signal": [
        "rationale",
        "stop_loss",
        "take_profit",
        "confidence",
    ],
}


class EvidenceValidator:
    """Validates that factual claims cite retrieved evidence.

    Parameters
    ----------
    max_uncited_ratio:
        Maximum fraction of claims that can be uncited before
        hard fail.  Default 0.3 (30%).
    claim_fields:
        Per-output-type mapping of field paths to validate.
        Uses defaults if not provided.
    """

    def __init__(
        self,
        *,
        max_uncited_ratio: float = 0.3,
        claim_fields: dict[str, list[str]] | None = None,
    ) -> None:
        self._max_uncited_ratio = max_uncited_ratio
        self._claim_fields: dict[str, list[str]] = (
            claim_fields if claim_fields is not None else dict(_DEFAULT_CLAIM_FIELDS)
        )

    @property
    def layer_name(self) -> str:
        return "evidence"

    def validate(
        self,
        parsed_output: dict[str, Any],
        envelope: LLMEnvelope,
        result: LLMResult,
    ) -> list[ValidationIssue]:
        """Run evidence validation.  Returns issues (empty = clean)."""
        issues: list[ValidationIssue] = []

        # Build set of available evidence sources
        available_sources = {e.source for e in envelope.retrieved_evidence}

        # Extract claims from the output
        output_type = parsed_output.get("_output_type", "")
        claims = self.extract_claims(
            parsed_output, output_type, available_sources
        )

        # Check uncited ratio
        if claims:
            uncited = [
                c for c in claims if c.claim_type == ClaimType.UNCITED
            ]
            uncited_ratio = len(uncited) / len(claims)

            for claim in uncited:
                issues.append(
                    ValidationIssue(
                        layer=ValidationLayer.EVIDENCE,
                        severity=ValidationSeverity.WARNING,
                        code="evidence.uncited_claim",
                        message=(
                            f"Claim at '{claim.field_path}' "
                            f"not backed by evidence"
                        ),
                        field_path=claim.field_path,
                        actual=claim.claim_text[:200],
                        metadata={"claim_id": claim.claim_id},
                    )
                )

            if uncited_ratio > self._max_uncited_ratio:
                issues.append(
                    ValidationIssue(
                        layer=ValidationLayer.EVIDENCE,
                        severity=ValidationSeverity.ERROR,
                        code="evidence.uncited_threshold_exceeded",
                        message=(
                            f"Uncited claim ratio {uncited_ratio:.0%} exceeds "
                            f"threshold {self._max_uncited_ratio:.0%} "
                            f"({len(uncited)}/{len(claims)} claims)"
                        ),
                        metadata={
                            "uncited_count": len(uncited),
                            "total_claims": len(claims),
                            "uncited_ratio": uncited_ratio,
                        },
                    )
                )

        # Check for evidence that was provided but never referenced
        if envelope.retrieved_evidence and claims:
            cited_sources: set[str] = set()
            for claim in claims:
                cited_sources.update(claim.evidence_ids)
            unused = available_sources - cited_sources
            if unused:
                issues.append(
                    ValidationIssue(
                        layer=ValidationLayer.EVIDENCE,
                        severity=ValidationSeverity.INFO,
                        code="evidence.unused_evidence",
                        message=(
                            f"Evidence provided but not cited: {unused}"
                        ),
                        metadata={"unused_sources": sorted(unused)},
                    )
                )

        return issues

    def extract_claims(
        self,
        parsed_output: dict[str, Any],
        output_type: str,
        available_sources: set[str],
    ) -> list[ClaimAnnotation]:
        """Extract and annotate factual claims from parsed output.

        Uses heuristics to classify claims:

        1. If the output includes ``evidence_refs`` mapping
           field → source_id, those are CITED.
        2. If a field value contains "assumption" or "estimated",
           it is marked ASSUMPTION.
        3. Numeric fields are marked DERIVED.
        4. String content > 10 chars with no link is UNCITED.
        """
        claims: list[ClaimAnnotation] = []

        evidence_refs: dict[str, Any] = parsed_output.get(
            "evidence_refs", {}
        )
        assumptions: set[str] = set(
            parsed_output.get("assumptions", [])
        )

        field_paths = self._claim_fields.get(output_type, [])
        if not field_paths:
            # Fall back: scan top-level string/numeric fields
            field_paths = [
                k
                for k, v in parsed_output.items()
                if k not in ("_output_type", "evidence_refs", "assumptions")
                and isinstance(v, (str, int, float))
            ]

        for path in field_paths:
            value = self._resolve_path(parsed_output, path)
            if value is None:
                continue

            # Handle list values from wildcard resolution
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    claim = self._classify_claim(
                        f"{path}[{idx}]",
                        item,
                        evidence_refs,
                        assumptions,
                    )
                    if claim is not None:
                        claims.append(claim)
                continue

            claim = self._classify_claim(
                path, value, evidence_refs, assumptions
            )
            if claim is not None:
                claims.append(claim)

        return claims

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_claim(
        path: str,
        value: Any,
        evidence_refs: dict[str, Any],
        assumptions: set[str],
    ) -> ClaimAnnotation | None:
        """Classify a single field value as a claim."""
        claim_text = str(value)[:500]

        if path in evidence_refs:
            ref = evidence_refs[path]
            evidence_ids = ref if isinstance(ref, list) else [ref]
            return ClaimAnnotation(
                field_path=path,
                claim_text=claim_text,
                claim_type=ClaimType.CITED,
                evidence_ids=evidence_ids,
            )

        if path in assumptions or "assumption" in claim_text.lower():
            return ClaimAnnotation(
                field_path=path,
                claim_text=claim_text,
                claim_type=ClaimType.ASSUMPTION,
            )

        if isinstance(value, (int, float)):
            return ClaimAnnotation(
                field_path=path,
                claim_text=claim_text,
                claim_type=ClaimType.DERIVED,
            )

        if isinstance(value, str) and len(value) > 10:
            return ClaimAnnotation(
                field_path=path,
                claim_text=claim_text,
                claim_type=ClaimType.UNCITED,
            )

        return None

    @staticmethod
    def _resolve_path(data: dict[str, Any], path: str) -> Any:
        """Resolve a dot-path from nested dict.  Supports ``*`` wildcard."""
        if "*" in path:
            parts = path.split(".")
            idx = parts.index("*")
            prefix = ".".join(parts[:idx])
            suffix = ".".join(parts[idx + 1 :])
            parent = (
                EvidenceValidator._resolve_path(data, prefix)
                if prefix
                else data
            )
            if isinstance(parent, (list, dict)):
                items = (
                    parent.values()
                    if isinstance(parent, dict)
                    else parent
                )
                results = []
                for item in items:
                    if suffix and isinstance(item, dict):
                        sub = EvidenceValidator._resolve_path(item, suffix)
                        if sub is not None:
                            results.append(sub)
                    elif not suffix:
                        results.append(item)
                return results if results else None
            return None

        current: Any = data
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current
