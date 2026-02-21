"""Tests for Layer 2: EvidenceValidator."""

from __future__ import annotations

from agentic_trading.llm.envelope import EvidenceItem, LLMEnvelope, LLMResult
from agentic_trading.validation.evidence_validator import EvidenceValidator
from agentic_trading.validation.models import (
    ClaimType,
    ValidationSeverity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    evidence: list[EvidenceItem] | None = None,
) -> LLMEnvelope:
    """Create a minimal envelope with evidence."""
    return LLMEnvelope(
        instructions="test",
        retrieved_evidence=evidence or [],
    )


def _make_result() -> LLMResult:
    return LLMResult(envelope_id="test-envelope")


def _make_evidence(source: str) -> EvidenceItem:
    return EvidenceItem(
        source=source,
        content={"data": "value"},
        relevance=0.9,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvidenceValidatorLayerName:
    def test_layer_name(self) -> None:
        v = EvidenceValidator()
        assert v.layer_name == "evidence"


class TestClaimExtraction:
    def test_cited_claim_via_evidence_refs(self) -> None:
        v = EvidenceValidator()
        output = {
            "_output_type": "Signal",
            "rationale": "BTC is breaking out above key resistance",
            "confidence": 0.85,
            "evidence_refs": {
                "rationale": ["candle_history"],
            },
        }
        claims = v.extract_claims(output, "Signal", {"candle_history"})
        rationale_claims = [c for c in claims if c.field_path == "rationale"]
        assert len(rationale_claims) == 1
        assert rationale_claims[0].claim_type == ClaimType.CITED
        assert "candle_history" in rationale_claims[0].evidence_ids

    def test_assumption_detected(self) -> None:
        v = EvidenceValidator()
        output = {
            "_output_type": "Signal",
            "rationale": "This is an assumption about market direction",
            "confidence": 0.5,
        }
        claims = v.extract_claims(output, "Signal", set())
        rationale_claims = [c for c in claims if c.field_path == "rationale"]
        assert len(rationale_claims) == 1
        assert rationale_claims[0].claim_type == ClaimType.ASSUMPTION

    def test_assumption_via_assumptions_list(self) -> None:
        v = EvidenceValidator()
        output = {
            "_output_type": "Signal",
            "rationale": "Market will go up",
            "assumptions": ["rationale"],
            "confidence": 0.5,
        }
        claims = v.extract_claims(output, "Signal", set())
        rationale_claims = [c for c in claims if c.field_path == "rationale"]
        assert len(rationale_claims) == 1
        assert rationale_claims[0].claim_type == ClaimType.ASSUMPTION

    def test_numeric_fields_classified_as_derived(self) -> None:
        v = EvidenceValidator()
        output = {
            "_output_type": "Signal",
            "confidence": 0.85,
            "rationale": "test short",
        }
        claims = v.extract_claims(output, "Signal", set())
        conf_claims = [c for c in claims if c.field_path == "confidence"]
        assert len(conf_claims) == 1
        assert conf_claims[0].claim_type == ClaimType.DERIVED

    def test_uncited_string_claim(self) -> None:
        v = EvidenceValidator()
        output = {
            "_output_type": "Signal",
            "rationale": "BTC is breaking above the 200 EMA resistance with strong volume",
            "confidence": 0.85,
        }
        claims = v.extract_claims(output, "Signal", set())
        rationale_claims = [c for c in claims if c.field_path == "rationale"]
        assert len(rationale_claims) == 1
        assert rationale_claims[0].claim_type == ClaimType.UNCITED

    def test_short_strings_skipped(self) -> None:
        v = EvidenceValidator()
        output = {
            "_output_type": "Signal",
            "rationale": "short",  # <= 10 chars
            "confidence": 0.5,
        }
        claims = v.extract_claims(output, "Signal", set())
        rationale_claims = [c for c in claims if c.field_path == "rationale"]
        assert len(rationale_claims) == 0

    def test_fallback_field_scan_for_unknown_type(self) -> None:
        v = EvidenceValidator()
        output = {
            "_output_type": "UnknownType",
            "description": "This is a long description that should be scanned",
            "count": 42,
        }
        claims = v.extract_claims(output, "UnknownType", set())
        assert len(claims) >= 1  # At least the description


class TestEvidenceValidatorValidation:
    def test_all_cited_passes(self) -> None:
        v = EvidenceValidator()
        evidence = [_make_evidence("candle_history")]
        envelope = _make_envelope(evidence)
        output = {
            "_output_type": "Signal",
            "rationale": "BTC is breaking out above key resistance level",
            "evidence_refs": {"rationale": ["candle_history"]},
            "confidence": 0.85,
        }
        issues = v.validate(output, envelope, _make_result())
        errors = [
            i
            for i in issues
            if i.severity
            in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        ]
        assert len(errors) == 0

    def test_uncited_threshold_exceeded(self) -> None:
        v = EvidenceValidator(max_uncited_ratio=0.3)
        envelope = _make_envelope()
        output = {
            "_output_type": "Signal",
            "rationale": "BTC is breaking out with strong volume confirmation now",
            "stop_loss": "41000 based on support level analysis confirmed",
            "take_profit": "48000 based on resistance level projection here",
            "confidence": 0.85,
        }
        issues = v.validate(output, envelope, _make_result())
        threshold_issues = [
            i
            for i in issues
            if i.code == "evidence.uncited_threshold_exceeded"
        ]
        assert len(threshold_issues) == 1
        assert threshold_issues[0].severity == ValidationSeverity.ERROR

    def test_individual_uncited_warnings(self) -> None:
        v = EvidenceValidator()
        envelope = _make_envelope()
        output = {
            "_output_type": "Signal",
            "rationale": "BTC is breaking out above resistance level now",
            "confidence": 0.85,
        }
        issues = v.validate(output, envelope, _make_result())
        uncited_warnings = [
            i for i in issues if i.code == "evidence.uncited_claim"
        ]
        assert len(uncited_warnings) >= 1
        assert all(
            i.severity == ValidationSeverity.WARNING for i in uncited_warnings
        )

    def test_unused_evidence_flagged(self) -> None:
        v = EvidenceValidator()
        evidence = [
            _make_evidence("candle_history"),
            _make_evidence("indicator_values"),
        ]
        envelope = _make_envelope(evidence)
        output = {
            "_output_type": "Signal",
            "rationale": "BTC is bullish based on the analysis",
            "evidence_refs": {"rationale": ["candle_history"]},
            "confidence": 0.85,
        }
        issues = v.validate(output, envelope, _make_result())
        unused = [
            i for i in issues if i.code == "evidence.unused_evidence"
        ]
        assert len(unused) == 1
        assert unused[0].severity == ValidationSeverity.INFO

    def test_no_claims_no_issues(self) -> None:
        v = EvidenceValidator()
        envelope = _make_envelope()
        output = {
            "_output_type": "Signal",
            "rationale": "short",
            "confidence": 0.5,
        }
        issues = v.validate(output, envelope, _make_result())
        # Short string skipped, confidence is DERIVED — no uncited claims
        uncited = [
            i for i in issues if i.code == "evidence.uncited_claim"
        ]
        assert len(uncited) == 0


class TestResolvePathWildcard:
    def test_simple_path(self) -> None:
        data = {"a": {"b": {"c": 42}}}
        assert EvidenceValidator._resolve_path(data, "a.b.c") == 42

    def test_missing_path(self) -> None:
        data = {"a": {"b": 1}}
        assert EvidenceValidator._resolve_path(data, "a.x.y") is None

    def test_wildcard_on_dict(self) -> None:
        data = {
            "layers": {
                "l1": {"score": 5},
                "l2": {"score": 8},
            }
        }
        result = EvidenceValidator._resolve_path(data, "layers.*.score")
        assert isinstance(result, list)
        assert set(result) == {5, 8}

    def test_wildcard_on_list(self) -> None:
        data = {
            "layers": [
                {"score": 5},
                {"score": 8},
            ]
        }
        result = EvidenceValidator._resolve_path(data, "layers.*.score")
        assert isinstance(result, list)
        assert set(result) == {5, 8}
