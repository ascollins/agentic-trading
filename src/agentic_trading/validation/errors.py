"""Validation-specific error types.

All inherit from TradingError via ValidationError.
"""

from __future__ import annotations

from agentic_trading.core.errors import TradingError


class ValidationError(TradingError):
    """Base for all validation framework errors."""


class SchemaValidationError(ValidationError):
    """Output failed structural schema validation (hard fail)."""


class EvidenceValidationError(ValidationError):
    """Too many uncited claims exceeded the threshold."""


class BusinessRuleValidationError(ValidationError):
    """Business rule invariant violated."""


class CritiqueValidationError(ValidationError):
    """Second-model critique score below acceptance threshold."""


class RemediationExhaustedError(ValidationError):
    """All remediation attempts exhausted without resolution."""
