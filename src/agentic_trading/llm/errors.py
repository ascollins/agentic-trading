"""LLM-specific error types.

All inherit from :class:`TradingError` via :class:`LLMError`.
"""

from __future__ import annotations

from agentic_trading.core.errors import TradingError


class LLMError(TradingError):
    """Base for all LLM envelope errors."""


class EnvelopeValidationError(LLMError):
    """Envelope fails pre-call validation.

    Raised when mandatory fields are missing, constraints are violated
    (e.g. temperature > 0 with require_deterministic), or the output
    schema is malformed.
    """


class LLMBudgetExhaustedError(LLMError):
    """API call budget or cost ceiling exceeded.

    Raised when daily call limits, per-call cost caps, or token
    budgets are breached.
    """


class LLMResponseValidationError(LLMError):
    """LLM output failed validation against expected_output_schema.

    Raised when the parsed LLM response does not conform to the
    JSON Schema specified in ``LLMEnvelope.expected_output_schema``.
    """
