"""Backward-compatibility shim — see ``agentic_trading.execution.risk.circuit_breakers``.

Will be removed in PR 16.
"""

from agentic_trading.execution.risk.circuit_breakers import *  # noqa: F401, F403
from agentic_trading.execution.risk.circuit_breakers import (  # noqa: F811
    CircuitBreaker,
    CircuitBreakerManager,
)

__all__ = ["CircuitBreaker", "CircuitBreakerManager"]
