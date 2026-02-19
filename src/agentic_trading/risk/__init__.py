"""Backward-compatibility shim — canonical code now lives in ``agentic_trading.execution.risk``.

Will be removed in PR 16.
"""

from agentic_trading.execution.risk import (  # noqa: F401
    AlertEngine,
    CircuitBreaker,
    CircuitBreakerManager,
    DrawdownMonitor,
    ExposureSnapshot,
    ExposureTracker,
    KillSwitch,
    PostTradeChecker,
    PreTradeChecker,
    RiskManager,
    RiskMetrics,
)

__all__ = [
    "AlertEngine",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "DrawdownMonitor",
    "ExposureSnapshot",
    "ExposureTracker",
    "KillSwitch",
    "PostTradeChecker",
    "PreTradeChecker",
    "RiskManager",
    "RiskMetrics",
]
