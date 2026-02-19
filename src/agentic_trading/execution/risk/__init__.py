"""Risk management subsystem — canonical location under execution layer.

Provides pre-trade checks, post-trade checks, circuit breakers, kill switch,
drawdown monitoring, exposure tracking, alerts, and VaR/ES computation.

Prior to this move the code lived under ``agentic_trading.risk``, which now
contains thin re-export shims for backward compatibility (removed in PR 16).

The :class:`RiskManager` is the single facade that orchestrates all
sub-systems and is injected into the :class:`ExecutionEngine`.
"""

from agentic_trading.execution.risk.alerts import AlertEngine
from agentic_trading.execution.risk.circuit_breakers import (
    CircuitBreaker,
    CircuitBreakerManager,
)
from agentic_trading.execution.risk.drawdown import DrawdownMonitor
from agentic_trading.execution.risk.exposure import ExposureSnapshot, ExposureTracker
from agentic_trading.execution.risk.kill_switch import KillSwitch
from agentic_trading.execution.risk.manager import RiskManager
from agentic_trading.execution.risk.post_trade import PostTradeChecker
from agentic_trading.execution.risk.pre_trade import PreTradeChecker
from agentic_trading.execution.risk.var_es import RiskMetrics

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
