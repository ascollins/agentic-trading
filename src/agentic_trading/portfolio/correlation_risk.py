"""Backward-compat re-export — canonical location: ``agentic_trading.signal.portfolio.correlation_risk``.

Will be removed in PR 16.
"""

from agentic_trading.signal.portfolio.correlation_risk import *  # noqa: F401, F403
from agentic_trading.signal.portfolio.correlation_risk import (  # noqa: F811
    CorrelationRiskAnalyzer,
    quick_correlation_check,
)

__all__ = ["CorrelationRiskAnalyzer", "quick_correlation_check"]
