"""Backward-compatibility shim — see ``agentic_trading.execution.risk.var_es``.

Will be removed in PR 16.
"""

from agentic_trading.execution.risk.var_es import *  # noqa: F401, F403
from agentic_trading.execution.risk.var_es import RiskMetrics  # noqa: F811

__all__ = ["RiskMetrics"]
