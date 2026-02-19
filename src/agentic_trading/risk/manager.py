"""Backward-compatibility shim — see ``agentic_trading.execution.risk.manager``.

Will be removed in PR 16.
"""

from agentic_trading.execution.risk.manager import *  # noqa: F401, F403
from agentic_trading.execution.risk.manager import RiskManager  # noqa: F811

__all__ = ["RiskManager"]
