"""Backward-compatibility shim — see ``agentic_trading.execution.risk.drawdown``.

Will be removed in PR 16.
"""

from agentic_trading.execution.risk.drawdown import *  # noqa: F401, F403
from agentic_trading.execution.risk.drawdown import DrawdownMonitor  # noqa: F811

__all__ = ["DrawdownMonitor"]
