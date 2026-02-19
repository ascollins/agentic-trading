"""Backward-compatibility shim — see ``agentic_trading.execution.risk.kill_switch``.

Will be removed in PR 16.
"""

from agentic_trading.execution.risk.kill_switch import *  # noqa: F401, F403
from agentic_trading.execution.risk.kill_switch import KillSwitch  # noqa: F811

__all__ = ["KillSwitch"]
