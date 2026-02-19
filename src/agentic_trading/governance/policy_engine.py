"""Backward-compatibility shim — see ``agentic_trading.policy.engine``.

Will be removed in PR 16.
"""

from agentic_trading.policy.engine import *  # noqa: F401, F403
from agentic_trading.policy.engine import PolicyEngine  # noqa: F811

__all__ = ["PolicyEngine"]
