"""Backward-compatibility shim — see ``agentic_trading.policy.store``.

Will be removed in PR 16.
"""

from agentic_trading.policy.store import *  # noqa: F401, F403
from agentic_trading.policy.store import PolicyStore  # noqa: F811

__all__ = ["PolicyStore"]
