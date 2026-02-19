"""Backward-compatibility shim — see ``agentic_trading.policy.canary``.

Will be removed in PR 16.
"""

from agentic_trading.policy.canary import *  # noqa: F401, F403
from agentic_trading.policy.canary import GovernanceCanary  # noqa: F811

__all__ = ["GovernanceCanary"]
