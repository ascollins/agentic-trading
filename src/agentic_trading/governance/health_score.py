"""Backward-compatibility shim — see ``agentic_trading.policy.health_score``.

Will be removed in PR 16.
"""

from agentic_trading.policy.health_score import *  # noqa: F401, F403
from agentic_trading.policy.health_score import HealthTracker  # noqa: F811

__all__ = ["HealthTracker"]
