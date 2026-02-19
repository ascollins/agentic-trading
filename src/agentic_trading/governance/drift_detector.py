"""Backward-compatibility shim — see ``agentic_trading.policy.drift_detector``.

Will be removed in PR 16.
"""

from agentic_trading.policy.drift_detector import *  # noqa: F401, F403
from agentic_trading.policy.drift_detector import DriftDetector  # noqa: F811

__all__ = ["DriftDetector"]
