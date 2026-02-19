"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.features.engine``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.features.engine import *  # noqa: F401, F403
from agentic_trading.intelligence.features.engine import FeatureEngine  # noqa: F811

__all__ = ["FeatureEngine"]
