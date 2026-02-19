"""Backward-compatibility shim — see ``agentic_trading.policy.strategy_lifecycle``.

Will be removed in PR 16.
"""

from agentic_trading.policy.strategy_lifecycle import *  # noqa: F401, F403
from agentic_trading.policy.strategy_lifecycle import (  # noqa: F811
    DEFAULT_DEMOTION_TRIGGERS,
    DEFAULT_EVIDENCE_GATES,
    STAGE_TO_MATURITY,
    StrategyLifecycleManager,
)

__all__ = [
    "DEFAULT_DEMOTION_TRIGGERS",
    "DEFAULT_EVIDENCE_GATES",
    "STAGE_TO_MATURITY",
    "StrategyLifecycleManager",
]
