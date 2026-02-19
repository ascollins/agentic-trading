"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.analysis.rr_calculator``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.analysis.rr_calculator import *  # noqa: F401, F403
from agentic_trading.intelligence.analysis.rr_calculator import (  # noqa: F811
    RRResult,
    TargetLevel,
    calculate_rr,
    project_pnl,
)

__all__ = ["RRResult", "TargetLevel", "calculate_rr", "project_pnl"]
