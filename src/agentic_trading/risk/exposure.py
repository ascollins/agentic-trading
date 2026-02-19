"""Backward-compatibility shim — see ``agentic_trading.execution.risk.exposure``.

Will be removed in PR 16.
"""

from agentic_trading.execution.risk.exposure import *  # noqa: F401, F403
from agentic_trading.execution.risk.exposure import (  # noqa: F811
    ExposureSnapshot,
    ExposureTracker,
)

__all__ = ["ExposureSnapshot", "ExposureTracker"]
