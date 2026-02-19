"""Backward-compatibility shim — see ``agentic_trading.reconciliation.journal.overtrading``.

Will be removed in PR 16.
"""

from agentic_trading.reconciliation.journal.overtrading import *  # noqa: F401, F403
from agentic_trading.reconciliation.journal.overtrading import OvertradingDetector  # noqa: F811

__all__ = ["OvertradingDetector"]
