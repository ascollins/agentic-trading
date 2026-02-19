"""Backward-compatibility shim — see ``agentic_trading.reconciliation.journal.session_analysis``.

Will be removed in PR 16.
"""

from agentic_trading.reconciliation.journal.session_analysis import *  # noqa: F401, F403
from agentic_trading.reconciliation.journal.session_analysis import SessionAnalyser  # noqa: F811

__all__ = ["SessionAnalyser"]
