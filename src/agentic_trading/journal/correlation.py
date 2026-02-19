"""Backward-compatibility shim — see ``agentic_trading.reconciliation.journal.correlation``.

Will be removed in PR 16.
"""

from agentic_trading.reconciliation.journal.correlation import *  # noqa: F401, F403
from agentic_trading.reconciliation.journal.correlation import CorrelationMatrix  # noqa: F811

__all__ = ["CorrelationMatrix"]
