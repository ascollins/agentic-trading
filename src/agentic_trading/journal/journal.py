"""Backward-compatibility shim — see ``agentic_trading.reconciliation.journal.journal``.

Will be removed in PR 16.
"""

from agentic_trading.reconciliation.journal.journal import *  # noqa: F401, F403
from agentic_trading.reconciliation.journal.journal import TradeJournal  # noqa: F811

__all__ = ["TradeJournal"]
