"""Backward-compatibility shim — see ``agentic_trading.reconciliation.journal.persistence``.

Will be removed in PR 16.
"""

from agentic_trading.reconciliation.journal.persistence import *  # noqa: F401, F403
from agentic_trading.reconciliation.journal.persistence import (  # noqa: F811
    TradeRecordDB,
    JournalRepo,
    trade_to_db,
)

__all__ = ["TradeRecordDB", "JournalRepo", "trade_to_db"]
