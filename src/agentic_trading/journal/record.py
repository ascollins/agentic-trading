"""Backward-compatibility shim — see ``agentic_trading.reconciliation.journal.record``.

Will be removed in PR 16.
"""

from agentic_trading.reconciliation.journal.record import *  # noqa: F401, F403
from agentic_trading.reconciliation.journal.record import TradeRecord, TradePhase, TradeOutcome  # noqa: F811

__all__ = ["TradeRecord", "TradePhase", "TradeOutcome"]
