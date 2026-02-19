"""Backward-compatibility shim — see ``agentic_trading.reconciliation.loop``.

Will be removed in PR 16.
"""

from agentic_trading.reconciliation.loop import *  # noqa: F401, F403
from agentic_trading.reconciliation.loop import ReconciliationLoop  # noqa: F811

__all__ = ["ReconciliationLoop"]
