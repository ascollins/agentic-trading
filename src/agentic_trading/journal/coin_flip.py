"""Backward-compatibility shim — see ``agentic_trading.reconciliation.journal.coin_flip``.

Will be removed in PR 16.
"""

from agentic_trading.reconciliation.journal.coin_flip import *  # noqa: F401, F403
from agentic_trading.reconciliation.journal.coin_flip import CoinFlipBaseline  # noqa: F811

__all__ = ["CoinFlipBaseline"]
