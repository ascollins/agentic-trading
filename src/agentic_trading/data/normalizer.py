"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.normalizer``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.normalizer import *  # noqa: F401, F403
from agentic_trading.intelligence.normalizer import (  # noqa: F811
    candle_to_event,
    normalize_ccxt_ohlcv,
    normalize_ccxt_ohlcv_batch,
    normalize_ccxt_orderbook,
    normalize_ccxt_ticker,
)

__all__ = [
    "normalize_ccxt_ohlcv",
    "normalize_ccxt_ohlcv_batch",
    "candle_to_event",
    "normalize_ccxt_ticker",
    "normalize_ccxt_orderbook",
]
