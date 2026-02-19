"""Backward-compat re-export — canonical location: ``agentic_trading.signal.portfolio.sizing``.

Will be removed in PR 16.
"""

from agentic_trading.signal.portfolio.sizing import *  # noqa: F401, F403
from agentic_trading.signal.portfolio.sizing import (  # noqa: F811
    volatility_adjusted_size,
    fixed_fractional_size,
    kelly_size,
    liquidity_adjusted_size,
    stop_loss_based_size,
    scaled_entry_size,
)

__all__ = [
    "volatility_adjusted_size",
    "fixed_fractional_size",
    "kelly_size",
    "liquidity_adjusted_size",
    "stop_loss_based_size",
    "scaled_entry_size",
]
