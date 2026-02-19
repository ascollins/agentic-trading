"""Backward-compat re-export — canonical location: ``agentic_trading.signal.portfolio.allocator``.

Will be removed in PR 16.
"""

from agentic_trading.signal.portfolio.allocator import *  # noqa: F401, F403
from agentic_trading.signal.portfolio.allocator import (  # noqa: F811
    PortfolioAllocator,
)

__all__ = ["PortfolioAllocator"]
