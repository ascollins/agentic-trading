"""Backward-compat re-export — canonical location: ``agentic_trading.signal.portfolio.manager``.

Will be removed in PR 16.
"""

from agentic_trading.signal.portfolio.manager import *  # noqa: F401, F403
from agentic_trading.signal.portfolio.manager import (  # noqa: F811
    PortfolioManager,
)

__all__ = ["PortfolioManager"]
