"""TradingContext: the mode-agnostic interface strategies use.

Re-exported from interfaces.py for convenience.
Strategies import from here and never need to know the mode.
"""

from .interfaces import PortfolioState, TradingContext

__all__ = ["TradingContext", "PortfolioState"]
