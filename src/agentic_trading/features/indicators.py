"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.features.indicators``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.features.indicators import *  # noqa: F401, F403

# Re-export commonly used indicator functions explicitly.
from agentic_trading.intelligence.features.indicators import (  # noqa: F811
    compute_adx,
    compute_atr,
    compute_bbw,
    compute_bollinger_bands,
    compute_donchian,
    compute_ema,
    compute_keltner,
)
