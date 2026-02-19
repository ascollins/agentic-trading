"""Backward-compat re-export — canonical location: ``agentic_trading.signal.strategies.registry``.

Will be removed in PR 16.
"""

from agentic_trading.signal.strategies.registry import (  # noqa: F401
    _REGISTRY,
    create_strategy,
    list_strategies,
    register_strategy,
)
