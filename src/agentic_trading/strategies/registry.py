"""Strategy registry and factory.

Strategies register themselves here. The runner uses this to instantiate strategies.
"""

from __future__ import annotations

from typing import Any, Type

from .base import BaseStrategy

_REGISTRY: dict[str, Type[BaseStrategy]] = {}


def register_strategy(strategy_id: str):
    """Decorator to register a strategy class."""

    def decorator(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
        _REGISTRY[strategy_id] = cls
        return cls

    return decorator


def create_strategy(strategy_id: str, params: dict[str, Any] | None = None) -> BaseStrategy:
    """Create a strategy instance by ID."""
    cls = _REGISTRY.get(strategy_id)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown strategy '{strategy_id}'. Available: {available}"
        )
    return cls(strategy_id=strategy_id, params=params)


def list_strategies() -> list[str]:
    """List all registered strategy IDs."""
    return sorted(_REGISTRY.keys())
