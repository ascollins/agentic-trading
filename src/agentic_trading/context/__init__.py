"""Context Manager — unified fact table and keyword-indexed memory store.

Every agent reads context before reasoning and writes to it after.
"""

from __future__ import annotations

from .fact_table import (
    FactTable,
    FactTableSnapshot,
    PortfolioSnapshot,
    PriceLevels,
    RiskSnapshot,
)
from .manager import AgentContext, ContextManager
from .memory_store import InMemoryMemoryStore, JsonFileMemoryStore, MemoryEntry

__all__ = [
    "AgentContext",
    "ContextManager",
    "FactTable",
    "FactTableSnapshot",
    "InMemoryMemoryStore",
    "JsonFileMemoryStore",
    "MemoryEntry",
    "PortfolioSnapshot",
    "PriceLevels",
    "RiskSnapshot",
]
