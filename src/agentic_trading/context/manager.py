"""Context Manager — unified facade for fact table and memory store.

Every agent reads and writes through this single entry point.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.enums import MemoryEntryType, Mode
from agentic_trading.core.ids import new_id
from agentic_trading.core.ids import utc_now as _now

from .fact_table import FactTable, FactTableSnapshot
from .memory_store import (
    InMemoryMemoryStore,
    JsonFileMemoryStore,
    MemoryEntry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent context bundle
# ---------------------------------------------------------------------------


class AgentContext(BaseModel):
    """Bundle returned by ``read_context()`` — everything an agent needs."""

    fact_snapshot: FactTableSnapshot = Field(
        default_factory=FactTableSnapshot
    )
    relevant_memories: list[MemoryEntry] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------


class ContextManager:
    """Unified context facade composing FactTable + MemoryStore.

    Owns both the real-time state (fact table) and the historical
    analysis memory (memory store). Agents call ``read_context()``
    before reasoning and ``write_analysis()`` after producing output.

    Parameters
    ----------
    fact_table:
        Real-time structured state table.
    memory_store:
        Keyword-indexed historical analysis store.
    """

    def __init__(
        self,
        fact_table: FactTable,
        memory_store: InMemoryMemoryStore | JsonFileMemoryStore,
    ) -> None:
        self._fact_table = fact_table
        self._memory_store = memory_store

    @classmethod
    def from_config(
        cls,
        mode: Mode,
        data_dir: str = "data",
        *,
        memory_ttl_hours: float = 24.0,
        max_memory_entries: int = 10_000,
        memory_store_path: str | None = None,
    ) -> ContextManager:
        """Create a ContextManager appropriate for the trading mode.

        - Backtest: in-memory stores (fast, no persistence).
        - Paper/Live: JSONL-backed memory store for persistence.
        """
        fact_table = FactTable()

        if mode == Mode.BACKTEST:
            memory_store: InMemoryMemoryStore | JsonFileMemoryStore = (
                InMemoryMemoryStore(max_entries=max_memory_entries)
            )
        else:
            path = memory_store_path or str(
                Path(data_dir) / "memory_store.jsonl"
            )
            memory_store = JsonFileMemoryStore(
                path, max_entries=max_memory_entries
            )

        return cls(fact_table=fact_table, memory_store=memory_store)

    # ------------------------------------------------------------------
    # Agent convenience API
    # ------------------------------------------------------------------

    def read_context(
        self,
        symbol: str | None = None,
        *,
        include_memory: bool = True,
        memory_limit: int = 5,
        memory_types: list[MemoryEntryType] | None = None,
    ) -> AgentContext:
        """Build a context bundle for an agent's reasoning cycle.

        Parameters
        ----------
        symbol:
            Filter memories to this symbol. If None, returns all.
        include_memory:
            Whether to include historical memories.
        memory_limit:
            Max number of memories to include.
        memory_types:
            Filter memories to specific types.
        """
        fact_snapshot = self._fact_table.snapshot()

        memories: list[MemoryEntry] = []
        if include_memory:
            # If specific types requested, query each
            if memory_types:
                for entry_type in memory_types:
                    memories.extend(
                        self._memory_store.query(
                            symbol=symbol,
                            entry_type=entry_type,
                            limit=memory_limit,
                        )
                    )
                # Re-sort by relevance and truncate
                memories.sort(
                    key=lambda m: m.relevance_score, reverse=True
                )
                memories = memories[:memory_limit]
            else:
                memories = self._memory_store.query(
                    symbol=symbol,
                    limit=memory_limit,
                )

        return AgentContext(
            fact_snapshot=fact_snapshot,
            relevant_memories=memories,
        )

    def write_analysis(
        self,
        entry_type: MemoryEntryType,
        content: dict[str, Any],
        *,
        symbol: str = "",
        timeframe: str = "",
        strategy_id: str = "",
        tags: list[str] | None = None,
        summary: str = "",
        ttl_hours: float = 24.0,
    ) -> str:
        """Store an analysis result in memory. Returns the entry_id."""
        entry = MemoryEntry(
            entry_id=new_id(),
            entry_type=entry_type,
            symbol=symbol,
            timeframe=timeframe,
            strategy_id=strategy_id,
            tags=tags or [],
            content=content,
            summary=summary,
            ttl_hours=ttl_hours,
        )
        self._memory_store.store(entry)
        return entry.entry_id

    # ------------------------------------------------------------------
    # Sync from TradingContext
    # ------------------------------------------------------------------

    def sync_from_trading_context(self, ctx: Any) -> None:
        """Pull current state from a TradingContext into the fact table.

        Parameters
        ----------
        ctx:
            A ``TradingContext`` instance. Typed as ``Any`` to avoid
            circular imports.
        """
        from .fact_table import PortfolioSnapshot

        try:
            portfolio = PortfolioSnapshot(
                gross_exposure=float(ctx.portfolio_state.gross_exposure),
                net_exposure=float(ctx.portfolio_state.net_exposure),
                open_position_count=len(ctx.portfolio_state.positions),
                positions={
                    sym: {
                        "qty": float(p.qty),
                        "entry_price": float(p.entry_price),
                    }
                    for sym, p in ctx.portfolio_state.positions.items()
                },
            )
            self._fact_table.update_portfolio(portfolio)
        except Exception:
            logger.debug("Could not sync portfolio from TradingContext")

        try:
            risk_limits = ctx.risk_limits or {}
            self._fact_table.update_risk(
                max_portfolio_leverage=risk_limits.get(
                    "max_portfolio_leverage", 3.0
                ),
                max_single_position_pct=risk_limits.get(
                    "max_single_position_pct", 0.1
                ),
                max_daily_loss_pct=risk_limits.get(
                    "max_daily_loss_pct", 0.05
                ),
            )
        except Exception:
            logger.debug("Could not sync risk limits from TradingContext")

    # ------------------------------------------------------------------
    # Pass-through properties
    # ------------------------------------------------------------------

    @property
    def facts(self) -> FactTable:
        """Access the underlying fact table."""
        return self._fact_table

    @property
    def memory(self) -> InMemoryMemoryStore | JsonFileMemoryStore:
        """Access the underlying memory store."""
        return self._memory_store
