"""Tests for the ContextManager component."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic_trading.context.fact_table import FactTable, PortfolioSnapshot
from agentic_trading.context.manager import AgentContext, ContextManager
from agentic_trading.context.memory_store import (
    InMemoryMemoryStore,
    MemoryEntry,
)
from agentic_trading.core.enums import MemoryEntryType, Mode


def _make_context_manager() -> ContextManager:
    """Create a ContextManager with in-memory stores."""
    return ContextManager(
        fact_table=FactTable(),
        memory_store=InMemoryMemoryStore(),
    )


class TestContextManagerReadContext:
    def test_returns_agent_context(self):
        cm = _make_context_manager()
        ctx = cm.read_context()
        assert isinstance(ctx, AgentContext)
        assert ctx.fact_snapshot is not None
        assert isinstance(ctx.relevant_memories, list)

    def test_includes_memories(self):
        cm = _make_context_manager()
        cm.write_analysis(
            MemoryEntryType.CMT_ASSESSMENT,
            {"test": True},
            symbol="BTC/USDT",
            summary="Test assessment",
        )
        ctx = cm.read_context(symbol="BTC/USDT")
        assert len(ctx.relevant_memories) == 1
        assert ctx.relevant_memories[0].symbol == "BTC/USDT"

    def test_excludes_memories_when_disabled(self):
        cm = _make_context_manager()
        cm.write_analysis(
            MemoryEntryType.CMT_ASSESSMENT,
            {"test": True},
        )
        ctx = cm.read_context(include_memory=False)
        assert len(ctx.relevant_memories) == 0

    def test_memory_limit(self):
        cm = _make_context_manager()
        for i in range(10):
            cm.write_analysis(
                MemoryEntryType.SIGNAL,
                {"i": i},
                symbol="BTC/USDT",
            )
        ctx = cm.read_context(symbol="BTC/USDT", memory_limit=3)
        assert len(ctx.relevant_memories) == 3


class TestContextManagerWriteAnalysis:
    def test_returns_entry_id(self):
        cm = _make_context_manager()
        entry_id = cm.write_analysis(
            MemoryEntryType.TRADE_PLAN,
            {"direction": "long"},
            symbol="ETH/USDT",
        )
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    def test_stored_entry_queryable(self):
        cm = _make_context_manager()
        cm.write_analysis(
            MemoryEntryType.HTF_ASSESSMENT,
            {"bias": "bullish"},
            symbol="BTC/USDT",
            tags=["bullish"],
            summary="Bullish HTF assessment",
        )
        results = cm.memory.query(
            symbol="BTC/USDT",
            entry_type=MemoryEntryType.HTF_ASSESSMENT,
        )
        assert len(results) == 1
        assert results[0].summary == "Bullish HTF assessment"


class TestContextManagerSyncFromTradingContext:
    def test_syncs_portfolio(self):
        cm = _make_context_manager()
        ctx = MagicMock()
        ctx.portfolio_state.gross_exposure = 50000.0
        ctx.portfolio_state.net_exposure = 30000.0
        ctx.portfolio_state.positions = {}
        ctx.risk_limits = {
            "max_portfolio_leverage": 5.0,
            "max_single_position_pct": 0.15,
            "max_daily_loss_pct": 0.03,
        }

        cm.sync_from_trading_context(ctx)

        portfolio = cm.facts.get_portfolio()
        assert portfolio.gross_exposure == 50000.0

        risk = cm.facts.get_risk()
        assert risk.max_portfolio_leverage == 5.0

    def test_handles_missing_portfolio(self):
        cm = _make_context_manager()
        ctx = MagicMock()
        ctx.portfolio_state = None  # Would cause AttributeError
        ctx.risk_limits = {}

        # Should not raise
        cm.sync_from_trading_context(ctx)


class TestContextManagerFromConfig:
    def test_backtest_creates_in_memory(self):
        cm = ContextManager.from_config(Mode.BACKTEST)
        assert isinstance(cm.memory, InMemoryMemoryStore)

    def test_paper_creates_jsonl(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            from agentic_trading.context.memory_store import (
                JsonFileMemoryStore,
            )

            cm = ContextManager.from_config(
                Mode.PAPER,
                data_dir=tmpdir,
                memory_store_path=str(
                    Path(tmpdir) / "test_memory.jsonl"
                ),
            )
            assert isinstance(cm.memory, JsonFileMemoryStore)


class TestContextManagerProperties:
    def test_facts_property(self):
        cm = _make_context_manager()
        assert isinstance(cm.facts, FactTable)

    def test_memory_property(self):
        cm = _make_context_manager()
        assert isinstance(cm.memory, InMemoryMemoryStore)
