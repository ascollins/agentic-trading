"""Tests for ReconciliationManager.handle_fill and reconcile_positions.

Validates the fill-handling and position-reconciliation logic extracted
from ``main._run_live_or_paper`` into the ReconciliationManager facade
(PR 14).

Tests cover:
1. Entry fill classification and journal recording
2. Exit fill classification via exit_map
3. Exit fill classification via opposing-side detection
4. Strategy ID fallback for unknown signals
5. FillResult dataclass correctness
6. reconcile_positions force-closes orphaned trades
7. reconcile_positions preserves active trades
8. reconcile_positions handles CCXT symbol format
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from agentic_trading.reconciliation.manager import FillResult, ReconciliationManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeFillEvent:
    """Minimal fill event for testing."""

    trace_id: str = "trace-001"
    fill_id: str = "fill-001"
    order_id: str = "order-001"
    client_order_id: str = "client-001"
    symbol: str = "BTC/USDT"
    exchange: str = "binance"
    side: str = "buy"
    price: Decimal = Decimal("50000")
    qty: Decimal = Decimal("0.1")
    fee: Decimal = Decimal("5")
    fee_currency: str = "USDT"
    is_maker: bool = False
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


@dataclass
class _FakeSignal:
    """Minimal signal for testing."""

    trace_id: str = "trace-001"
    strategy_id: str = "trend_following"
    symbol: str = "BTC/USDT"
    direction: Any = None
    confidence: float = 0.8
    rationale: str = "test"
    features_used: list = field(default_factory=list)

    def __post_init__(self):
        if self.direction is None:
            self.direction = _FakeDirection("long")


@dataclass
class _FakeDirection:
    value: str


def _make_mgr() -> ReconciliationManager:
    """Create a ReconciliationManager with a real TradeJournal."""
    return ReconciliationManager.from_config(max_closed_trades=100)


# ---------------------------------------------------------------------------
# Entry fill tests
# ---------------------------------------------------------------------------

class TestHandleFillEntry:
    """handle_fill classifies and records entry fills."""

    def test_entry_fill_opens_trade(self):
        mgr = _make_mgr()
        fill = _FakeFillEvent()
        sig = _FakeSignal()
        cache: dict[str, Any] = {sig.trace_id: sig}
        exit_map: dict[str, str] = {}

        result = mgr.handle_fill(fill, cache, exit_map)

        assert result.is_exit is False
        assert result.strategy_id == "trend_following"
        assert result.direction == "long"
        assert result.entry_trace_id is None
        assert mgr.journal.open_trade_count == 1

    def test_entry_fill_records_fill_leg(self):
        mgr = _make_mgr()
        fill = _FakeFillEvent()
        sig = _FakeSignal()
        cache: dict[str, Any] = {sig.trace_id: sig}

        mgr.handle_fill(fill, cache, {})

        trade = mgr.journal.get_trade(fill.trace_id)
        assert trade is not None
        assert len(trade.entry_fills) == 1
        assert trade.entry_fills[0].price == Decimal("50000")

    def test_entry_fill_short_direction(self):
        mgr = _make_mgr()
        fill = _FakeFillEvent(side="sell")
        sig = _FakeSignal(direction=_FakeDirection("short"))
        cache: dict[str, Any] = {sig.trace_id: sig}

        result = mgr.handle_fill(fill, cache, {})

        assert result.direction == "short"
        trade = mgr.journal.get_trade(fill.trace_id)
        assert trade.direction == "short"

    def test_entry_fill_unknown_strategy_defaults_to_long(self):
        """With no cached signal, side=buy implies long."""
        mgr = _make_mgr()
        fill = _FakeFillEvent()
        cache: dict[str, Any] = {}

        result = mgr.handle_fill(fill, cache, {})

        assert result.strategy_id == "unknown"
        assert result.direction == "long"

    def test_entry_fill_unknown_sell_defaults_to_short(self):
        """With no cached signal, side=sell implies short."""
        mgr = _make_mgr()
        fill = _FakeFillEvent(side="sell")
        cache: dict[str, Any] = {}

        result = mgr.handle_fill(fill, cache, {})

        assert result.direction == "short"


# ---------------------------------------------------------------------------
# Exit fill tests
# ---------------------------------------------------------------------------

class TestHandleFillExit:
    """handle_fill classifies and records exit fills."""

    def _open_trade(self, mgr: ReconciliationManager) -> str:
        """Open a long trade and return its trace_id."""
        fill = _FakeFillEvent(trace_id="entry-001")
        sig = _FakeSignal(trace_id="entry-001")
        cache = {sig.trace_id: sig}
        mgr.handle_fill(fill, cache, {})
        return "entry-001"

    def test_exit_via_exit_map(self):
        mgr = _make_mgr()
        entry_tid = self._open_trade(mgr)

        exit_fill = _FakeFillEvent(
            trace_id="exit-001", side="sell",
        )
        exit_map = {"exit-001": entry_tid}
        cache: dict[str, Any] = {}

        result = mgr.handle_fill(exit_fill, cache, exit_map)

        assert result.is_exit is True
        assert result.entry_trace_id == entry_tid
        assert "exit-001" not in exit_map  # consumed

    def test_exit_via_opposing_side(self):
        """Exit detected by opposing side when no exit_map entry."""
        mgr = _make_mgr()
        self._open_trade(mgr)

        exit_fill = _FakeFillEvent(
            trace_id="exit-002", side="sell",
        )
        sig = _FakeSignal(trace_id="exit-002")
        cache = {sig.trace_id: sig}

        result = mgr.handle_fill(exit_fill, cache, {})

        assert result.is_exit is True
        assert result.entry_trace_id == "entry-001"

    def test_exit_records_exit_fill(self):
        mgr = _make_mgr()
        entry_tid = self._open_trade(mgr)

        exit_fill = _FakeFillEvent(
            trace_id="exit-003", side="sell", price=Decimal("55000"),
        )
        exit_map = {"exit-003": entry_tid}

        mgr.handle_fill(exit_fill, {}, exit_map)

        trade = mgr.journal.get_trade(entry_tid)
        assert len(trade.exit_fills) == 1
        assert trade.exit_fills[0].price == Decimal("55000")

    def test_exit_with_fallback_strategy_ids(self):
        """Unknown strategy is resolved via fallback_strategy_ids."""
        mgr = _make_mgr()
        self._open_trade(mgr)  # opened with strategy_id="trend_following"

        exit_fill = _FakeFillEvent(
            trace_id="exit-004", side="sell",
        )
        # No cached signal → strategy_id will be "unknown"
        cache: dict[str, Any] = {}

        result = mgr.handle_fill(
            exit_fill, cache, {},
            fallback_strategy_ids=["trend_following", "mean_reversion"],
        )

        assert result.is_exit is True
        assert result.strategy_id == "trend_following"


# ---------------------------------------------------------------------------
# FillResult tests
# ---------------------------------------------------------------------------

class TestFillResult:
    """FillResult dataclass."""

    def test_defaults(self):
        r = FillResult(is_exit=False, strategy_id="test")
        assert r.entry_trace_id is None
        assert r.direction == ""

    def test_fields(self):
        r = FillResult(
            is_exit=True,
            strategy_id="tf",
            entry_trace_id="entry-1",
            direction="long",
        )
        assert r.is_exit is True
        assert r.strategy_id == "tf"


# ---------------------------------------------------------------------------
# Position reconciliation tests
# ---------------------------------------------------------------------------

@dataclass
class _FakePosition:
    symbol: str
    qty: Decimal = Decimal("1")


class TestReconcilePositions:
    """reconcile_positions force-closes orphaned trades."""

    def _open_trade(
        self, mgr: ReconciliationManager, symbol: str = "BTC/USDT",
        trace_id: str = "trace-r1",
    ) -> str:
        fill = _FakeFillEvent(trace_id=trace_id, symbol=symbol)
        sig = _FakeSignal(trace_id=trace_id, symbol=symbol)
        cache = {sig.trace_id: sig}
        mgr.handle_fill(fill, cache, {})
        return trace_id

    def test_no_orphans(self):
        """Trades with matching exchange positions are kept."""
        mgr = _make_mgr()
        self._open_trade(mgr, "BTC/USDT", "t1")

        positions = [_FakePosition(symbol="BTC/USDT")]
        closed = mgr.reconcile_positions(positions)

        assert closed == []
        assert mgr.journal.open_trade_count == 1

    def test_force_closes_orphan(self):
        """Trade with no exchange position is force-closed."""
        mgr = _make_mgr()
        self._open_trade(mgr, "BTC/USDT", "t1")

        positions: list = []  # no exchange positions
        closed = mgr.reconcile_positions(positions)

        assert "t1" in closed
        assert mgr.journal.open_trade_count == 0

    def test_ccxt_symbol_format(self):
        """CCXT `:USDT` suffix is normalised for comparison."""
        mgr = _make_mgr()
        self._open_trade(mgr, "BTC/USDT", "t1")

        # Exchange position uses CCXT format
        positions = [_FakePosition(symbol="BTC/USDT:USDT")]
        closed = mgr.reconcile_positions(positions)

        assert closed == []
        assert mgr.journal.open_trade_count == 1

    def test_returns_closed_trace_ids(self):
        mgr = _make_mgr()
        self._open_trade(mgr, "BTC/USDT", "t1")
        self._open_trade(mgr, "ETH/USDT", "t2")

        # Only BTC has exchange position
        positions = [_FakePosition(symbol="BTC/USDT")]
        closed = mgr.reconcile_positions(positions)

        assert "t2" in closed
        assert "t1" not in closed

    def test_empty_open_trades(self):
        mgr = _make_mgr()
        closed = mgr.reconcile_positions([])
        assert closed == []
