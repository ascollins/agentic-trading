"""Tests for SignalManager.process_signal and alias_features.

Validates the signal-processing pipeline and feature aliasing extracted
from ``main._run_live_or_paper`` into the SignalManager facade (PR 14).

Tests cover:
1. alias_features — canonical indicator aliasing
2. process_signal — directional entry signals
3. process_signal — FLAT exit signals
4. process_signal — signal cache population
5. process_signal — exit map population
6. SignalResult dataclass correctness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic_trading.signal.manager import SignalManager, SignalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeDirection:
    value: str


@dataclass
class _FakeSignal:
    """Minimal signal for testing."""

    trace_id: str = "sig-001"
    strategy_id: str = "trend_following"
    symbol: str = "BTC/USDT"
    direction: Any = None
    confidence: float = 0.85
    rationale: str = "test rationale"
    features_used: list = field(default_factory=list)
    risk_constraints: dict = field(default_factory=dict)
    take_profit: Decimal | None = None
    stop_loss: Decimal | None = None
    trailing_stop: Decimal | None = None

    def __post_init__(self):
        if self.direction is None:
            self.direction = _FakeDirection("long")


@dataclass
class _FakeFillLeg:
    price: Decimal = Decimal("50000")


@dataclass
class _FakeOpenTrade:
    trace_id: str = "entry-001"
    direction: str = "long"
    remaining_qty: Decimal = Decimal("0.1")
    entry_fills: list = field(default_factory=lambda: [_FakeFillLeg()])


@dataclass
class _FakeClock:
    def now(self) -> datetime:
        return datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@dataclass
class _FakeCtx:
    clock: _FakeClock = field(default_factory=_FakeClock)


class _FakeJournal:
    """Minimal journal stub for testing."""

    def __init__(self, open_trade: _FakeOpenTrade | None = None):
        self._trade = open_trade

    def get_trade_by_position(self, strategy_id: str, symbol: str):
        if self._trade is not None:
            return self._trade
        return None


class _FakePortfolioManager:
    """Minimal portfolio manager that records calls."""

    def __init__(self, targets: list | None = None):
        self._targets = targets or []
        self.signals: list = []

    def on_signal(self, sig):
        self.signals.append(sig)

    def generate_targets(self, ctx, capital):
        return self._targets


def _make_mgr(
    portfolio_manager: _FakePortfolioManager | None = None,
) -> SignalManager:
    """Create a SignalManager with the given portfolio manager."""
    pm = portfolio_manager or _FakePortfolioManager()
    return SignalManager(
        runner=None,
        portfolio_manager=pm,
    )


# ---------------------------------------------------------------------------
# alias_features tests
# ---------------------------------------------------------------------------

class TestAliasFeatures:
    """SignalManager.alias_features canonical aliasing."""

    def test_aliases_adx(self):
        features = {"adx_14": 25.0, "close": 50000.0}
        result = SignalManager.alias_features(features)
        assert result["adx"] == 25.0
        assert result["adx_14"] == 25.0

    def test_aliases_atr(self):
        result = SignalManager.alias_features({"atr_14": 500.0})
        assert result["atr"] == 500.0

    def test_aliases_rsi(self):
        result = SignalManager.alias_features({"rsi_14": 65.0})
        assert result["rsi"] == 65.0

    def test_aliases_donchian_upper(self):
        result = SignalManager.alias_features({"donchian_upper_20": 51000.0})
        assert result["donchian_upper"] == 51000.0

    def test_aliases_donchian_lower(self):
        result = SignalManager.alias_features({"donchian_lower_20": 49000.0})
        assert result["donchian_lower"] == 49000.0

    def test_does_not_overwrite_existing(self):
        """If the short name already exists, don't overwrite."""
        features = {"adx_14": 25.0, "adx": 30.0}
        result = SignalManager.alias_features(features)
        assert result["adx"] == 30.0  # not overwritten

    def test_does_not_mutate_original(self):
        features = {"adx_14": 25.0}
        result = SignalManager.alias_features(features)
        assert "adx" not in features  # original unchanged
        assert "adx" in result

    def test_empty_features(self):
        result = SignalManager.alias_features({})
        assert result == {}

    def test_no_matching_keys(self):
        features = {"close": 50000.0, "volume": 1000.0}
        result = SignalManager.alias_features(features)
        assert result == features

    def test_all_aliases_at_once(self):
        features = {
            "adx_14": 1, "atr_14": 2, "rsi_14": 3,
            "donchian_upper_20": 4, "donchian_lower_20": 5,
        }
        result = SignalManager.alias_features(features)
        assert result["adx"] == 1
        assert result["atr"] == 2
        assert result["rsi"] == 3
        assert result["donchian_upper"] == 4
        assert result["donchian_lower"] == 5


# ---------------------------------------------------------------------------
# process_signal — directional entry
# ---------------------------------------------------------------------------

class TestProcessSignalEntry:
    """process_signal handles directional (non-FLAT) signals."""

    def test_no_targets_returns_empty_intents(self):
        pm = _FakePortfolioManager(targets=[])
        mgr = _make_mgr(pm)
        sig = _FakeSignal()
        ctx = _FakeCtx()

        result = mgr.process_signal(
            sig, _FakeJournal(), ctx, "binance", 100_000.0,
        )

        assert result.intents == []
        assert result.is_exit is False
        assert pm.signals == [sig]

    def test_with_targets_generates_intents(self):
        """When PM returns targets, intents are produced."""
        from agentic_trading.core.enums import Exchange, Side
        from agentic_trading.core.events import TargetPosition

        target = TargetPosition(
            strategy_id="trend_following",
            symbol="BTC/USDT",
            side=Side.BUY,
            target_qty=Decimal("0.1"),
            trace_id="sig-001",
        )
        pm = _FakePortfolioManager(targets=[target])
        mgr = _make_mgr(pm)
        sig = _FakeSignal()
        ctx = _FakeCtx()

        result = mgr.process_signal(
            sig, _FakeJournal(), ctx, Exchange.BINANCE, 100_000.0,
        )

        assert len(result.intents) >= 1
        assert result.is_exit is False

    def test_populates_signal_cache(self):
        pm = _FakePortfolioManager()
        mgr = _make_mgr(pm)
        sig = _FakeSignal()
        ctx = _FakeCtx()
        cache: dict[str, Any] = {}

        mgr.process_signal(
            sig, _FakeJournal(), ctx, "binance", 100_000.0,
            signal_cache=cache,
        )

        assert cache["sig-001"] is sig


# ---------------------------------------------------------------------------
# process_signal — FLAT exit
# ---------------------------------------------------------------------------

class TestProcessSignalExit:
    """process_signal handles FLAT signals by generating exit intents."""

    def test_flat_signal_no_open_trade(self):
        """FLAT with no open trade → no intents."""
        mgr = _make_mgr()
        sig = _FakeSignal(direction=_FakeDirection("flat"))
        ctx = _FakeCtx()

        result = mgr.process_signal(
            sig, _FakeJournal(open_trade=None), ctx, "binance", 100_000.0,
        )

        assert result.intents == []
        assert result.is_exit is False

    def test_flat_signal_with_open_trade_generates_exit(self):
        """FLAT with open trade → exit intent."""
        mgr = _make_mgr()
        sig = _FakeSignal(direction=_FakeDirection("flat"))
        ctx = _FakeCtx()
        journal = _FakeJournal(open_trade=_FakeOpenTrade())

        result = mgr.process_signal(
            sig, journal, ctx, "binance", 100_000.0,
        )

        assert len(result.intents) == 1
        assert result.is_exit is True
        assert result.entry_trace_id == "entry-001"

        intent = result.intents[0]
        assert intent.reduce_only is True
        assert intent.qty == Decimal("0.1")
        assert intent.side.value == "sell"  # reverse of long

    def test_flat_signal_short_trade_generates_buy_exit(self):
        """FLAT on a short trade → buy exit."""
        mgr = _make_mgr()
        sig = _FakeSignal(direction=_FakeDirection("flat"))
        ctx = _FakeCtx()
        journal = _FakeJournal(
            open_trade=_FakeOpenTrade(direction="short"),
        )

        result = mgr.process_signal(
            sig, journal, ctx, "binance", 100_000.0,
        )

        assert result.intents[0].side.value == "buy"

    def test_flat_populates_exit_map(self):
        mgr = _make_mgr()
        sig = _FakeSignal(direction=_FakeDirection("flat"))
        ctx = _FakeCtx()
        journal = _FakeJournal(open_trade=_FakeOpenTrade())
        exit_map: dict[str, str] = {}

        mgr.process_signal(
            sig, journal, ctx, "binance", 100_000.0,
            exit_map=exit_map,
        )

        assert exit_map[sig.trace_id] == "entry-001"

    def test_flat_zero_remaining_qty_no_intent(self):
        """FLAT with remaining_qty=0 → no exit intent."""
        mgr = _make_mgr()
        sig = _FakeSignal(direction=_FakeDirection("flat"))
        ctx = _FakeCtx()
        trade = _FakeOpenTrade(remaining_qty=Decimal("0"))
        journal = _FakeJournal(open_trade=trade)

        result = mgr.process_signal(
            sig, journal, ctx, "binance", 100_000.0,
        )

        assert result.intents == []
        assert result.is_exit is False

    def test_flat_with_recon_manager_as_journal(self):
        """process_signal accepts a ReconciliationManager as journal."""
        from agentic_trading.reconciliation.manager import ReconciliationManager

        mgr = _make_mgr()
        sig = _FakeSignal(direction=_FakeDirection("flat"))
        ctx = _FakeCtx()

        # ReconciliationManager wraps a real journal
        recon = ReconciliationManager.from_config()
        # Inject a fake open trade via the underlying journal's internal state
        # (not ideal, but tests the hasattr('journal') branch)
        result = mgr.process_signal(
            sig, recon, ctx, "binance", 100_000.0,
        )

        # No open trade in the empty journal → no intent
        assert result.intents == []


# ---------------------------------------------------------------------------
# SignalResult tests
# ---------------------------------------------------------------------------

class TestSignalResult:
    """SignalResult dataclass."""

    def test_defaults(self):
        r = SignalResult()
        assert r.intents == []
        assert r.is_exit is False
        assert r.exit_trace_id is None
        assert r.entry_trace_id is None

    def test_fields(self):
        r = SignalResult(
            intents=["a", "b"],
            is_exit=True,
            exit_trace_id="e1",
            entry_trace_id="e2",
        )
        assert len(r.intents) == 2
        assert r.is_exit is True
