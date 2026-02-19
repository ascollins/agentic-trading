"""Tests for portfolio coordinator: batch signal processing + position-aware suppression.

Covers:
1. Batch signal processing — conflicting, agreeing, multi-symbol
2. Position-aware suppression — same direction, opposite direction, no position
3. Price estimate propagation from signal → TargetPosition → PortfolioAllocator
4. process_signal_batch on SignalManager
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from agentic_trading.core.clock import SimClock
from agentic_trading.core.enums import (
    Exchange,
    PositionSide,
    Side,
    SignalDirection,
    Timeframe,
)
from agentic_trading.core.events import Signal, TargetPosition
from agentic_trading.core.interfaces import PortfolioState, TradingContext
from agentic_trading.core.models import Position
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.signal.manager import SignalManager
from agentic_trading.signal.portfolio.allocator import PortfolioAllocator
from agentic_trading.signal.portfolio.manager import PortfolioManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    strategy_id: str = "trend_following",
    symbol: str = "ETH/USDT",
    direction: SignalDirection = SignalDirection.LONG,
    confidence: float = 0.7,
    price: float = 2500.0,
) -> Signal:
    """Create a minimal Signal for testing."""
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        timeframe=Timeframe.M5,
        features_used={"close": price},
        risk_constraints={
            "sizing_method": "fixed_fractional",
            "atr": 50.0,
            "price": price,
        },
    )


def _make_position(
    symbol: str = "ETH/USDT",
    side: PositionSide = PositionSide.LONG,
    qty: float = 1.0,
) -> Position:
    """Create a Position with given side and qty."""
    return Position(
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        qty=Decimal(str(qty)),
        entry_price=Decimal("2500"),
        mark_price=Decimal("2500"),
        notional=Decimal(str(qty * 2500)),
    )


def _make_ctx(
    positions: dict[str, Position] | None = None,
) -> TradingContext:
    """Create a TradingContext with optional portfolio positions."""
    portfolio = PortfolioState(positions=positions or {})
    return TradingContext(
        clock=SimClock(),
        event_bus=MemoryEventBus(),
        instruments={},
        portfolio_state=portfolio,
    )


# ===========================================================================
# Batch signal processing
# ===========================================================================


class TestBatchSignalProcessing:
    """Test that batching signals enables proper conflict resolution."""

    def test_conflicting_signals_cancel_out(self):
        """BUY and SELL signals of similar confidence cancel each other."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(
            strategy_id="trend_following",
            direction=SignalDirection.LONG,
            confidence=0.55,
        ))
        pm.on_signal(_make_signal(
            strategy_id="mean_reversion",
            direction=SignalDirection.SHORT,
            confidence=0.55,
        ))
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        # net_score = 0.55 - 0.55 = 0.0 < 0.1 → cancelled
        assert targets == []

    def test_agreeing_signals_produce_one_target(self):
        """Two BUY signals for the same symbol produce a single target."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(
            strategy_id="trend_following",
            direction=SignalDirection.LONG,
            confidence=0.8,
        ))
        pm.on_signal(_make_signal(
            strategy_id="breakout",
            direction=SignalDirection.LONG,
            confidence=0.6,
        ))
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1
        assert targets[0].side == Side.BUY
        assert targets[0].symbol == "ETH/USDT"

    def test_different_symbols_produce_separate_targets(self):
        """Signals for different symbols produce one target each."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(symbol="ETH/USDT", confidence=0.8))
        pm.on_signal(_make_signal(
            symbol="BTC/USDT",
            confidence=0.7,
            price=95000.0,
        ))
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        symbols = {t.symbol for t in targets}
        assert symbols == {"ETH/USDT", "BTC/USDT"}

    def test_strong_buy_wins_over_weak_sell(self):
        """Strong BUY (0.8) wins over weak SELL (0.3)."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(
            strategy_id="trend_following",
            direction=SignalDirection.LONG,
            confidence=0.8,
        ))
        pm.on_signal(_make_signal(
            strategy_id="mean_reversion",
            direction=SignalDirection.SHORT,
            confidence=0.3,
        ))
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        # net_score = 0.8 - 0.3 = 0.5 > 0.1 → BUY wins
        assert len(targets) == 1
        assert targets[0].side == Side.BUY

    def test_pending_signals_cleared_after_batch(self):
        """generate_targets() clears pending signals after processing."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(confidence=0.8))
        ctx = _make_ctx()
        pm.generate_targets(ctx, capital=100_000.0)
        assert len(pm._pending_signals) == 0


# ===========================================================================
# Position-aware suppression
# ===========================================================================


class TestPositionAwareSuppression:
    """Test that existing positions suppress redundant or conflicting signals."""

    def test_suppressed_when_already_long_and_signal_is_buy(self):
        """BUY signal suppressed when already LONG on the symbol."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(direction=SignalDirection.LONG, confidence=0.8))
        ctx = _make_ctx(positions={
            "ETH/USDT": _make_position(side=PositionSide.LONG),
        })
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert targets == []

    def test_suppressed_when_already_short_and_signal_is_sell(self):
        """SELL signal suppressed when already SHORT on the symbol."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(direction=SignalDirection.SHORT, confidence=0.8))
        ctx = _make_ctx(positions={
            "ETH/USDT": _make_position(side=PositionSide.SHORT),
        })
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert targets == []

    def test_suppressed_opposite_direction_without_flat(self):
        """SELL signal suppressed when LONG — must FLAT to close first."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(direction=SignalDirection.SHORT, confidence=0.8))
        ctx = _make_ctx(positions={
            "ETH/USDT": _make_position(side=PositionSide.LONG),
        })
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert targets == []

    def test_no_suppression_when_no_position(self):
        """Signal passes through when no existing position."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(direction=SignalDirection.LONG, confidence=0.8))
        ctx = _make_ctx()  # No positions
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1

    def test_no_suppression_when_position_closed(self):
        """Signal passes through when position exists but qty=0 (closed)."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(direction=SignalDirection.LONG, confidence=0.8))
        ctx = _make_ctx(positions={
            "ETH/USDT": _make_position(side=PositionSide.LONG, qty=0),
        })
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1

    def test_different_symbol_not_suppressed(self):
        """BTC position doesn't suppress ETH signal."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(symbol="ETH/USDT", confidence=0.8))
        ctx = _make_ctx(positions={
            "BTC/USDT": _make_position(
                symbol="BTC/USDT", side=PositionSide.LONG,
            ),
        })
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1
        assert targets[0].symbol == "ETH/USDT"

    def test_batch_some_suppressed_some_pass(self):
        """In a batch, only signals with conflicting positions are suppressed."""
        pm = PortfolioManager()
        # ETH: already long → should be suppressed
        pm.on_signal(_make_signal(
            symbol="ETH/USDT", direction=SignalDirection.LONG, confidence=0.8,
        ))
        # BTC: no position → should pass
        pm.on_signal(_make_signal(
            symbol="BTC/USDT", direction=SignalDirection.LONG,
            confidence=0.7, price=95000.0,
        ))
        ctx = _make_ctx(positions={
            "ETH/USDT": _make_position(side=PositionSide.LONG),
        })
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1
        assert targets[0].symbol == "BTC/USDT"


# ===========================================================================
# Price estimate
# ===========================================================================


class TestPriceEstimate:
    """Test that price flows from signal → TargetPosition → Allocator."""

    def test_price_from_risk_constraints(self):
        """price_estimate populated from risk_constraints['price']."""
        pm = PortfolioManager()
        pm.on_signal(_make_signal(price=2500.0, confidence=0.8))
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1
        assert targets[0].price_estimate == 2500.0

    def test_price_falls_back_to_features_close(self):
        """price_estimate from features_used['close'] when rc has no price."""
        sig = _make_signal(price=3000.0, confidence=0.8)
        # Remove price from risk_constraints, keep in features_used
        sig.risk_constraints = {
            "sizing_method": "fixed_fractional",
            "atr": 50.0,
            # no "price" key
        }
        sig.features_used = {"close": 3000.0}
        pm = PortfolioManager()
        pm.on_signal(sig)
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1
        assert targets[0].price_estimate == 3000.0

    def test_allocator_uses_price_estimate(self):
        """PortfolioAllocator._estimate_price uses price_estimate field."""
        target = TargetPosition(
            strategy_id="test",
            symbol="ETH/USDT",
            target_qty=Decimal("1"),
            side=Side.BUY,
            price_estimate=2500.0,
        )
        price = PortfolioAllocator._estimate_price(target)
        assert price == 2500.0

    def test_allocator_fallback_when_no_price(self):
        """PortfolioAllocator._estimate_price returns 1.0 when price_estimate=0."""
        target = TargetPosition(
            strategy_id="test",
            symbol="ETH/USDT",
            target_qty=Decimal("1"),
            side=Side.BUY,
        )
        price = PortfolioAllocator._estimate_price(target)
        assert price == 1.0


# ===========================================================================
# SignalManager.process_signal_batch
# ===========================================================================


class TestProcessSignalBatch:
    """Test the batch processing method on SignalManager."""

    def _make_signal_manager(self) -> SignalManager:
        """Create a SignalManager with real PortfolioManager."""
        pm = PortfolioManager()
        return SignalManager(
            runner=None,
            portfolio_manager=pm,
            allocator=None,
            correlation_analyzer=None,
        )

    def test_batch_with_agreeing_signals(self):
        """Two agreeing BUY signals produce intents."""
        mgr = self._make_signal_manager()
        signals = [
            _make_signal(strategy_id="trend", confidence=0.8),
            _make_signal(strategy_id="breakout", confidence=0.6),
        ]
        ctx = _make_ctx()
        results = mgr.process_signal_batch(
            signals, None, ctx, Exchange.BYBIT, 100_000.0,
        )
        assert len(results) >= 1
        total_intents = sum(len(r.intents) for r in results)
        assert total_intents >= 1

    def test_batch_conflicting_signals_no_intents(self):
        """Conflicting BUY+SELL signals produce no intents."""
        mgr = self._make_signal_manager()
        signals = [
            _make_signal(strategy_id="trend", direction=SignalDirection.LONG, confidence=0.5),
            _make_signal(strategy_id="mean_rev", direction=SignalDirection.SHORT, confidence=0.5),
        ]
        ctx = _make_ctx()
        results = mgr.process_signal_batch(
            signals, None, ctx, Exchange.BYBIT, 100_000.0,
        )
        total_intents = sum(len(r.intents) for r in results)
        assert total_intents == 0

    def test_batch_empty_signals(self):
        """Empty signal list produces no results."""
        mgr = self._make_signal_manager()
        ctx = _make_ctx()
        results = mgr.process_signal_batch(
            [], None, ctx, Exchange.BYBIT, 100_000.0,
        )
        assert results == []

    def test_batch_populates_signal_cache(self):
        """signal_cache is populated for all signals in batch."""
        mgr = self._make_signal_manager()
        cache: dict = {}
        signals = [
            _make_signal(strategy_id="trend", confidence=0.8),
            _make_signal(strategy_id="breakout", confidence=0.6),
        ]
        ctx = _make_ctx()
        mgr.process_signal_batch(
            signals, None, ctx, Exchange.BYBIT, 100_000.0,
            signal_cache=cache,
        )
        assert len(cache) == 2
        for sig in signals:
            assert sig.trace_id in cache

    def test_batch_with_position_suppression(self):
        """Batch produces no intents when existing position matches."""
        mgr = self._make_signal_manager()
        signals = [
            _make_signal(direction=SignalDirection.LONG, confidence=0.8),
        ]
        ctx = _make_ctx(positions={
            "ETH/USDT": _make_position(side=PositionSide.LONG),
        })
        results = mgr.process_signal_batch(
            signals, None, ctx, Exchange.BYBIT, 100_000.0,
        )
        total_intents = sum(len(r.intents) for r in results)
        assert total_intents == 0
