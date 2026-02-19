"""Test PortfolioManager.on_signal produces TargetPosition."""

from decimal import Decimal

from agentic_trading.core.clock import SimClock
from agentic_trading.core.enums import Exchange, Side, SignalDirection, Timeframe
from agentic_trading.core.events import Signal, TargetPosition
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Instrument
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.portfolio.manager import PortfolioManager


def _make_ctx(instruments=None) -> TradingContext:
    return TradingContext(
        clock=SimClock(),
        event_bus=MemoryEventBus(),
        instruments=instruments or {},
    )


def _make_signal(
    direction: SignalDirection = SignalDirection.LONG,
    confidence: float = 0.7,
    strategy_id: str = "test_strategy",
    symbol: str = "BTC/USDT",
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        timeframe=Timeframe.M5,
        features_used={"close": 67000.0},
        risk_constraints={
            "sizing_method": "fixed_fractional",
            "atr": 500.0,
            "price": 67000.0,
        },
    )


class TestPortfolioManagerOnSignal:
    def test_on_signal_collects_signal(self):
        pm = PortfolioManager()
        signal = _make_signal()
        pm.on_signal(signal)
        assert len(pm._pending_signals) == 1

    def test_generate_targets_produces_target(self):
        pm = PortfolioManager()
        signal = _make_signal(direction=SignalDirection.LONG, confidence=0.8)
        pm.on_signal(signal)
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1
        assert isinstance(targets[0], TargetPosition)
        assert targets[0].side == Side.BUY
        assert targets[0].target_qty > Decimal("0")

    def test_generate_targets_short_signal(self):
        pm = PortfolioManager()
        signal = _make_signal(direction=SignalDirection.SHORT, confidence=0.8)
        pm.on_signal(signal)
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1
        assert targets[0].side == Side.SELL

    def test_generate_targets_clears_pending(self):
        pm = PortfolioManager()
        pm.on_signal(_make_signal())
        ctx = _make_ctx()
        pm.generate_targets(ctx, capital=100_000.0)
        assert len(pm._pending_signals) == 0

    def test_no_signals_produces_no_targets(self):
        pm = PortfolioManager()
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert targets == []

    def test_conflicting_signals_cancel_out(self):
        pm = PortfolioManager()
        pm.on_signal(_make_signal(direction=SignalDirection.LONG, confidence=0.5))
        pm.on_signal(_make_signal(direction=SignalDirection.SHORT, confidence=0.5))
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        # Net score ~0 -> should cancel out
        assert targets == []

    def test_multiple_symbols(self):
        pm = PortfolioManager()
        pm.on_signal(_make_signal(symbol="BTC/USDT", confidence=0.8))
        pm.on_signal(_make_signal(symbol="ETH/USDT", confidence=0.7))
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        symbols = {t.symbol for t in targets}
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols

    def test_target_inherits_signal_trace_id(self):
        """Regression: TargetPosition must carry the Signal's trace_id so the
        fill handler can look up TP/SL from the signal cache."""
        pm = PortfolioManager()
        signal = _make_signal(confidence=0.8)
        pm.on_signal(signal)
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1
        assert targets[0].trace_id == signal.trace_id

    def test_target_inherits_best_signal_trace_id_multi(self):
        """When multiple signals exist for the same symbol, the target
        should carry the trace_id of the highest-confidence signal."""
        pm = PortfolioManager()
        low = _make_signal(confidence=0.5, strategy_id="strat_a")
        high = _make_signal(confidence=0.9, strategy_id="strat_b")
        pm.on_signal(low)
        pm.on_signal(high)
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)
        assert len(targets) == 1
        assert targets[0].trace_id == high.trace_id
        assert targets[0].strategy_id == high.strategy_id

    def test_trace_id_propagates_through_intent_converter(self):
        """Full chain: Signal.trace_id -> TargetPosition -> OrderIntent."""
        from datetime import datetime, timezone
        from agentic_trading.signal.portfolio.intent_converter import build_order_intents

        pm = PortfolioManager()
        signal = _make_signal(confidence=0.8)
        pm.on_signal(signal)
        ctx = _make_ctx()
        targets = pm.generate_targets(ctx, capital=100_000.0)

        intents = build_order_intents(
            targets,
            exchange=Exchange.BYBIT,
            timestamp=datetime.now(timezone.utc),
        )
        assert len(intents) == 1
        assert intents[0].trace_id == signal.trace_id

    def test_sizing_multiplier(self):
        pm1 = PortfolioManager(sizing_multiplier=1.0)
        pm1.on_signal(_make_signal(confidence=0.8))
        ctx = _make_ctx()
        targets1 = pm1.generate_targets(ctx, capital=100_000.0)

        pm2 = PortfolioManager(sizing_multiplier=0.5)
        pm2.on_signal(_make_signal(confidence=0.8))
        targets2 = pm2.generate_targets(ctx, capital=100_000.0)

        if targets1 and targets2:
            assert targets2[0].target_qty < targets1[0].target_qty
