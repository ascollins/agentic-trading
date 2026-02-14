"""Test PreTradeChecker blocks oversized orders and PostTradeChecker detects leverage spikes."""

from decimal import Decimal

from agentic_trading.core.enums import (
    Exchange,
    MarginMode,
    OrderType,
    OrderStatus,
    PositionSide,
    Side,
    TimeInForce,
)
from agentic_trading.core.events import OrderIntent
from agentic_trading.core.interfaces import PortfolioState
from agentic_trading.core.models import Balance, Fill, Position
from agentic_trading.risk.pre_trade import PreTradeChecker
from agentic_trading.risk.post_trade import PostTradeChecker

from datetime import datetime, timezone


def _make_intent(
    qty: Decimal = Decimal("1.0"),
    price: Decimal = Decimal("67000"),
) -> OrderIntent:
    return OrderIntent(
        dedupe_key="test-001",
        strategy_id="test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        qty=qty,
        price=price,
    )


def _make_portfolio(
    equity: Decimal = Decimal("100000"),
    gross_notional: Decimal = Decimal("0"),
) -> PortfolioState:
    balances = {"USDT": Balance(
        currency="USDT",
        exchange=Exchange.BINANCE,
        total=equity,
        free=equity,
        used=Decimal("0"),
    )}
    positions = {}
    if gross_notional > Decimal("0"):
        positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=PositionSide.LONG,
            qty=Decimal("1"),
            entry_price=Decimal("67000"),
            mark_price=Decimal("67000"),
            notional=gross_notional,
        )
    return PortfolioState(positions=positions, balances=balances)


class TestPreTradeChecker:
    def test_small_order_passes(self):
        checker = PreTradeChecker(max_position_pct=0.10)
        intent = _make_intent(qty=Decimal("0.01"), price=Decimal("67000"))
        portfolio = _make_portfolio(equity=Decimal("100000"))
        results = checker.check(intent, portfolio)
        # All checks should pass for a small order
        assert all(r.passed for r in results)

    def test_oversized_order_blocked(self):
        checker = PreTradeChecker(max_position_pct=0.05)
        # Order notional: 1.0 * 67000 = 67000 > 5% of 100000 = 5000
        intent = _make_intent(qty=Decimal("1.0"), price=Decimal("67000"))
        portfolio = _make_portfolio(equity=Decimal("100000"))
        results = checker.check(intent, portfolio)
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0
        assert any("max_position_size" in r.check_name for r in failed)

    def test_max_notional_exceeded(self):
        checker = PreTradeChecker(max_notional=10_000.0)
        intent = _make_intent(qty=Decimal("1.0"), price=Decimal("67000"))
        portfolio = _make_portfolio()
        results = checker.check(intent, portfolio)
        notional_checks = [r for r in results if r.check_name == "max_notional"]
        assert len(notional_checks) == 1
        assert not notional_checks[0].passed

    def test_max_leverage_exceeded(self):
        checker = PreTradeChecker(max_portfolio_leverage=1.0)
        # Existing exposure: 150000 + new order: 67000 -> leverage > 1x
        intent = _make_intent(qty=Decimal("1.0"), price=Decimal("67000"))
        portfolio = _make_portfolio(
            equity=Decimal("100000"),
            gross_notional=Decimal("150000"),
        )
        results = checker.check(intent, portfolio)
        leverage_checks = [r for r in results if r.check_name == "max_leverage"]
        assert len(leverage_checks) == 1
        assert not leverage_checks[0].passed


class TestPostTradeChecker:
    def _make_fill(self) -> Fill:
        return Fill(
            fill_id="fill-001",
            order_id="order-001",
            client_order_id="test-001",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            price=Decimal("67000"),
            qty=Decimal("0.01"),
            fee=Decimal("0.27"),
            fee_currency="USDT",
            timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

    def test_normal_fill_passes(self):
        checker = PostTradeChecker()
        fill = self._make_fill()
        portfolio = _make_portfolio(equity=Decimal("100000"))
        # Add the position that was created by the fill
        portfolio.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=PositionSide.LONG,
            qty=Decimal("0.01"),
            entry_price=Decimal("67000"),
            mark_price=Decimal("67000"),
            notional=Decimal("670"),
        )
        results = checker.check(fill, portfolio)
        assert all(r.passed for r in results)

    def test_leverage_spike_detected(self):
        checker = PostTradeChecker(max_leverage_after_fill=2.0)
        fill = self._make_fill()
        # High gross exposure relative to equity => leverage spike
        portfolio = _make_portfolio(equity=Decimal("10000"))
        portfolio.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=PositionSide.LONG,
            qty=Decimal("1"),
            entry_price=Decimal("67000"),
            mark_price=Decimal("67000"),
            notional=Decimal("67000"),  # 67000/10000 = 6.7x leverage
        )
        results = checker.check(fill, portfolio)
        spike_checks = [r for r in results if r.check_name == "leverage_spike"]
        assert len(spike_checks) == 1
        assert not spike_checks[0].passed

    def test_missing_position_flagged(self):
        checker = PostTradeChecker()
        fill = self._make_fill()
        portfolio = _make_portfolio()
        # No position for the symbol
        results = checker.check(fill, portfolio)
        consistency_checks = [r for r in results if r.check_name == "position_consistency"]
        assert len(consistency_checks) == 1
        assert not consistency_checks[0].passed
