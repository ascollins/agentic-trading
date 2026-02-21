"""Integration test: pre-trade safety checks.

Exercises PreTradeChecker for price collars, self-match prevention,
and message throttle rate limiting.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from agentic_trading.core.enums import (
    Exchange,
    OrderType,
    PositionSide,
    Side,
    TimeInForce,
)
from agentic_trading.core.events import OrderIntent, RiskCheckResult
from agentic_trading.core.interfaces import PortfolioState
from agentic_trading.core.models import Balance, Position
from agentic_trading.execution.risk.pre_trade import PreTradeChecker


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_intent(
    *,
    side: Side = Side.BUY,
    price: Decimal | None = Decimal("50000"),
    qty: Decimal = Decimal("0.1"),
    symbol: str = "BTC/USDT",
    strategy_id: str = "trend_following",
    order_type: OrderType = OrderType.LIMIT,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        order_type=order_type,
        qty=qty,
        price=price,
        time_in_force=TimeInForce.GTC,
        dedupe_key="test-dedupe",
        strategy_id=strategy_id,
    )


def _make_portfolio(
    *,
    positions: dict[str, Position] | None = None,
    balances: dict[str, Balance] | None = None,
) -> PortfolioState:
    return PortfolioState(
        positions=positions or {},
        balances=balances or {
            "USDT": Balance(
                currency="USDT",
                exchange=Exchange.BYBIT,
                total=Decimal("100000"),
                free=Decimal("100000"),
                used=Decimal("0"),
            ),
        },
    )


def _make_resting_order(
    *,
    symbol: str = "BTC/USDT",
    side: str = "sell",
    price: float = 50000.0,
) -> SimpleNamespace:
    """Lightweight resting order stub for self-match prevention."""
    return SimpleNamespace(symbol=symbol, side=side, price=price)


def _find_result(results: list[RiskCheckResult], check_name: str) -> RiskCheckResult:
    """Find a specific check result by name."""
    for r in results:
        if r.check_name == check_name:
            return r
    raise ValueError(f"No result for check '{check_name}'")


# ---------------------------------------------------------------------------
# Price collar tests
# ---------------------------------------------------------------------------


class TestPriceCollar:
    def test_price_collar_passes_within_band(self):
        """Limit order within collar_bps of mark price passes."""
        checker = PreTradeChecker(price_collar_bps=200.0)

        # Position with mark_price=50000 as reference
        portfolio = _make_portfolio(
            positions={
                "BTC/USDT": Position(
                    symbol="BTC/USDT",
                    exchange=Exchange.BYBIT,
                    side=PositionSide.LONG,
                    qty=Decimal("0.1"),
                    entry_price=Decimal("49000"),
                    mark_price=Decimal("50000"),
                    notional=Decimal("5000"),
                ),
            },
        )

        # Price 50500 is 100 bps from 50000 — within 200 bps collar
        intent = _make_intent(price=Decimal("50500"))
        results = checker.check(intent, portfolio)
        collar = _find_result(results, "price_collar")
        assert collar.passed is True

    def test_price_collar_rejects_outside_band(self):
        """Limit price deviating > collar_bps from mark is rejected."""
        checker = PreTradeChecker(price_collar_bps=100.0)

        portfolio = _make_portfolio(
            positions={
                "BTC/USDT": Position(
                    symbol="BTC/USDT",
                    exchange=Exchange.BYBIT,
                    side=PositionSide.LONG,
                    qty=Decimal("0.1"),
                    entry_price=Decimal("49000"),
                    mark_price=Decimal("50000"),
                    notional=Decimal("5000"),
                ),
            },
        )

        # Price 51000 is 200 bps from 50000 — exceeds 100 bps collar
        intent = _make_intent(price=Decimal("51000"))
        results = checker.check(intent, portfolio)
        collar = _find_result(results, "price_collar")
        assert collar.passed is False
        assert "collar" in collar.reason.lower() or "bps" in collar.reason.lower()

    def test_price_collar_skips_market_orders(self):
        """Market orders (price=None) bypass collar check."""
        checker = PreTradeChecker(price_collar_bps=10.0)

        portfolio = _make_portfolio(
            positions={
                "BTC/USDT": Position(
                    symbol="BTC/USDT",
                    exchange=Exchange.BYBIT,
                    side=PositionSide.LONG,
                    qty=Decimal("0.1"),
                    entry_price=Decimal("49000"),
                    mark_price=Decimal("50000"),
                    notional=Decimal("5000"),
                ),
            },
        )

        intent = _make_intent(price=None, order_type=OrderType.MARKET)
        results = checker.check(intent, portfolio)
        collar = _find_result(results, "price_collar")
        assert collar.passed is True

    def test_price_collar_passes_when_no_reference_price(self):
        """When no position/mark price exists, collar check passes."""
        checker = PreTradeChecker(price_collar_bps=10.0)

        # No positions → no reference price
        portfolio = _make_portfolio()

        intent = _make_intent(price=Decimal("99999"))
        results = checker.check(intent, portfolio)
        collar = _find_result(results, "price_collar")
        assert collar.passed is True


# ---------------------------------------------------------------------------
# Self-match prevention tests
# ---------------------------------------------------------------------------


class TestSelfMatchPrevention:
    def test_self_match_blocks_crossing_buy(self):
        """Buy order at or above a resting sell on same symbol is blocked."""
        checker = PreTradeChecker()
        portfolio = _make_portfolio()
        open_orders = [_make_resting_order(side="sell", price=50000.0)]

        # Buy at 50000 crosses the resting sell at 50000
        intent = _make_intent(side=Side.BUY, price=Decimal("50000"))
        results = checker.check(intent, portfolio, open_orders)
        sm = _find_result(results, "self_match_prevention")
        assert sm.passed is False
        assert "self-match" in sm.reason.lower()

    def test_self_match_blocks_crossing_sell(self):
        """Sell order at or below a resting buy on same symbol is blocked."""
        checker = PreTradeChecker()
        portfolio = _make_portfolio()
        open_orders = [_make_resting_order(side="buy", price=50000.0)]

        # Sell at 50000 crosses the resting buy at 50000
        intent = _make_intent(side=Side.SELL, price=Decimal("50000"))
        results = checker.check(intent, portfolio, open_orders)
        sm = _find_result(results, "self_match_prevention")
        assert sm.passed is False

    def test_self_match_allows_same_side(self):
        """Same-side orders on same symbol do not trigger self-match."""
        checker = PreTradeChecker()
        portfolio = _make_portfolio()
        open_orders = [_make_resting_order(side="buy", price=49000.0)]

        # Buy intent won't cross a resting buy
        intent = _make_intent(side=Side.BUY, price=Decimal("50000"))
        results = checker.check(intent, portfolio, open_orders)
        sm = _find_result(results, "self_match_prevention")
        assert sm.passed is True

    def test_self_match_allows_different_symbol(self):
        """Crossing order on a different symbol is not self-match."""
        checker = PreTradeChecker()
        portfolio = _make_portfolio()
        open_orders = [_make_resting_order(symbol="ETH/USDT", side="sell", price=3000.0)]

        # BTC/USDT buy won't match ETH/USDT sell
        intent = _make_intent(side=Side.BUY, price=Decimal("50000"))
        results = checker.check(intent, portfolio, open_orders)
        sm = _find_result(results, "self_match_prevention")
        assert sm.passed is True

    def test_self_match_passes_with_no_open_orders(self):
        """No resting orders means self-match passes."""
        checker = PreTradeChecker()
        portfolio = _make_portfolio()

        intent = _make_intent()
        results = checker.check(intent, portfolio, open_orders=[])
        sm = _find_result(results, "self_match_prevention")
        assert sm.passed is True


# ---------------------------------------------------------------------------
# Message throttle tests
# ---------------------------------------------------------------------------


class TestMessageThrottle:
    def test_message_throttle_per_strategy_blocks_excess(self):
        """Exceeding max_messages_per_minute_per_strategy triggers rejection."""
        checker = PreTradeChecker(
            max_messages_per_minute_per_strategy=3,
            max_messages_per_minute_per_symbol=100,
        )
        portfolio = _make_portfolio()

        # First 3 calls should pass
        for _ in range(3):
            results = checker.check(
                _make_intent(strategy_id="strat_a"), portfolio,
            )
            throttle = _find_result(results, "message_throttle")
            assert throttle.passed is True

        # 4th call exceeds limit
        results = checker.check(
            _make_intent(strategy_id="strat_a"), portfolio,
        )
        throttle = _find_result(results, "message_throttle")
        assert throttle.passed is False
        assert "per_strategy" in (throttle.details or {}).get("throttle_type", "")

    def test_message_throttle_per_symbol_blocks_excess(self):
        """Exceeding max_messages_per_minute_per_symbol triggers rejection."""
        checker = PreTradeChecker(
            max_messages_per_minute_per_strategy=100,
            max_messages_per_minute_per_symbol=2,
        )
        portfolio = _make_portfolio()

        # First 2 calls should pass
        for i in range(2):
            results = checker.check(
                _make_intent(strategy_id=f"strat_{i}"), portfolio,
            )
            throttle = _find_result(results, "message_throttle")
            assert throttle.passed is True

        # 3rd call exceeds symbol limit
        results = checker.check(
            _make_intent(strategy_id="strat_x"), portfolio,
        )
        throttle = _find_result(results, "message_throttle")
        assert throttle.passed is False
        assert "per_symbol" in (throttle.details or {}).get("throttle_type", "")

    def test_message_throttle_allows_under_limit(self):
        """Messages under the threshold pass."""
        checker = PreTradeChecker(
            max_messages_per_minute_per_strategy=10,
            max_messages_per_minute_per_symbol=10,
        )
        portfolio = _make_portfolio()

        results = checker.check(_make_intent(), portfolio)
        throttle = _find_result(results, "message_throttle")
        assert throttle.passed is True
