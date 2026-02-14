"""Test FeeModel computes maker/taker fees correctly."""

from decimal import Decimal

from agentic_trading.backtester.fee_model import FeeModel, FundingModel


class TestFeeModel:
    def test_taker_fee(self):
        fm = FeeModel(maker_fee=0.0002, taker_fee=0.0004)
        fee = fm.compute_fee(price=67000.0, qty=1.0, is_maker=False)
        # 67000 * 1 * 0.0004 = 26.80
        assert fee == Decimal("26.8")

    def test_maker_fee(self):
        fm = FeeModel(maker_fee=0.0002, taker_fee=0.0004)
        fee = fm.compute_fee(price=67000.0, qty=1.0, is_maker=True)
        # 67000 * 1 * 0.0002 = 13.40
        assert fee == Decimal("13.4")

    def test_maker_less_than_taker(self):
        fm = FeeModel(maker_fee=0.0002, taker_fee=0.0004)
        maker = fm.compute_fee(67000.0, 1.0, is_maker=True)
        taker = fm.compute_fee(67000.0, 1.0, is_maker=False)
        assert maker < taker

    def test_zero_qty_gives_zero_fee(self):
        fm = FeeModel()
        fee = fm.compute_fee(67000.0, 0.0, is_maker=False)
        assert fee == Decimal("0.0")

    def test_small_trade_fee(self):
        fm = FeeModel(maker_fee=0.0002, taker_fee=0.0004)
        fee = fm.compute_fee(price=67000.0, qty=0.001, is_maker=False)
        # 67000 * 0.001 * 0.0004 = 0.0268
        expected = Decimal(str(67000.0 * 0.001 * 0.0004))
        assert fee == expected

    def test_custom_fee_rates(self):
        fm = FeeModel(maker_fee=0.001, taker_fee=0.002)
        fee = fm.compute_fee(10000.0, 2.0, is_maker=False)
        # 10000 * 2 * 0.002 = 40
        assert fee == Decimal("40.0")

    def test_returns_decimal(self):
        fm = FeeModel()
        fee = fm.compute_fee(67000.0, 1.0, is_maker=True)
        assert isinstance(fee, Decimal)


class TestFundingModel:
    def test_compute_funding_payment_long(self):
        fm = FundingModel()
        fm.set_funding_rates("BTC/USDT", [0.0001])
        payment = fm.compute_funding_payment(
            symbol="BTC/USDT",
            position_qty=1.0,  # Long
            mark_price=67000.0,
            period_index=0,
        )
        # Long pays positive rate: payment = -1.0 * 67000 * 0.0001 = -6.7
        assert payment == Decimal("-6.7")

    def test_no_rates_returns_zero(self):
        fm = FundingModel()
        payment = fm.compute_funding_payment("ETH/USDT", 1.0, 2000.0, 0)
        assert payment == Decimal("0")

    def test_period_index_wraps(self):
        fm = FundingModel()
        fm.set_funding_rates("BTC/USDT", [0.0001, 0.0002, 0.0003])
        # period_index=5 -> 5 % 3 = 2 -> rate = 0.0003
        payment = fm.compute_funding_payment("BTC/USDT", 1.0, 10000.0, 5)
        expected = Decimal(str(-1.0 * 10000.0 * 0.0003))
        assert payment == expected

    def test_hours_to_periods(self):
        fm = FundingModel(funding_interval_hours=8)
        assert fm.hours_to_periods(24.0) == 3
        assert fm.hours_to_periods(8.0) == 1
