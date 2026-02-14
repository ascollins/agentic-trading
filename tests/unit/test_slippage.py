"""Test FixedBPSSlippage, VolatilityBasedSlippage models produce expected adjustments."""

import pytest

from agentic_trading.backtester.slippage import (
    FixedBPSSlippage,
    VolatilityBasedSlippage,
    create_slippage_model,
)


class TestFixedBPSSlippage:
    def test_buy_price_increases(self):
        model = FixedBPSSlippage(bps=5.0, seed=42)
        slipped = model.compute_slippage(price=67000.0, qty=1.0, is_buy=True)
        assert slipped > 67000.0

    def test_sell_price_decreases(self):
        model = FixedBPSSlippage(bps=5.0, seed=42)
        slipped = model.compute_slippage(price=67000.0, qty=1.0, is_buy=False)
        assert slipped < 67000.0

    def test_deterministic_with_seed(self):
        m1 = FixedBPSSlippage(bps=5.0, seed=42)
        m2 = FixedBPSSlippage(bps=5.0, seed=42)
        p1 = m1.compute_slippage(67000.0, 1.0, True)
        p2 = m2.compute_slippage(67000.0, 1.0, True)
        assert p1 == p2

    def test_slippage_magnitude_reasonable(self):
        model = FixedBPSSlippage(bps=5.0, seed=42)
        slipped = model.compute_slippage(67000.0, 1.0, True)
        # 5 bps = 0.05%, so slippage should be around $33.50
        # With jitter (0.8-1.2x), range: ~$26.8 to ~$40.2
        diff = slipped - 67000.0
        assert 20 < diff < 50

    def test_zero_bps(self):
        model = FixedBPSSlippage(bps=0.0, seed=42)
        slipped = model.compute_slippage(67000.0, 1.0, True)
        # With 0 bps, slippage should be very small (only jitter * 0)
        assert abs(slipped - 67000.0) < 0.01


class TestVolatilityBasedSlippage:
    def test_buy_price_increases(self):
        model = VolatilityBasedSlippage(base_bps=2.0, vol_multiplier=0.5, seed=42)
        slipped = model.compute_slippage(67000.0, 1.0, True, atr=500.0)
        assert slipped > 67000.0

    def test_sell_price_decreases(self):
        model = VolatilityBasedSlippage(base_bps=2.0, vol_multiplier=0.5, seed=42)
        slipped = model.compute_slippage(67000.0, 1.0, False, atr=500.0)
        assert slipped < 67000.0

    def test_higher_atr_more_slippage(self):
        model_low = VolatilityBasedSlippage(base_bps=2.0, vol_multiplier=0.5, seed=42)
        model_high = VolatilityBasedSlippage(base_bps=2.0, vol_multiplier=0.5, seed=42)
        slip_low = model_low.compute_slippage(67000.0, 1.0, True, atr=100.0)
        slip_high = model_high.compute_slippage(67000.0, 1.0, True, atr=2000.0)
        assert slip_high > slip_low

    def test_zero_atr_uses_fallback(self):
        model = VolatilityBasedSlippage(base_bps=2.0, vol_multiplier=0.5, seed=42)
        # With atr=0, should use fallback (0.1% of price)
        slipped = model.compute_slippage(67000.0, 1.0, True, atr=0.0)
        assert slipped > 67000.0

    def test_deterministic_with_seed(self):
        m1 = VolatilityBasedSlippage(seed=42)
        m2 = VolatilityBasedSlippage(seed=42)
        p1 = m1.compute_slippage(67000.0, 1.0, True, atr=500.0)
        p2 = m2.compute_slippage(67000.0, 1.0, True, atr=500.0)
        assert p1 == p2


class TestSlippageFactory:
    def test_create_fixed_bps(self):
        model = create_slippage_model("fixed_bps", seed=42, bps=10.0)
        assert isinstance(model, FixedBPSSlippage)

    def test_create_volatility_based(self):
        model = create_slippage_model("volatility_based", seed=42)
        assert isinstance(model, VolatilityBasedSlippage)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown slippage model"):
            create_slippage_model("nonexistent_model")
