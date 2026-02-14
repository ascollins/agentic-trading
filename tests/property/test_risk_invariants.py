"""Property tests for risk/sizing invariants.

Uses hypothesis to verify:
- Position size is always non-negative after sizing
- Fixed fractional size never exceeds max fraction of capital
- Kelly size clamps to a reasonable range
"""

from decimal import Decimal

from hypothesis import given, settings, strategies as st

from agentic_trading.portfolio.sizing import (
    fixed_fractional_size,
    kelly_size,
    volatility_adjusted_size,
)


@given(
    capital=st.floats(min_value=100, max_value=1e8),
    risk_per_trade=st.floats(min_value=0.001, max_value=0.1),
    atr=st.floats(min_value=0.01, max_value=1000),
    price=st.floats(min_value=0.01, max_value=100000),
)
def test_vol_adjusted_size_non_negative(capital, risk_per_trade, atr, price):
    """Volatility-adjusted size is always >= 0."""
    size = volatility_adjusted_size(capital, risk_per_trade, atr, price)
    assert size >= 0


@given(
    capital=st.floats(min_value=100, max_value=1e8),
    risk_per_trade=st.floats(min_value=0.001, max_value=0.1),
    atr=st.floats(min_value=0.01, max_value=1000),
    price=st.floats(min_value=0.01, max_value=100000),
)
def test_vol_adjusted_size_is_decimal(capital, risk_per_trade, atr, price):
    """Volatility-adjusted size always returns a Decimal."""
    size = volatility_adjusted_size(capital, risk_per_trade, atr, price)
    assert isinstance(size, Decimal)


@given(
    capital=st.floats(min_value=100, max_value=1e8),
    fraction=st.floats(min_value=0.001, max_value=0.5),
    price=st.floats(min_value=0.01, max_value=100000),
)
def test_fixed_fractional_never_exceeds_fraction(capital, fraction, price):
    """Fixed fractional size never exceeds the allocated fraction of capital.

    notional = size * price <= capital * fraction (within floating point tolerance)
    """
    size = fixed_fractional_size(capital, fraction, price)
    assert size >= 0

    notional = float(size) * price
    max_notional = capital * fraction

    # Allow a small tolerance for floating point / Decimal conversion
    assert notional <= max_notional * 1.001, (
        f"notional={notional} exceeds max_notional={max_notional} "
        f"(capital={capital}, fraction={fraction}, price={price})"
    )


@given(
    capital=st.floats(min_value=100, max_value=1e8),
    fraction=st.floats(min_value=0.001, max_value=0.5),
    price=st.floats(min_value=0.01, max_value=100000),
)
def test_fixed_fractional_is_decimal(capital, fraction, price):
    """Fixed fractional size always returns a Decimal."""
    size = fixed_fractional_size(capital, fraction, price)
    assert isinstance(size, Decimal)


@given(
    capital=st.floats(min_value=1000, max_value=1e8),
    win_rate=st.floats(min_value=0.01, max_value=0.99),
    avg_win=st.floats(min_value=0.01, max_value=10000),
    avg_loss=st.floats(min_value=0.01, max_value=10000),
    price=st.floats(min_value=0.01, max_value=100000),
)
def test_kelly_size_non_negative(capital, win_rate, avg_win, avg_loss, price):
    """Kelly size is always >= 0."""
    size = kelly_size(capital, win_rate, avg_win, avg_loss, price)
    assert size >= 0


@given(
    capital=st.floats(min_value=1000, max_value=1e8),
    win_rate=st.floats(min_value=0.01, max_value=0.99),
    avg_win=st.floats(min_value=0.01, max_value=10000),
    avg_loss=st.floats(min_value=0.01, max_value=10000),
    price=st.floats(min_value=0.01, max_value=100000),
)
def test_kelly_size_capped_at_20_pct(capital, win_rate, avg_win, avg_loss, price):
    """Kelly size notional never exceeds 20% of capital (the internal cap).

    The kelly_size function clamps kelly_pct to max 0.20, so the notional
    should never exceed 20% of capital.
    """
    size = kelly_size(capital, win_rate, avg_win, avg_loss, price)
    notional = float(size) * price
    max_allowed = capital * 0.20

    # Allow tolerance for Decimal conversion rounding
    assert notional <= max_allowed * 1.001, (
        f"Kelly notional={notional} exceeds 20% cap={max_allowed} "
        f"(capital={capital}, win_rate={win_rate}, "
        f"avg_win={avg_win}, avg_loss={avg_loss}, price={price})"
    )


@given(
    capital=st.floats(min_value=100, max_value=1e8),
    risk_per_trade=st.floats(min_value=0.001, max_value=0.1),
    price=st.floats(min_value=0.01, max_value=100000),
)
def test_zero_atr_returns_zero(capital, risk_per_trade, price):
    """When ATR is zero or negative, volatility-adjusted size is zero."""
    assert volatility_adjusted_size(capital, risk_per_trade, 0.0, price) == Decimal("0")
    assert volatility_adjusted_size(capital, risk_per_trade, -1.0, price) == Decimal("0")


@given(
    capital=st.floats(min_value=100, max_value=1e8),
    fraction=st.floats(min_value=0.001, max_value=0.5),
)
def test_zero_price_returns_zero(capital, fraction):
    """When price is zero or negative, all sizing methods return zero."""
    assert fixed_fractional_size(capital, fraction, 0.0) == Decimal("0")
    assert fixed_fractional_size(capital, fraction, -1.0) == Decimal("0")
    assert volatility_adjusted_size(capital, 0.01, 10.0, 0.0) == Decimal("0")
    assert kelly_size(capital, 0.5, 100, 100, 0.0) == Decimal("0")
