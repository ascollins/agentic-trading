"""Fee models for backtesting.

Handles maker/taker fees and funding rate simulation for perpetual futures.
"""

from __future__ import annotations

from decimal import Decimal


class FeeModel:
    """Computes trading fees."""

    def __init__(
        self,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
    ) -> None:
        self._maker = maker_fee
        self._taker = taker_fee

    def compute_fee(
        self,
        price: float,
        qty: float,
        is_maker: bool = False,
    ) -> Decimal:
        """Compute fee for a trade.

        Returns absolute fee amount (in quote currency).
        """
        rate = self._maker if is_maker else self._taker
        notional = abs(price * qty)
        return Decimal(str(notional * rate))


class FundingModel:
    """Simulates funding rate payments for perpetual futures.

    Funding is exchanged every funding_interval hours.
    Positive rate: longs pay shorts.
    Negative rate: shorts pay longs.
    """

    def __init__(self, funding_interval_hours: int = 8) -> None:
        self._interval_hours = funding_interval_hours
        self._funding_rates: dict[str, list[float]] = {}

    def set_funding_rates(self, symbol: str, rates: list[float]) -> None:
        """Set historical funding rates for a symbol."""
        self._funding_rates[symbol] = list(rates)

    def get_funding_rate(self, symbol: str, period_index: int) -> float:
        """Get funding rate for a specific period.

        Returns 0.0 if no data available.
        """
        rates = self._funding_rates.get(symbol, [])
        if not rates or period_index < 0:
            return 0.0
        idx = period_index % len(rates)
        return rates[idx]

    def compute_funding_payment(
        self,
        symbol: str,
        position_qty: float,
        mark_price: float,
        period_index: int,
    ) -> Decimal:
        """Compute funding payment for a position.

        Returns:
            Positive = received (favorable), Negative = paid.
        """
        rate = self.get_funding_rate(symbol, period_index)
        # Long positions pay when rate > 0
        # Short positions receive when rate > 0
        notional = position_qty * mark_price
        payment = -notional * rate  # Negative for longs when rate > 0
        return Decimal(str(payment))

    def hours_to_periods(self, hours: float) -> int:
        """Convert hours to number of funding periods."""
        return int(hours / self._interval_hours)
