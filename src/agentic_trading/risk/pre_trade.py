"""Pre-trade risk checks.

Every :class:`OrderIntent` passes through :class:`PreTradeChecker` before
reaching the exchange adapter.  Each check produces a
:class:`~agentic_trading.core.events.RiskCheckResult`; the first failure
short-circuits the pipeline and the order is rejected.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agentic_trading.core.events import OrderIntent, RiskCheckResult
from agentic_trading.core.interfaces import PortfolioState
from agentic_trading.core.models import Instrument

logger = logging.getLogger(__name__)


class PreTradeChecker:
    """Runs a battery of pre-trade risk checks against an :class:`OrderIntent`.

    Configuration is passed as keyword arguments so it can be wired
    directly from :class:`~agentic_trading.core.config.RiskConfig`.

    Args:
        max_position_pct: Maximum single-position size as a fraction of
            total portfolio equity (e.g. 0.10 for 10%).
        max_notional: Absolute cap on a single order's notional value
            in quote currency.
        max_portfolio_leverage: Maximum allowed gross portfolio leverage
            (e.g. 3.0 for 3x).
        max_gross_exposure_pct: Maximum gross exposure as a multiple of
            capital (synonymous with leverage for fully-collateralized
            portfolios).
        instruments: Optional mapping of symbol -> :class:`Instrument`
            used for per-instrument limit checks (min_qty, min_notional).
    """

    def __init__(
        self,
        *,
        max_position_pct: float = 0.10,
        max_notional: float = 500_000.0,
        max_portfolio_leverage: float = 3.0,
        max_gross_exposure_pct: float = 3.0,
        instruments: dict[str, Instrument] | None = None,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_notional = Decimal(str(max_notional))
        self.max_portfolio_leverage = max_portfolio_leverage
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.instruments = instruments or {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def check(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
    ) -> list[RiskCheckResult]:
        """Run all pre-trade checks.

        Args:
            intent: The order the strategy wants to place.
            portfolio: Current portfolio snapshot.

        Returns:
            A list of :class:`RiskCheckResult`, one per check.
            If every result has ``passed=True`` the order is safe to send.
        """
        results: list[RiskCheckResult] = [
            self._check_max_position_size(intent, portfolio),
            self._check_max_notional(intent),
            self._check_max_leverage(intent, portfolio),
            self._check_exposure_limits(intent, portfolio),
            self._check_instrument_limits(intent),
        ]

        for r in results:
            if not r.passed:
                logger.warning(
                    "Pre-trade check FAILED [%s]: %s  (order=%s symbol=%s)",
                    r.check_name,
                    r.reason,
                    intent.dedupe_key,
                    intent.symbol,
                )
        return results

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_max_position_size(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Ensure that the resulting position does not exceed max_position_pct
        of total portfolio equity."""
        total_equity = self._total_equity(portfolio)
        if total_equity <= 0:
            return self._pass("max_position_size", intent, {"total_equity": 0})

        # Estimate resulting notional
        price = intent.price or self._last_price_estimate(intent, portfolio)
        if price is None:
            # Cannot estimate -- pass conservatively and rely on post-trade
            return self._pass(
                "max_position_size",
                intent,
                {"reason": "no_price_estimate"},
            )

        existing_pos = portfolio.get_position(intent.symbol)
        existing_notional = abs(existing_pos.notional) if existing_pos else Decimal("0")
        order_notional = intent.qty * price
        resulting_notional = existing_notional + order_notional

        max_allowed = Decimal(str(self.max_position_pct)) * total_equity
        if resulting_notional > max_allowed:
            return self._fail(
                "max_position_size",
                intent,
                f"Resulting notional {resulting_notional:.2f} exceeds "
                f"max {max_allowed:.2f} ({self.max_position_pct:.0%} of "
                f"{total_equity:.2f})",
                {
                    "resulting_notional": float(resulting_notional),
                    "max_allowed": float(max_allowed),
                    "total_equity": float(total_equity),
                },
            )
        return self._pass("max_position_size", intent, {
            "resulting_notional": float(resulting_notional),
            "max_allowed": float(max_allowed),
        })

    def _check_max_notional(self, intent: OrderIntent) -> RiskCheckResult:
        """Ensure the order notional does not exceed the absolute cap."""
        price = intent.price or Decimal("0")
        order_notional = intent.qty * price if price else Decimal("0")

        if price and order_notional > self.max_notional:
            return self._fail(
                "max_notional",
                intent,
                f"Order notional {order_notional:.2f} exceeds "
                f"max {self.max_notional:.2f}",
                {
                    "order_notional": float(order_notional),
                    "max_notional": float(self.max_notional),
                },
            )
        return self._pass("max_notional", intent, {
            "order_notional": float(order_notional),
        })

    def _check_max_leverage(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Ensure adding this order would not push portfolio leverage above limit."""
        total_equity = self._total_equity(portfolio)
        if total_equity <= 0:
            return self._pass("max_leverage", intent, {"total_equity": 0})

        current_gross = portfolio.gross_exposure
        price = intent.price or self._last_price_estimate(intent, portfolio)
        additional = intent.qty * price if price else Decimal("0")
        projected_gross = current_gross + additional
        projected_leverage = float(projected_gross / total_equity)

        if projected_leverage > self.max_portfolio_leverage:
            return self._fail(
                "max_leverage",
                intent,
                f"Projected leverage {projected_leverage:.2f}x exceeds "
                f"max {self.max_portfolio_leverage:.2f}x",
                {
                    "projected_leverage": projected_leverage,
                    "max_leverage": self.max_portfolio_leverage,
                    "current_gross": float(current_gross),
                    "additional": float(additional),
                },
            )
        return self._pass("max_leverage", intent, {
            "projected_leverage": projected_leverage,
        })

    def _check_exposure_limits(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Ensure gross exposure stays within the allowed multiple of capital."""
        total_equity = self._total_equity(portfolio)
        if total_equity <= 0:
            return self._pass("exposure_limits", intent, {"total_equity": 0})

        current_gross = portfolio.gross_exposure
        price = intent.price or self._last_price_estimate(intent, portfolio)
        additional = intent.qty * price if price else Decimal("0")
        projected = current_gross + additional
        ratio = float(projected / total_equity)

        if ratio > self.max_gross_exposure_pct:
            return self._fail(
                "exposure_limits",
                intent,
                f"Projected gross exposure {ratio:.2f}x exceeds "
                f"limit {self.max_gross_exposure_pct:.2f}x",
                {
                    "projected_ratio": ratio,
                    "limit": self.max_gross_exposure_pct,
                },
            )
        return self._pass("exposure_limits", intent, {"projected_ratio": ratio})

    def _check_instrument_limits(self, intent: OrderIntent) -> RiskCheckResult:
        """Enforce per-instrument minimums (min_qty, min_notional)."""
        instrument = self.instruments.get(intent.symbol)
        if instrument is None:
            # No instrument metadata -- pass conservatively
            return self._pass(
                "instrument_limits",
                intent,
                {"reason": "no_instrument_metadata"},
            )

        # Min quantity
        if intent.qty < instrument.min_qty:
            return self._fail(
                "instrument_limits",
                intent,
                f"Qty {intent.qty} below instrument min_qty "
                f"{instrument.min_qty} for {intent.symbol}",
                {
                    "qty": float(intent.qty),
                    "min_qty": float(instrument.min_qty),
                },
            )

        # Min notional
        price = intent.price or Decimal("0")
        if price:
            notional = intent.qty * price
            if notional < instrument.min_notional:
                return self._fail(
                    "instrument_limits",
                    intent,
                    f"Notional {notional:.2f} below instrument min_notional "
                    f"{instrument.min_notional} for {intent.symbol}",
                    {
                        "notional": float(notional),
                        "min_notional": float(instrument.min_notional),
                    },
                )

        return self._pass("instrument_limits", intent)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _total_equity(portfolio: PortfolioState) -> Decimal:
        """Compute total equity from balances (sum of all ``total`` fields)."""
        return sum(
            (b.total for b in portfolio.balances.values()),
            Decimal("0"),
        )

    @staticmethod
    def _last_price_estimate(
        intent: OrderIntent,
        portfolio: PortfolioState,
    ) -> Decimal | None:
        """Try to get a price estimate from the existing position's mark_price."""
        pos = portfolio.get_position(intent.symbol)
        if pos and pos.mark_price > 0:
            return pos.mark_price
        return None

    @staticmethod
    def _pass(
        check_name: str,
        intent: OrderIntent,
        details: dict[str, Any] | None = None,
    ) -> RiskCheckResult:
        return RiskCheckResult(
            check_name=check_name,
            passed=True,
            order_intent_id=intent.event_id,
            details=details or {},
        )

    @staticmethod
    def _fail(
        check_name: str,
        intent: OrderIntent,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> RiskCheckResult:
        return RiskCheckResult(
            check_name=check_name,
            passed=False,
            reason=reason,
            order_intent_id=intent.event_id,
            details=details or {},
        )
