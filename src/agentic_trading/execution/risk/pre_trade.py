"""Pre-trade risk checks.

Every :class:`OrderIntent` passes through :class:`PreTradeChecker` before
reaching the exchange adapter.  Each check produces a
:class:`~agentic_trading.core.events.RiskCheckResult`; the first failure
short-circuits the pipeline and the order is rejected.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import PositionSide, Side
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
        max_concurrent_positions: int = 4,
        max_daily_entries: int = 10,
        portfolio_cooldown_seconds: int = 3600,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_notional = Decimal(str(max_notional))
        self.max_portfolio_leverage = max_portfolio_leverage
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.instruments = instruments or {}
        self.max_concurrent_positions = max_concurrent_positions
        self.max_daily_entries = max_daily_entries
        self.portfolio_cooldown_seconds = portfolio_cooldown_seconds

        # Stateful tracking for rate-limiting
        self._daily_entry_count: dict[str, int] = {}  # date_str → count
        self._last_entry_time: datetime | None = None

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
            self._check_position_direction_conflict(intent, portfolio),
            self._check_max_concurrent_positions(intent, portfolio),
            self._check_daily_entry_limit(intent),
            self._check_portfolio_cooldown(intent),
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

    def _check_position_direction_conflict(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Block orders that would trade AGAINST an existing position.

        When strategy A is long BTC and strategy B fires a short signal
        on BTC, the short is blocked.  This prevents contradictory
        strategies from partially cancelling each other and churning the
        account.

        ``reduce_only`` orders are exempt (they are intentional closes).
        """
        # reduce_only orders are intentional closes — always allow
        if getattr(intent, "reduce_only", False):
            return self._pass("position_direction_conflict", intent, {
                "reason": "reduce_only_exempt",
            })

        existing_pos = portfolio.get_position(intent.symbol)
        if existing_pos is None or not existing_pos.is_open:
            return self._pass("position_direction_conflict", intent, {
                "reason": "no_existing_position",
            })

        # Map order side to position direction
        order_direction = (
            PositionSide.LONG if intent.side == Side.BUY else PositionSide.SHORT
        )
        existing_direction = existing_pos.side

        # BOTH means one-way mode — no directional conflict possible
        if existing_direction == PositionSide.BOTH:
            return self._pass("position_direction_conflict", intent, {
                "reason": "one_way_mode",
            })

        if order_direction != existing_direction:
            return self._fail(
                "position_direction_conflict",
                intent,
                f"Order would {intent.side.value} {intent.symbol} but "
                f"existing position is {existing_direction.value} "
                f"(qty={existing_pos.qty}). Conflicting strategies "
                f"are not allowed on the same symbol.",
                {
                    "order_side": intent.side.value,
                    "existing_side": existing_direction.value,
                    "existing_qty": float(existing_pos.qty),
                },
            )

        return self._pass("position_direction_conflict", intent, {
            "order_side": intent.side.value,
            "existing_side": existing_direction.value,
        })

    def _check_max_concurrent_positions(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Block new entries when max concurrent positions is reached."""
        # Only applies to new entries, not closes/reduces
        if getattr(intent, "reduce_only", False):
            return self._pass("max_concurrent_positions", intent, {
                "reason": "reduce_only_exempt",
            })

        open_count = sum(
            1 for p in portfolio.positions.values()
            if p.is_open
        )
        if open_count >= self.max_concurrent_positions:
            return self._fail(
                "max_concurrent_positions",
                intent,
                f"Already {open_count} open positions "
                f"(max {self.max_concurrent_positions})",
                {
                    "open_positions": open_count,
                    "max_concurrent": self.max_concurrent_positions,
                },
            )
        return self._pass("max_concurrent_positions", intent, {
            "open_positions": open_count,
        })

    def _check_daily_entry_limit(
        self,
        intent: OrderIntent,
    ) -> RiskCheckResult:
        """Block new entries when daily trade count is exceeded."""
        if getattr(intent, "reduce_only", False):
            return self._pass("daily_entry_limit", intent, {
                "reason": "reduce_only_exempt",
            })

        now = datetime.now(timezone.utc)
        date_key = now.strftime("%Y-%m-%d")
        current_count = self._daily_entry_count.get(date_key, 0)

        if current_count >= self.max_daily_entries:
            return self._fail(
                "daily_entry_limit",
                intent,
                f"Daily entry limit reached: {current_count} "
                f"(max {self.max_daily_entries})",
                {
                    "daily_entries": current_count,
                    "max_daily_entries": self.max_daily_entries,
                },
            )
        return self._pass("daily_entry_limit", intent, {
            "daily_entries": current_count,
        })

    def _check_portfolio_cooldown(
        self,
        intent: OrderIntent,
    ) -> RiskCheckResult:
        """Enforce minimum time between entries across the portfolio."""
        if getattr(intent, "reduce_only", False):
            return self._pass("portfolio_cooldown", intent, {
                "reason": "reduce_only_exempt",
            })

        if self._last_entry_time is not None:
            now = datetime.now(timezone.utc)
            elapsed = (now - self._last_entry_time).total_seconds()
            if elapsed < self.portfolio_cooldown_seconds:
                return self._fail(
                    "portfolio_cooldown",
                    intent,
                    f"Portfolio cooldown active: {elapsed:.0f}s "
                    f"(need {self.portfolio_cooldown_seconds}s)",
                    {
                        "elapsed_seconds": elapsed,
                        "cooldown_seconds": self.portfolio_cooldown_seconds,
                    },
                )
        return self._pass("portfolio_cooldown", intent)

    def record_entry(self) -> None:
        """Call after a successful order fill to update rate-limit state."""
        now = datetime.now(timezone.utc)
        date_key = now.strftime("%Y-%m-%d")
        self._daily_entry_count[date_key] = self._daily_entry_count.get(date_key, 0) + 1
        self._last_entry_time = now

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
