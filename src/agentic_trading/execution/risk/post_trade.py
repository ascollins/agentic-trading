"""Post-trade sanity checks.

After a fill is received the :class:`PostTradeChecker` verifies that
the resulting portfolio state is consistent and within tolerances.
These checks act as a safety net -- they cannot prevent a bad fill but
they can trigger alerts or a kill switch when something looks wrong.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agentic_trading.core.events import RiskCheckResult
from agentic_trading.core.interfaces import PortfolioState
from agentic_trading.core.models import Fill

logger = logging.getLogger(__name__)


class PostTradeChecker:
    """Runs post-trade sanity checks after a fill arrives.

    Args:
        max_unexpected_loss_pct: Threshold (as fraction of capital) for
            a single fill to be flagged as an unexpected loss.
            Default 0.02 (2%).
        max_leverage_after_fill: Maximum leverage the portfolio may
            have immediately after a fill.  Exceeding this triggers
            a critical alert.  Default 5.0.
        max_fill_deviation_pct: Maximum acceptable deviation between
            the fill price and the position entry / mark price.
            Default 0.05 (5%).
    """

    def __init__(
        self,
        *,
        max_unexpected_loss_pct: float = 0.02,
        max_leverage_after_fill: float = 5.0,
        max_fill_deviation_pct: float = 0.05,
    ) -> None:
        self.max_unexpected_loss_pct = max_unexpected_loss_pct
        self.max_leverage_after_fill = max_leverage_after_fill
        self.max_fill_deviation_pct = max_fill_deviation_pct

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def check(
        self,
        fill: Fill,
        portfolio: PortfolioState,
    ) -> list[RiskCheckResult]:
        """Run all post-trade checks.

        Args:
            fill: The fill that was just received.
            portfolio: Current portfolio snapshot *after* the fill has
                been applied.

        Returns:
            A list of :class:`RiskCheckResult` for each check.
        """
        results: list[RiskCheckResult] = [
            self._check_position_consistency(fill, portfolio),
            self._check_pnl_sanity(fill, portfolio),
            self._check_leverage_spike(portfolio),
            self._check_fill_price_deviation(fill, portfolio),
        ]

        for r in results:
            if not r.passed:
                logger.warning(
                    "Post-trade check FAILED [%s]: %s  (fill=%s symbol=%s)",
                    r.check_name,
                    r.reason,
                    fill.fill_id,
                    fill.symbol,
                )
        return results

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_position_consistency(
        self,
        fill: Fill,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Verify the position for this symbol exists and has non-zero qty
        (unless the fill fully closed it)."""
        pos = portfolio.get_position(fill.symbol)

        if pos is None:
            # Position missing from state after a fill is suspicious
            return self._fail(
                "position_consistency",
                fill,
                f"No position found for {fill.symbol} after fill "
                f"{fill.fill_id}; state may be stale",
                {"symbol": fill.symbol},
            )

        # If position qty is 0, the fill closed it -- that is fine
        return self._pass("position_consistency", fill, {
            "symbol": fill.symbol,
            "position_qty": float(pos.qty),
        })

    def _check_pnl_sanity(
        self,
        fill: Fill,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Flag fills that cause an unreasonably large single-trade loss."""
        total_equity = self._total_equity(portfolio)
        if total_equity <= 0:
            return self._pass("pnl_sanity", fill, {"total_equity": 0})

        pos = portfolio.get_position(fill.symbol)
        if pos is None:
            return self._pass("pnl_sanity", fill, {"reason": "no_position"})

        # Determine if this fill is opening or closing the position.
        # A closing fill has the opposite side from the position:
        #   - Long position closed by a sell
        #   - Short position closed by a buy
        fill_side_str = (
            fill.side.value if hasattr(fill.side, "value") else str(fill.side)
        ).lower()
        pos_side_str = (
            pos.side.value if hasattr(pos.side, "value") else str(pos.side)
        ).lower()

        is_closing = (
            (pos_side_str == "long" and fill_side_str == "sell")
            or (pos_side_str == "short" and fill_side_str == "buy")
        )

        if not is_closing:
            # Opening fills have no realized PnL to check
            return self._pass("pnl_sanity", fill, {
                "reason": "opening_fill",
                "position_side": pos_side_str,
                "fill_side": fill_side_str,
            })

        # For closing fills, PnL depends on position direction:
        #   Long close (sell):  PnL = (fill_price - entry_price) * qty
        #   Short close (buy):  PnL = (entry_price - fill_price) * qty
        if pos_side_str == "long":
            estimated_pnl = (fill.price - pos.entry_price) * fill.qty
        else:
            estimated_pnl = (pos.entry_price - fill.price) * fill.qty

        loss_pct = float(-estimated_pnl / total_equity) if estimated_pnl < 0 else 0.0

        if loss_pct >= self.max_unexpected_loss_pct:
            return self._fail(
                "pnl_sanity",
                fill,
                f"Fill caused estimated loss of {loss_pct:.2%} of equity "
                f"(threshold {self.max_unexpected_loss_pct:.2%})",
                {
                    "estimated_pnl": float(estimated_pnl),
                    "loss_pct": loss_pct,
                    "threshold": self.max_unexpected_loss_pct,
                    "position_side": pos_side_str,
                },
            )
        return self._pass("pnl_sanity", fill, {
            "estimated_pnl": float(estimated_pnl),
            "loss_pct": loss_pct,
            "position_side": pos_side_str,
        })

    def _check_leverage_spike(
        self,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Verify portfolio leverage has not spiked above the post-trade ceiling."""
        total_equity = self._total_equity(portfolio)
        if total_equity <= 0:
            return RiskCheckResult(
                check_name="leverage_spike",
                passed=True,
                details={"total_equity": 0},
            )

        gross = portfolio.gross_exposure
        leverage = float(gross / total_equity)

        if leverage > self.max_leverage_after_fill:
            return RiskCheckResult(
                check_name="leverage_spike",
                passed=False,
                reason=(
                    f"Portfolio leverage {leverage:.2f}x exceeds post-trade "
                    f"ceiling {self.max_leverage_after_fill:.2f}x"
                ),
                details={
                    "leverage": leverage,
                    "ceiling": self.max_leverage_after_fill,
                    "gross": float(gross),
                    "equity": float(total_equity),
                },
            )
        return RiskCheckResult(
            check_name="leverage_spike",
            passed=True,
            details={"leverage": leverage},
        )

    def _check_fill_price_deviation(
        self,
        fill: Fill,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Flag fills whose price deviates significantly from the mark price,
        which may indicate stale pricing or a flash crash."""
        pos = portfolio.get_position(fill.symbol)
        if pos is None or pos.mark_price <= 0:
            return self._pass("fill_price_deviation", fill, {
                "reason": "no_mark_price",
            })

        deviation = abs(float(fill.price - pos.mark_price)) / float(pos.mark_price)

        if deviation >= self.max_fill_deviation_pct:
            return self._fail(
                "fill_price_deviation",
                fill,
                f"Fill price {fill.price} deviates {deviation:.2%} from "
                f"mark {pos.mark_price} (threshold "
                f"{self.max_fill_deviation_pct:.2%})",
                {
                    "fill_price": float(fill.price),
                    "mark_price": float(pos.mark_price),
                    "deviation_pct": deviation,
                    "threshold": self.max_fill_deviation_pct,
                },
            )
        return self._pass("fill_price_deviation", fill, {
            "deviation_pct": deviation,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _total_equity(portfolio: PortfolioState) -> Decimal:
        return sum(
            (b.total for b in portfolio.balances.values()),
            Decimal("0"),
        )

    @staticmethod
    def _pass(
        check_name: str,
        fill: Fill,
        details: dict[str, Any] | None = None,
    ) -> RiskCheckResult:
        return RiskCheckResult(
            check_name=check_name,
            passed=True,
            details=details or {},
        )

    @staticmethod
    def _fail(
        check_name: str,
        fill: Fill,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> RiskCheckResult:
        return RiskCheckResult(
            check_name=check_name,
            passed=False,
            reason=reason,
            details=details or {},
        )
