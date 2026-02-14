"""Drawdown monitoring and daily loss limits.

Tracks peak equity, running drawdown, and intra-day PnL to enforce
configurable loss limits.  The monitor is designed to be called on
every portfolio valuation tick so that breaches are caught immediately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class DrawdownMonitor:
    """Tracks peak equity, current drawdown, and daily PnL.

    Typical usage::

        monitor = DrawdownMonitor(initial_equity=100_000.0)
        # On every portfolio valuation tick:
        if monitor.check_drawdown(current_equity, max_drawdown_pct=0.15):
            # drawdown limit breached -- activate kill switch / reduce
            ...
        if monitor.check_daily_loss(daily_pnl, max_daily_loss_pct=0.05, capital=100_000):
            # daily loss limit breached
            ...
    """

    initial_equity: float = 0.0

    # ---- internal state (managed automatically) ----
    peak_equity: float = field(init=False, default=0.0)
    current_drawdown_pct: float = field(init=False, default=0.0)
    daily_pnl: float = field(init=False, default=0.0)
    _daily_start_equity: float = field(init=False, default=0.0)
    _current_date: date | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.peak_equity = self.initial_equity
        self._daily_start_equity = self.initial_equity
        self._current_date = datetime.now(timezone.utc).date()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_equity(self, current_equity: float) -> None:
        """Update the monitor with the latest equity value.

        Should be called on every valuation tick (e.g. after each fill or
        on a periodic timer).  Automatically rolls the daily counters
        when a new calendar day is detected.

        Args:
            current_equity: Current total portfolio equity (USD or quote).
        """
        today = datetime.now(timezone.utc).date()
        if self._current_date is not None and today != self._current_date:
            logger.info(
                "DrawdownMonitor: new trading day detected (%s -> %s), "
                "resetting daily counters",
                self._current_date,
                today,
            )
            self.reset_daily(current_equity)
            self._current_date = today

        # Track running peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Compute running drawdown
        if self.peak_equity > 0.0:
            self.current_drawdown_pct = (
                (self.peak_equity - current_equity) / self.peak_equity
            )
        else:
            self.current_drawdown_pct = 0.0

        # Compute intraday PnL
        self.daily_pnl = current_equity - self._daily_start_equity

    def check_drawdown(
        self,
        current_equity: float,
        max_drawdown_pct: float,
    ) -> bool:
        """Check whether the max drawdown limit has been breached.

        Automatically calls :meth:`update_equity` so callers do not need
        to call both.

        Args:
            current_equity: Current total portfolio equity.
            max_drawdown_pct: Maximum allowed drawdown as a fraction
                (e.g. 0.15 for 15%).

        Returns:
            ``True`` if the drawdown limit **is violated** (trading should
            be halted), ``False`` otherwise.
        """
        self.update_equity(current_equity)

        violated = self.current_drawdown_pct >= max_drawdown_pct
        if violated:
            logger.warning(
                "Drawdown limit BREACHED: current=%.2f%%, limit=%.2f%% "
                "(peak=%.2f, equity=%.2f)",
                self.current_drawdown_pct * 100,
                max_drawdown_pct * 100,
                self.peak_equity,
                current_equity,
            )
        return violated

    def check_daily_loss(
        self,
        daily_pnl: float,
        max_daily_loss_pct: float,
        capital: float,
    ) -> bool:
        """Check whether the daily loss limit has been breached.

        Args:
            daily_pnl: Intra-day PnL (negative means a loss).
            max_daily_loss_pct: Maximum allowed daily loss as a fraction
                (e.g. 0.05 for 5%).
            capital: Reference capital for computing the limit in
                absolute terms.

        Returns:
            ``True`` if the daily loss limit **is violated**, ``False``
            otherwise.
        """
        if capital <= 0.0:
            logger.warning("check_daily_loss: capital=%.2f is non-positive", capital)
            return False

        loss_pct = abs(daily_pnl) / capital if daily_pnl < 0 else 0.0
        violated = loss_pct >= max_daily_loss_pct

        if violated:
            logger.warning(
                "Daily loss limit BREACHED: pnl=%.2f (%.2f%%), limit=%.2f%%",
                daily_pnl,
                loss_pct * 100,
                max_daily_loss_pct * 100,
            )
        return violated

    def reset_daily(self, current_equity: float | None = None) -> None:
        """Reset daily counters for a new trading day.

        Args:
            current_equity: Starting equity for the new day.  If ``None``
                the peak equity is used.
        """
        equity = current_equity if current_equity is not None else self.peak_equity
        self._daily_start_equity = equity
        self.daily_pnl = 0.0
        self._current_date = datetime.now(timezone.utc).date()
        logger.info(
            "DrawdownMonitor: daily counters reset  "
            "(start_equity=%.2f, peak=%.2f)",
            self._daily_start_equity,
            self.peak_equity,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def max_drawdown_from_peak(self) -> float:
        """Return the current drawdown as a fraction of peak equity."""
        return self.current_drawdown_pct
