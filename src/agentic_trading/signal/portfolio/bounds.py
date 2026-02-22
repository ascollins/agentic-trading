"""Deterministic position bounds calculator.

Pre-computes which symbols can trade and max quantities BEFORE strategy
reasoning begins.  Combines risk config constraints, gross exposure
headroom, correlation cluster limits, position slots, and instrument
limits into a single ``PortfolioBounds`` snapshot.

Usage::

    calc = PositionBoundsCalculator(
        max_single_position_pct=0.10,
        max_concurrent_positions=8,
        max_daily_entries=10,
        max_leverage=3.0,
    )
    bounds = calc.compute(
        symbols=["BTC/USDT", "ETH/USDT"],
        portfolio=portfolio_state,
        capital=100_000.0,
        instruments=ctx.instruments,
        prices={"BTC/USDT": 60000.0, "ETH/USDT": 3200.0},
    )
    # bounds.bounds["BTC/USDT"].can_open_long  -> True / False
    # bounds.bounds["BTC/USDT"].max_buy_qty    -> Decimal("0.16")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from agentic_trading.core.interfaces import PortfolioState
from agentic_trading.core.models import Instrument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-symbol result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AllowedActions:
    """Pre-computed bounds for a single symbol."""

    symbol: str

    # Whether new entries are allowed
    can_open_long: bool = False
    can_open_short: bool = False
    can_close: bool = False

    # Maximum quantities for new orders
    max_buy_qty: Decimal = Decimal("0")
    max_sell_qty: Decimal = Decimal("0")
    max_notional: float = 0.0

    # Why actions are allowed/blocked
    reason: str = ""

    # Existing position info
    existing_direction: str = ""  # "long", "short", or ""
    existing_qty: Decimal = Decimal("0")

    # How much of the per-symbol capital headroom is left (0.0-1.0)
    capital_headroom_pct: float = 0.0


# ---------------------------------------------------------------------------
# All-symbols result
# ---------------------------------------------------------------------------

@dataclass
class PortfolioBounds:
    """Pre-computed bounds for the entire portfolio."""

    bounds: dict[str, AllowedActions] = field(default_factory=dict)
    capital: float = 0.0
    available_slots: int = 0
    gross_exposure_headroom_pct: float = 0.0

    def get(self, symbol: str) -> AllowedActions | None:
        """Look up bounds for a symbol."""
        return self.bounds.get(symbol)

    def can_open(self, symbol: str) -> bool:
        """Whether any new entry is allowed for *symbol*."""
        b = self.bounds.get(symbol)
        if b is None:
            return False
        return b.can_open_long or b.can_open_short


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class PositionBoundsCalculator:
    """Pre-computes deterministic position bounds.

    Parameters
    ----------
    max_single_position_pct:
        Max notional per position as fraction of capital.
    max_concurrent_positions:
        Max number of open positions at any time.
    max_daily_entries:
        Max new entries per calendar day.
    max_leverage:
        Max gross portfolio leverage.
    max_correlated_exposure_pct:
        Max notional for a correlated cluster as fraction of capital.
    """

    def __init__(
        self,
        max_single_position_pct: float = 0.10,
        max_concurrent_positions: int = 8,
        max_daily_entries: int = 10,
        max_leverage: float = 3.0,
        max_correlated_exposure_pct: float = 0.25,
    ) -> None:
        self._max_single_pct = max_single_position_pct
        self._max_concurrent = max_concurrent_positions
        self._max_daily_entries = max_daily_entries
        self._max_leverage = max_leverage
        self._max_correlated_pct = max_correlated_exposure_pct
        self._daily_entry_count: int = 0
        self._last_entry_date: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        symbols: list[str],
        portfolio: PortfolioState,
        capital: float,
        instruments: dict[str, Instrument] | None = None,
        prices: dict[str, float] | None = None,
        atrs: dict[str, float] | None = None,
        correlation_clusters: list[list[str]] | None = None,
    ) -> PortfolioBounds:
        """Compute position bounds for all *symbols*.

        Returns a :class:`PortfolioBounds` snapshot that strategies and
        the portfolio manager can consult before sizing.
        """
        instruments = instruments or {}
        prices = prices or {}
        atrs = atrs or {}

        if capital <= 0:
            return PortfolioBounds(capital=capital)

        # Portfolio-level metrics
        gross_exposure = float(portfolio.gross_exposure)
        max_gross = capital * self._max_leverage
        gross_headroom = max(0.0, max_gross - gross_exposure)
        gross_headroom_pct = gross_headroom / max_gross if max_gross > 0 else 0.0

        # Available position slots
        open_count = sum(
            1 for p in portfolio.positions.values()
            if getattr(p, "is_open", True) and p.qty > 0
        )
        available_slots = max(0, self._max_concurrent - open_count)

        # Daily entry limit
        daily_entries_remaining = max(
            0, self._max_daily_entries - self._daily_entry_count
        )

        # Build per-symbol cluster exposure map
        cluster_exposure = self._build_cluster_exposure(
            portfolio, correlation_clusters, capital,
        )

        result_bounds: dict[str, AllowedActions] = {}

        for symbol in symbols:
            result_bounds[symbol] = self._compute_symbol(
                symbol=symbol,
                portfolio=portfolio,
                capital=capital,
                gross_headroom=gross_headroom,
                available_slots=available_slots,
                daily_entries_remaining=daily_entries_remaining,
                instrument=instruments.get(symbol),
                price=prices.get(symbol, 0.0),
                cluster_cap=cluster_exposure.get(symbol),
            )

        bounds = PortfolioBounds(
            bounds=result_bounds,
            capital=capital,
            available_slots=available_slots,
            gross_exposure_headroom_pct=gross_headroom_pct,
        )

        logger.info(
            "Position bounds computed: %d symbols, %d slots, "
            "gross_headroom=%.1f%%",
            len(symbols),
            available_slots,
            gross_headroom_pct * 100,
        )

        return bounds

    def record_entry(self, date_str: str) -> None:
        """Increment daily entry counter.  Call when a new entry is filled."""
        if date_str != self._last_entry_date:
            self._daily_entry_count = 0
            self._last_entry_date = date_str
        self._daily_entry_count += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_symbol(
        self,
        *,
        symbol: str,
        portfolio: PortfolioState,
        capital: float,
        gross_headroom: float,
        available_slots: int,
        daily_entries_remaining: int,
        instrument: Instrument | None,
        price: float,
        cluster_cap: float | None,
    ) -> AllowedActions:
        """Compute bounds for a single symbol."""
        existing = portfolio.get_position(symbol)
        existing_open = (
            existing is not None
            and getattr(existing, "is_open", True)
            and existing.qty > 0
        )

        existing_dir = ""
        existing_qty = Decimal("0")

        if existing_open:
            existing_dir = (
                existing.side.value
                if hasattr(existing.side, "value")
                else str(existing.side)
            )
            existing_qty = existing.qty

        # --- Close-only if already positioned ---
        if existing_open:
            return AllowedActions(
                symbol=symbol,
                can_open_long=False,
                can_open_short=False,
                can_close=True,
                max_buy_qty=(
                    existing_qty if existing_dir == "short" else Decimal("0")
                ),
                max_sell_qty=(
                    existing_qty if existing_dir == "long" else Decimal("0")
                ),
                max_notional=0.0,
                reason=f"close-only: existing {existing_dir} position",
                existing_direction=existing_dir,
                existing_qty=existing_qty,
                capital_headroom_pct=0.0,
            )

        # --- No slots or daily entries left ---
        if available_slots <= 0:
            return self._blocked(symbol, "no position slots available")

        if daily_entries_remaining <= 0:
            return self._blocked(symbol, "daily entry limit reached")

        # --- Capital headroom ---
        max_notional = min(
            capital * self._max_single_pct,
            gross_headroom,
        )

        # Apply correlation cluster cap
        if cluster_cap is not None:
            max_notional = min(max_notional, cluster_cap)

        if max_notional <= 0:
            return self._blocked(symbol, "no capital headroom")

        # --- Instrument limits ---
        max_qty = Decimal("999999999")
        min_qty = Decimal("0")
        min_notional = Decimal("0")

        if instrument is not None:
            max_qty = instrument.max_qty
            min_qty = instrument.min_qty
            min_notional = instrument.min_notional

        # Convert max_notional to max_qty if we have a price
        if price > 0:
            notional_qty = Decimal(str(max_notional / price))
            max_qty = min(max_qty, notional_qty)

            # Check instrument min_notional
            if min_notional > 0 and max_notional < float(min_notional):
                return self._blocked(
                    symbol,
                    f"max_notional {max_notional:.0f} < min_notional {min_notional}",
                )

        # Clamp to instrument min_qty
        if max_qty < min_qty:
            return self._blocked(
                symbol,
                f"max_qty {max_qty} < min_qty {min_qty}",
            )

        # Round to instrument step
        if instrument is not None:
            max_qty = instrument.round_qty(max_qty)
            if max_qty <= 0:
                return self._blocked(symbol, "qty rounds to 0")

        headroom_pct = (
            max_notional / (capital * self._max_single_pct)
            if capital * self._max_single_pct > 0
            else 0.0
        )

        return AllowedActions(
            symbol=symbol,
            can_open_long=True,
            can_open_short=True,
            can_close=False,
            max_buy_qty=max_qty,
            max_sell_qty=max_qty,
            max_notional=max_notional,
            reason="open",
            existing_direction="",
            existing_qty=Decimal("0"),
            capital_headroom_pct=min(1.0, headroom_pct),
        )

    def _build_cluster_exposure(
        self,
        portfolio: PortfolioState,
        correlation_clusters: list[list[str]] | None,
        capital: float,
    ) -> dict[str, float]:
        """Per-symbol remaining notional budget from cluster limits.

        Returns a dict mapping symbol → remaining notional headroom
        within its cluster.  Symbols not in any cluster are omitted.
        """
        if not correlation_clusters:
            return {}

        result: dict[str, float] = {}
        max_cluster_notional = capital * self._max_correlated_pct

        for cluster in correlation_clusters:
            # Existing exposure in this cluster
            cluster_exposure = sum(
                float(abs(p.notional))
                for sym, p in portfolio.positions.items()
                if sym in cluster
            )
            remaining = max(0.0, max_cluster_notional - cluster_exposure)

            for sym in cluster:
                # Take the tightest constraint if symbol appears in
                # multiple clusters (shouldn't happen, but be safe).
                if sym in result:
                    result[sym] = min(result[sym], remaining)
                else:
                    result[sym] = remaining

        return result

    @staticmethod
    def _blocked(symbol: str, reason: str) -> AllowedActions:
        """Create a fully-blocked AllowedActions."""
        return AllowedActions(
            symbol=symbol,
            can_open_long=False,
            can_open_short=False,
            can_close=False,
            reason=reason,
        )
