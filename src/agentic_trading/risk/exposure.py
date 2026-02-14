"""Exposure tracking across the portfolio.

Maintains real-time gross / net / per-asset / per-exchange exposure and
enforces configurable limits.  Updated on every position change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

from agentic_trading.core.enums import Exchange, PositionSide
from agentic_trading.core.models import Position

logger = logging.getLogger(__name__)


@dataclass
class ExposureSnapshot:
    """Point-in-time snapshot of portfolio exposure."""

    gross_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")
    per_asset: dict[str, Decimal] = field(default_factory=dict)
    per_exchange: dict[str, Decimal] = field(default_factory=dict)


class ExposureTracker:
    """Tracks and validates portfolio exposure in real time.

    Usage::

        tracker = ExposureTracker()
        tracker.update(portfolio_state.positions)

        if tracker.check_limits(max_gross_pct=3.0, capital=100_000):
            # limit breached
            ...
    """

    def __init__(self) -> None:
        self._snapshot = ExposureSnapshot()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, positions: dict[str, Position]) -> ExposureSnapshot:
        """Recompute exposure from the current position set.

        Args:
            positions: Mapping of symbol -> Position from PortfolioState.

        Returns:
            The updated :class:`ExposureSnapshot`.
        """
        gross = Decimal("0")
        net = Decimal("0")
        per_asset: dict[str, Decimal] = {}
        per_exchange: dict[str, Decimal] = {}

        for symbol, pos in positions.items():
            notional = abs(pos.notional)
            signed_notional = (
                pos.notional
                if pos.side == PositionSide.LONG
                else -abs(pos.notional)
            )

            gross += notional
            net += signed_notional

            # Per-asset: aggregate by base symbol (e.g. "BTC" from "BTC/USDT")
            base = symbol.split("/")[0] if "/" in symbol else symbol
            per_asset[base] = per_asset.get(base, Decimal("0")) + notional

            # Per-exchange
            exch_key = pos.exchange.value
            per_exchange[exch_key] = (
                per_exchange.get(exch_key, Decimal("0")) + notional
            )

        self._snapshot = ExposureSnapshot(
            gross_exposure=gross,
            net_exposure=net,
            per_asset=per_asset,
            per_exchange=per_exchange,
        )

        logger.debug(
            "Exposure updated: gross=%s, net=%s, assets=%d, exchanges=%d",
            gross,
            net,
            len(per_asset),
            len(per_exchange),
        )
        return self._snapshot

    def get_gross_exposure(self) -> Decimal:
        """Return the current gross exposure (sum of absolute notionals)."""
        return self._snapshot.gross_exposure

    def get_net_exposure(self) -> Decimal:
        """Return the current net exposure (longs minus shorts)."""
        return self._snapshot.net_exposure

    def get_per_asset_exposure(self) -> dict[str, Decimal]:
        """Return exposure broken down by base asset."""
        return dict(self._snapshot.per_asset)

    def get_per_exchange_exposure(self) -> dict[str, Decimal]:
        """Return exposure broken down by exchange."""
        return dict(self._snapshot.per_exchange)

    def get_snapshot(self) -> ExposureSnapshot:
        """Return a copy of the current exposure snapshot."""
        return ExposureSnapshot(
            gross_exposure=self._snapshot.gross_exposure,
            net_exposure=self._snapshot.net_exposure,
            per_asset=dict(self._snapshot.per_asset),
            per_exchange=dict(self._snapshot.per_exchange),
        )

    # ------------------------------------------------------------------
    # Limit checks
    # ------------------------------------------------------------------

    def check_limits(
        self,
        max_gross_pct: float,
        capital: float,
    ) -> bool:
        """Check whether gross exposure exceeds the allowed multiple of capital.

        Args:
            max_gross_pct: Maximum gross exposure as a multiple of capital
                (e.g. 3.0 means 300% / 3x leverage).
            capital: Reference capital in quote currency.

        Returns:
            ``True`` if the limit **is violated**, ``False`` otherwise.
        """
        if capital <= 0.0:
            logger.warning(
                "check_limits: capital=%.2f is non-positive, skipping",
                capital,
            )
            return False

        gross = float(self._snapshot.gross_exposure)
        ratio = gross / capital

        violated = ratio >= max_gross_pct
        if violated:
            logger.warning(
                "Gross exposure limit BREACHED: %.2fx (limit %.2fx) "
                "gross=%.2f capital=%.2f",
                ratio,
                max_gross_pct,
                gross,
                capital,
            )
        return violated

    def check_asset_concentration(
        self,
        max_single_asset_pct: float,
        capital: float,
    ) -> list[tuple[str, float]]:
        """Return assets whose exposure exceeds the per-asset concentration limit.

        Args:
            max_single_asset_pct: Maximum per-asset exposure as a fraction
                of capital (e.g. 0.10 for 10%).
            capital: Reference capital.

        Returns:
            List of ``(asset, ratio)`` tuples for breached assets.
            Empty list if all are within limits.
        """
        breached: list[tuple[str, float]] = []
        if capital <= 0.0:
            return breached

        for asset, notional in self._snapshot.per_asset.items():
            ratio = float(notional) / capital
            if ratio >= max_single_asset_pct:
                breached.append((asset, ratio))
                logger.warning(
                    "Asset concentration BREACHED: %s at %.2f%% (limit %.2f%%)",
                    asset,
                    ratio * 100,
                    max_single_asset_pct * 100,
                )
        return breached
