"""Cross-strategy allocation and conflict resolution.

Ensures that aggregate positions across strategies
don't exceed portfolio-level limits.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agentic_trading.core.events import TargetPosition
from agentic_trading.core.interfaces import PortfolioState

logger = logging.getLogger(__name__)


class PortfolioAllocator:
    """Allocates capital across strategies and resolves position conflicts."""

    def __init__(
        self,
        max_gross_exposure_pct: float = 1.0,
        max_single_position_pct: float = 0.10,
        max_correlated_exposure_pct: float = 0.25,
    ) -> None:
        self._max_gross = max_gross_exposure_pct
        self._max_single = max_single_position_pct
        self._max_correlated = max_correlated_exposure_pct

    def allocate(
        self,
        targets: list[TargetPosition],
        portfolio: PortfolioState,
        capital: float,
        correlation_clusters: list[list[str]] | None = None,
    ) -> list[TargetPosition]:
        """Apply portfolio-level constraints to target positions.

        Returns adjusted targets that respect:
        - Max gross exposure
        - Max single position
        - Max correlated cluster exposure
        """
        if not targets or capital <= 0:
            return []

        adjusted = []

        for target in targets:
            # Cap individual position
            max_notional = capital * self._max_single
            target_notional = float(target.target_qty) * self._estimate_price(target)

            if target_notional > max_notional:
                scale = max_notional / target_notional
                target.target_qty = Decimal(str(float(target.target_qty) * scale))
                logger.info(
                    "Capped %s position to %.2f%% of capital",
                    target.symbol,
                    self._max_single * 100,
                )

            adjusted.append(target)

        # Check gross exposure constraint
        total_notional = sum(
            float(t.target_qty) * self._estimate_price(t) for t in adjusted
        )
        existing_notional = float(portfolio.gross_exposure)
        new_gross = total_notional + existing_notional

        if new_gross > capital * self._max_gross:
            # Scale down all new targets proportionally
            excess = new_gross - capital * self._max_gross
            if total_notional > 0:
                scale = max(0, 1.0 - excess / total_notional)
                for t in adjusted:
                    t.target_qty = Decimal(str(float(t.target_qty) * scale))
                logger.warning(
                    "Scaled targets by %.2f to respect gross exposure limit",
                    scale,
                )

        # Check correlated cluster exposure
        if correlation_clusters:
            adjusted = self._apply_cluster_limits(
                adjusted, correlation_clusters, portfolio, capital
            )

        return [t for t in adjusted if t.target_qty > Decimal("0")]

    def _apply_cluster_limits(
        self,
        targets: list[TargetPosition],
        clusters: list[list[str]],
        portfolio: PortfolioState,
        capital: float,
    ) -> list[TargetPosition]:
        """Reduce positions in correlated clusters if exposure exceeds limit."""
        max_cluster_notional = capital * self._max_correlated

        for cluster in clusters:
            # Sum existing exposure in this cluster
            cluster_exposure = sum(
                float(abs(p.notional))
                for sym, p in portfolio.positions.items()
                if sym in cluster
            )
            # Add new targets in this cluster
            cluster_targets = [t for t in targets if t.symbol in cluster]
            new_cluster_notional = sum(
                float(t.target_qty) * self._estimate_price(t)
                for t in cluster_targets
            )

            total = cluster_exposure + new_cluster_notional
            if total > max_cluster_notional and new_cluster_notional > 0:
                scale = max(0, (max_cluster_notional - cluster_exposure) / new_cluster_notional)
                for t in cluster_targets:
                    t.target_qty = Decimal(str(float(t.target_qty) * scale))
                logger.warning(
                    "Cluster %s: scaled by %.2f to respect correlated exposure limit",
                    cluster[:3],
                    scale,
                )

        return targets

    @staticmethod
    def _estimate_price(target: TargetPosition) -> float:
        """Rough price estimate for notional calculation.

        In production, this would look up the latest price.
        For now, use 1.0 as a safe fallback (qty is in base units).
        """
        # TODO: Wire up price lookup from data feeds
        return 1.0
