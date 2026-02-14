"""Correlation-based risk analysis.

Discovers clusters of correlated assets and computes
concentration risk metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationRiskAnalyzer:
    """Analyzes cross-asset correlation for risk management."""

    def __init__(
        self,
        lookback_periods: int = 60,
        correlation_threshold: float = 0.7,
    ) -> None:
        self._lookback = lookback_periods
        self._threshold = correlation_threshold
        self._returns_history: dict[str, list[float]] = {}

    def update_returns(self, symbol: str, ret: float) -> None:
        """Add a return observation for a symbol."""
        if symbol not in self._returns_history:
            self._returns_history[symbol] = []
        self._returns_history[symbol].append(ret)
        # Keep only lookback window
        if len(self._returns_history[symbol]) > self._lookback * 2:
            self._returns_history[symbol] = self._returns_history[symbol][-self._lookback:]

    def compute_correlation_matrix(self) -> pd.DataFrame | None:
        """Compute rolling correlation matrix across all tracked assets."""
        symbols = [
            s for s, r in self._returns_history.items()
            if len(r) >= self._lookback
        ]
        if len(symbols) < 2:
            return None

        data = {s: self._returns_history[s][-self._lookback:] for s in symbols}
        df = pd.DataFrame(data)
        return df.corr()

    def find_clusters(self) -> list[list[str]]:
        """Find clusters of highly correlated assets.

        Uses simple threshold-based clustering:
        assets with correlation > threshold are in the same cluster.
        """
        corr = self.compute_correlation_matrix()
        if corr is None:
            return []

        symbols = list(corr.columns)
        visited: set[str] = set()
        clusters: list[list[str]] = []

        for sym in symbols:
            if sym in visited:
                continue
            cluster = [sym]
            visited.add(sym)

            for other in symbols:
                if other in visited:
                    continue
                if abs(corr.loc[sym, other]) >= self._threshold:
                    cluster.append(other)
                    visited.add(other)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def compute_concentration_risk(
        self, positions: dict[str, float]
    ) -> dict[str, Any]:
        """Compute concentration risk metrics.

        Returns:
        - herfindahl_index: HHI (0 = diversified, 1 = concentrated)
        - top_3_concentration: % of exposure in top 3 positions
        - cluster_exposure: exposure by correlation cluster
        """
        if not positions:
            return {
                "herfindahl_index": 0.0,
                "top_3_concentration": 0.0,
                "cluster_exposure": {},
            }

        total = sum(abs(v) for v in positions.values())
        if total == 0:
            return {
                "herfindahl_index": 0.0,
                "top_3_concentration": 0.0,
                "cluster_exposure": {},
            }

        weights = {s: abs(v) / total for s, v in positions.items()}

        # HHI
        hhi = sum(w**2 for w in weights.values())

        # Top 3 concentration
        sorted_weights = sorted(weights.values(), reverse=True)
        top_3 = sum(sorted_weights[:3])

        # Cluster exposure
        clusters = self.find_clusters()
        cluster_exp = {}
        for i, cluster in enumerate(clusters):
            cluster_weight = sum(weights.get(s, 0) for s in cluster)
            cluster_exp[f"cluster_{i}"] = {
                "symbols": cluster,
                "weight": round(cluster_weight, 4),
            }

        return {
            "herfindahl_index": round(hhi, 4),
            "top_3_concentration": round(top_3, 4),
            "cluster_exposure": cluster_exp,
        }
