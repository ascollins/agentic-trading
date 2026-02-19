"""Correlation-based risk analysis.

Discovers clusters of correlated assets and computes
concentration risk metrics.

Includes both *dynamic* correlation (from rolling return data via
:class:`CorrelationRiskAnalyzer`) and *structural* correlation (predefined
crypto-pair tiers via :func:`quick_correlation_check`) for instant checks
without historical data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structural / predefined correlations (no historical data needed)
# ---------------------------------------------------------------------------

STRUCTURAL_CORRELATIONS: dict[str, dict[str, Any]] = {
    "very_high": {
        "threshold": 0.85,
        "description": "Moves nearly identically — treat as single position for risk",
        "pairs": [
            ("BTC/USDT", "BTC/USDC"),
            ("ETH/USDT", "ETH/USDC"),
            ("BTC/USDT:USDT", "BTC/USDT"),  # perp vs spot
            ("ETH/USDT:USDT", "ETH/USDT"),  # perp vs spot
        ],
    },
    "high": {
        "threshold": 0.70,
        "description": "Strong correlation — cap combined risk at 1.5x single position",
        "pairs": [
            ("BTC/USDT", "ETH/USDT"),
            ("ETH/USDT", "SOL/USDT"),
            ("SOL/USDT", "AVAX/USDT"),
            ("SOL/USDT", "SUI/USDT"),
            ("SOL/USDT", "APT/USDT"),
            ("LINK/USDT", "UNI/USDT"),
            ("LINK/USDT", "AAVE/USDT"),
            ("ARB/USDT", "OP/USDT"),
            ("DOGE/USDT", "SHIB/USDT"),
            ("DOGE/USDT", "PEPE/USDT"),
        ],
    },
    "moderate": {
        "threshold": 0.50,
        "description": "Moderate correlation — cap combined risk at 2x single position",
        "pairs": [
            ("BTC/USDT", "SOL/USDT"),
            ("BTC/USDT", "DOGE/USDT"),
            ("ETH/USDT", "AVAX/USDT"),
            ("ETH/USDT", "LINK/USDT"),
            ("BNB/USDT", "ETH/USDT"),
            ("BTC/USDT", "XRP/USDT"),
        ],
    },
}

_TIER_ORDER = ["very_high", "high", "moderate"]


def quick_correlation_check(
    positions: list[str],
    threshold: str = "high",
) -> dict[str, Any]:
    """Fast structural correlation check without historical data.

    Uses predefined correlation tiers to identify clusters of correlated
    positions and compute a diversification score.

    Args:
        positions: List of symbol strings (e.g. ``["BTC/USDT", "ETH/USDT"]``).
        threshold: Minimum tier to flag — ``"very_high"``, ``"high"``, or
            ``"moderate"``.

    Returns:
        Dict with keys:

        - ``correlated_clusters``: list of lists of correlated symbols
        - ``diversification_score``: float 0–1 (1 = fully diversified)
        - ``warnings``: list of warning strings
        - ``flagged_pairs``: list of (sym_a, sym_b, tier) tuples
    """
    if threshold not in _TIER_ORDER:
        threshold = "high"

    active_tiers = _TIER_ORDER[: _TIER_ORDER.index(threshold) + 1]
    pos_set = set(positions)

    flagged_pairs: list[tuple[str, str, str]] = []
    for tier in active_tiers:
        tier_data = STRUCTURAL_CORRELATIONS.get(tier, {})
        for a, b in tier_data.get("pairs", []):
            if a in pos_set and b in pos_set:
                flagged_pairs.append((a, b, tier))

    # Build clusters via simple union-find grouping
    clusters: list[set[str]] = []
    for a, b, _ in flagged_pairs:
        merged = False
        for cluster in clusters:
            if a in cluster or b in cluster:
                cluster.add(a)
                cluster.add(b)
                merged = True
                break
        if not merged:
            clusters.append({a, b})

    correlated_count = len(set().union(*clusters)) if clusters else 0
    total = max(len(positions), 1)
    diversification_score = 1.0 - (correlated_count / total)

    warnings = []
    for a, b, tier in flagged_pairs:
        desc = STRUCTURAL_CORRELATIONS.get(tier, {}).get("description", tier)
        warnings.append(f"{a} ↔ {b}: {desc}")

    return {
        "correlated_clusters": [sorted(c) for c in clusters],
        "diversification_score": round(max(0.0, diversification_score), 4),
        "warnings": warnings,
        "flagged_pairs": flagged_pairs,
    }


# ---------------------------------------------------------------------------
# Dynamic correlation analyzer (rolling returns)
# ---------------------------------------------------------------------------


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
