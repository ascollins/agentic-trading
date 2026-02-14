"""Cross-asset correlation and lead/lag discovery.

Provides tools for:

* **Rolling pairwise correlation** between return series.
* **Correlation matrix** across a universe of assets.
* **Lead/lag detection** via cross-correlation to discover which asset
  moves first.
* **Hierarchical clustering** of correlated assets into groups for
  portfolio-level risk management.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional pandas import -- only needed for matrix operations.
try:
    import pandas as pd  # type: ignore[import-untyped]

    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False
    logger.info(
        "pandas not installed - correlation_matrix and cluster_assets "
        "will not be available"
    )


class CorrelationEngine:
    """Stateless utilities for cross-asset correlation analysis.

    All methods are pure functions operating on numpy/pandas inputs.
    The class groups related functionality and stores no mutable state,
    making it safe to share across threads.

    Usage::

        engine = CorrelationEngine()

        # Pairwise rolling correlation
        rho = engine.compute_rolling_correlation(rets_btc, rets_eth, window=30)

        # Full correlation matrix
        matrix = engine.compute_correlation_matrix(
            {"BTC": rets_btc, "ETH": rets_eth, "SOL": rets_sol},
            window=30,
        )

        # Lead/lag discovery
        lag, corr = engine.detect_lead_lag(rets_btc, rets_eth, max_lag=10)

        # Cluster correlated assets
        groups = engine.cluster_assets(matrix, threshold=0.7)
    """

    # ------------------------------------------------------------------
    # Rolling pairwise correlation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rolling_correlation(
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        window: int,
    ) -> float:
        """Compute the rolling Pearson correlation over the most recent
        *window* observations.

        Args:
            returns_a: 1-D return series for asset A.
            returns_b: 1-D return series for asset B.
            window: Number of most-recent observations to use.

        Returns:
            Scalar correlation coefficient in ``[-1, 1]``.
            Returns ``nan`` if either series has insufficient data or
            zero variance within the window.
        """
        a = np.asarray(returns_a, dtype=np.float64)
        b = np.asarray(returns_b, dtype=np.float64)

        # Align lengths.
        min_len = min(len(a), len(b))
        if min_len < window:
            return float("nan")

        a_win = a[-window:]
        b_win = b[-window:]

        # Remove any NaN pairs.
        mask = ~(np.isnan(a_win) | np.isnan(b_win))
        a_clean = a_win[mask]
        b_clean = b_win[mask]

        if len(a_clean) < 3:
            return float("nan")

        std_a = np.std(a_clean, ddof=1)
        std_b = np.std(b_clean, ddof=1)
        if std_a == 0 or std_b == 0:
            return float("nan")

        corr = np.corrcoef(a_clean, b_clean)[0, 1]
        return float(corr)

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def compute_correlation_matrix(
        returns_dict: dict[str, np.ndarray],
        window: int,
    ) -> "pd.DataFrame":
        """Compute a Pearson correlation matrix from a dict of return
        series, using the most recent *window* observations.

        Args:
            returns_dict: Mapping from asset name to 1-D return array.
            window: Number of trailing observations.

        Returns:
            A ``pandas.DataFrame`` correlation matrix with asset names
            as both index and columns.

        Raises:
            RuntimeError: If pandas is not installed.
        """
        if not _HAS_PANDAS:
            raise RuntimeError(
                "pandas is required for compute_correlation_matrix"
            )

        # Build aligned DataFrame from trailing window.
        data: dict[str, np.ndarray] = {}
        for name, rets in returns_dict.items():
            arr = np.asarray(rets, dtype=np.float64)
            if len(arr) >= window:
                data[name] = arr[-window:]
            else:
                # Pad front with NaN if series is shorter.
                padded = np.full(window, np.nan)
                padded[-len(arr):] = arr
                data[name] = padded

        df = pd.DataFrame(data)
        return df.corr(method="pearson")

    # ------------------------------------------------------------------
    # Lead / lag detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_lead_lag(
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        max_lag: int,
    ) -> tuple[int, float]:
        """Find the lag at which *returns_a* and *returns_b* are most
        correlated, using normalised cross-correlation.

        A **positive** optimal lag means ``returns_a`` *leads*
        ``returns_b`` by that many periods.  A **negative** lag means
        ``returns_b`` leads.

        Args:
            returns_a: 1-D return series for asset A.
            returns_b: 1-D return series for asset B.
            max_lag: Maximum lag (in both directions) to test.

        Returns:
            Tuple of (optimal_lag, correlation_at_lag).
        """
        a = np.asarray(returns_a, dtype=np.float64)
        b = np.asarray(returns_b, dtype=np.float64)

        # Align lengths.
        min_len = min(len(a), len(b))
        if min_len < max_lag + 1:
            return 0, float("nan")

        a = a[-min_len:]
        b = b[-min_len:]

        # Remove NaNs pairwise on the full overlap.
        mask = ~(np.isnan(a) | np.isnan(b))
        a = a[mask]
        b = b[mask]
        if len(a) < max_lag + 3:
            return 0, float("nan")

        best_lag = 0
        best_corr = -2.0  # Correlation is in [-1, 1].

        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                a_seg = a[lag:]
                b_seg = b[: len(a_seg)]
            elif lag < 0:
                b_seg = b[-lag:]
                a_seg = a[: len(b_seg)]
            else:
                a_seg = a
                b_seg = b

            if len(a_seg) < 3:
                continue

            std_a = np.std(a_seg, ddof=1)
            std_b = np.std(b_seg, ddof=1)
            if std_a == 0 or std_b == 0:
                continue

            corr = float(np.corrcoef(a_seg, b_seg)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        return best_lag, best_corr

    # ------------------------------------------------------------------
    # Asset clustering
    # ------------------------------------------------------------------

    @staticmethod
    def cluster_assets(
        correlation_matrix: "pd.DataFrame",
        threshold: float,
    ) -> list[list[str]]:
        """Group assets whose pairwise correlation exceeds *threshold*
        using a simple single-linkage approach.

        This is intentionally a lightweight, dependency-free clustering
        method suitable for real-time use.  For more sophisticated
        clustering (e.g. hierarchical with scipy), extend this method.

        Args:
            correlation_matrix: Square correlation DataFrame (from
                :meth:`compute_correlation_matrix`).
            threshold: Minimum absolute correlation to consider two
                assets as belonging to the same cluster.

        Returns:
            List of clusters, where each cluster is a list of asset
            name strings.
        """
        if not _HAS_PANDAS:
            raise RuntimeError(
                "pandas is required for cluster_assets"
            )

        assets = list(correlation_matrix.columns)
        n = len(assets)

        # Union-Find for single-linkage clustering.
        parent: dict[str, str] = {a: a for a in assets}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Path compression.
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(n):
            for j in range(i + 1, n):
                val = correlation_matrix.iloc[i, j]
                if not np.isnan(val) and abs(val) >= threshold:
                    union(assets[i], assets[j])

        # Collect groups.
        groups: dict[str, list[str]] = defaultdict(list)
        for asset in assets:
            root = find(asset)
            groups[root].append(asset)

        return list(groups.values())

    # ------------------------------------------------------------------
    # Convenience: full-series rolling correlation array
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rolling_correlation_series(
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Compute a rolling Pearson correlation over the full length of
        the two return series.

        Args:
            returns_a: 1-D return series for asset A.
            returns_b: 1-D return series for asset B.
            window: Rolling window size.

        Returns:
            1-D array of length ``min(len(a), len(b))`` with leading
            NaN for the warmup period.
        """
        a = np.asarray(returns_a, dtype=np.float64)
        b = np.asarray(returns_b, dtype=np.float64)

        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        result = np.full(min_len, np.nan)

        for i in range(window - 1, min_len):
            a_win = a[i - window + 1: i + 1]
            b_win = b[i - window + 1: i + 1]

            mask = ~(np.isnan(a_win) | np.isnan(b_win))
            a_c = a_win[mask]
            b_c = b_win[mask]

            if len(a_c) < 3:
                continue

            std_a = np.std(a_c, ddof=1)
            std_b = np.std(b_c, ddof=1)
            if std_a == 0 or std_b == 0:
                continue

            result[i] = float(np.corrcoef(a_c, b_c)[0, 1])

        return result
