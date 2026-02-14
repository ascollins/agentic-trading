"""Coin-flip baseline — statistical edge verification.

Answers the fundamental question: "Is my strategy doing better than
random?"  Uses a binomial test and bootstrapped comparison to
determine if observed win rates, R-multiples, and profit factors
are statistically significant or could arise from chance.

Inspired by Edgewonk's emphasis on separating skill from luck
through sufficient sample sizes and statistical validation.

Usage::

    baseline = CoinFlipBaseline()
    baseline.add_trades("trend", pnl_series=[12.5, -8.0, 25.0, ...])
    result = baseline.evaluate("trend")
    print(result["has_edge"])           # True if statistically significant
    print(result["p_value_win_rate"])   # 0.03 → significant
    print(result["random_better_pct"])  # 12% of random walks beat this
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _StrategyResults:
    """Per-strategy trade results for baseline comparison."""

    pnl_series: list[float] = field(default_factory=list)
    r_series: list[float] = field(default_factory=list)


class CoinFlipBaseline:
    """Statistical edge verification against random baseline.

    Parameters
    ----------
    n_simulations : int
        Number of random baseline simulations.  Default 10000.
    significance_level : float
        P-value threshold for declaring statistical significance.
        Default 0.05 (95% confidence).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        n_simulations: int = 10_000,
        significance_level: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self._n_sims = max(1000, n_simulations)
        self._sig_level = significance_level
        self._rng = random.Random(seed)
        self._data: dict[str, _StrategyResults] = defaultdict(_StrategyResults)

    # ------------------------------------------------------------------ #
    # Data input                                                           #
    # ------------------------------------------------------------------ #

    def add_trades(
        self,
        strategy_id: str,
        pnl_series: list[float],
        r_series: list[float] | None = None,
    ) -> None:
        """Set the complete trade history for comparison."""
        data = self._data[strategy_id]
        data.pnl_series = list(pnl_series)
        data.r_series = list(r_series) if r_series else []

    def add_trade(
        self,
        strategy_id: str,
        pnl: float,
        r_multiple: float = 0.0,
    ) -> None:
        """Incrementally add a single trade outcome."""
        data = self._data[strategy_id]
        data.pnl_series.append(pnl)
        if r_multiple != 0.0:
            data.r_series.append(r_multiple)

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self, strategy_id: str) -> dict[str, Any]:
        """Evaluate whether the strategy has a statistically significant edge.

        Performs three tests:
        1. Binomial test on win rate vs 50%
        2. Sign test on median P&L vs 0
        3. Bootstrap comparison of total P&L vs random direction flips

        Returns
        -------
        dict
            Comprehensive edge analysis results.
        """
        data = self._data.get(strategy_id)
        if data is None or len(data.pnl_series) < 10:
            return {
                "strategy_id": strategy_id,
                "has_edge": False,
                "error": "insufficient_data",
                "min_trades": 10,
                "current_trades": len(data.pnl_series) if data else 0,
            }

        pnls = data.pnl_series
        n = len(pnls)

        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        win_rate = wins / n
        total_pnl = sum(pnls)
        mean_pnl = total_pnl / n

        # --- Test 1: Binomial test on win rate vs 0.5 ---
        p_value_binomial = self._binomial_p_value(n, wins, 0.5)

        # --- Test 2: Sign test on median P&L ---
        positives = sum(1 for p in pnls if p > 0)
        p_value_sign = self._binomial_p_value(n, positives, 0.5)

        # --- Test 3: Bootstrap comparison ---
        # Randomly flip signs of P&L to simulate random entry direction
        random_better_count = 0
        abs_pnls = [abs(p) for p in pnls]

        for _ in range(self._n_sims):
            random_pnl = sum(
                p if self._rng.random() > 0.5 else -p for p in abs_pnls
            )
            if random_pnl >= total_pnl:
                random_better_count += 1

        random_better_pct = random_better_count / self._n_sims
        p_value_bootstrap = random_better_pct

        # --- Composite edge assessment ---
        # Require at least 2 of 3 tests to be significant AND positive expectancy
        significant_tests = sum([
            p_value_binomial < self._sig_level and win_rate > 0.5,
            p_value_sign < self._sig_level,
            p_value_bootstrap < self._sig_level,
        ])
        has_edge = significant_tests >= 2 and mean_pnl > 0

        # --- Sample size adequacy ---
        # How many more trades needed for power = 0.80?
        trades_needed = self._min_trades_for_significance(
            win_rate, power=0.80
        )

        # --- Effect size (Cohen's d) ---
        if n >= 2:
            std_pnl = statistics.stdev(pnls)
            cohens_d = mean_pnl / std_pnl if std_pnl > 0 else 0.0
        else:
            cohens_d = 0.0

        return {
            "strategy_id": strategy_id,
            "has_edge": has_edge,
            "trade_count": n,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "mean_pnl": round(mean_pnl, 2),
            # Test results
            "p_value_win_rate": round(p_value_binomial, 6),
            "p_value_sign_test": round(p_value_sign, 6),
            "p_value_bootstrap": round(p_value_bootstrap, 6),
            "significant_tests": significant_tests,
            "significance_level": self._sig_level,
            # Effect size
            "cohens_d": round(cohens_d, 4),
            "effect_size": (
                "large"
                if abs(cohens_d) >= 0.8
                else "medium"
                if abs(cohens_d) >= 0.5
                else "small"
                if abs(cohens_d) >= 0.2
                else "negligible"
            ),
            # Random comparison
            "random_better_pct": round(random_better_pct, 4),
            # Sample adequacy
            "min_trades_for_power": trades_needed,
            "sample_adequate": n >= trades_needed,
        }

    def has_edge(self, strategy_id: str) -> bool:
        """Quick check: does this strategy have a significant edge?"""
        result = self.evaluate(strategy_id)
        return result.get("has_edge", False)

    # ------------------------------------------------------------------ #
    # Comparison across strategies                                         #
    # ------------------------------------------------------------------ #

    def rank_strategies(self) -> list[dict[str, Any]]:
        """Rank all strategies by strength of statistical edge.

        Returns a list of evaluation summaries sorted by p_value_bootstrap.
        """
        results = []
        for sid in self._data:
            if len(self._data[sid].pnl_series) >= 10:
                result = self.evaluate(sid)
                results.append(result)
        return sorted(results, key=lambda r: r.get("p_value_bootstrap", 1.0))

    def get_all_strategy_ids(self) -> list[str]:
        """Return all strategy IDs with baseline data."""
        return [
            sid
            for sid, data in self._data.items()
            if len(data.pnl_series) > 0
        ]

    # ------------------------------------------------------------------ #
    # Private statistical helpers                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _binomial_p_value(n: int, k: int, p: float = 0.5) -> float:
        """One-tailed binomial test p-value.

        P(X >= k) under H0: X ~ Binomial(n, p).
        Uses normal approximation for n > 30, exact otherwise.
        """
        if n <= 0:
            return 1.0

        if n > 30:
            # Normal approximation
            mu = n * p
            sigma = math.sqrt(n * p * (1 - p))
            if sigma == 0:
                return 0.0 if k > mu else 1.0
            z = (k - 0.5 - mu) / sigma  # continuity correction
            # P(Z >= z) using complementary error function
            return 0.5 * math.erfc(z / math.sqrt(2))
        else:
            # Exact binomial (sum of PMFs)
            total_p = 0.0
            for i in range(k, n + 1):
                total_p += CoinFlipBaseline._binom_pmf(n, i, p)
            return min(1.0, total_p)

    @staticmethod
    def _binom_pmf(n: int, k: int, p: float) -> float:
        """Binomial probability mass function."""
        if k < 0 or k > n:
            return 0.0
        coeff = math.comb(n, k)
        return coeff * (p ** k) * ((1 - p) ** (n - k))

    @staticmethod
    def _min_trades_for_significance(
        observed_win_rate: float, power: float = 0.80
    ) -> int:
        """Estimate minimum trades needed to detect the observed edge.

        Uses the formula for sample size of a proportion test:
            n = (z_alpha + z_beta)^2 * p*(1-p) / (p - p0)^2
        where p0 = 0.5 (null hypothesis), p = observed win rate.
        """
        p0 = 0.5
        p = observed_win_rate

        delta = abs(p - p0)
        if delta < 0.001:
            return 10000  # Edge too small to detect

        # z-scores for common confidence / power levels
        z_alpha = 1.645  # one-tailed, alpha=0.05
        z_beta = 0.842   # power=0.80

        if power >= 0.90:
            z_beta = 1.282
        elif power >= 0.95:
            z_beta = 1.645

        numerator = (z_alpha + z_beta) ** 2 * p * (1 - p)
        denominator = delta ** 2

        return max(10, math.ceil(numerator / denominator))
