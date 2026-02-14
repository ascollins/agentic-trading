"""Confidence calibrator — signal quality vs realised outcome.

Measures how well strategy confidence scores predict actual outcomes.
A well-calibrated strategy has high-confidence signals that win more
often and with larger R-multiples than low-confidence ones.

Inspired by Edgewonk's "mistake tracking" and "self-rating" features,
this module builds a feedback loop between the confidence a strategy
assigns to its signals and what actually happens.

Usage::

    calibrator = ConfidenceCalibrator(n_buckets=5)
    calibrator.record(strategy_id="trend", confidence=0.85, won=True, r_multiple=2.3)
    report = calibrator.report("trend")
    print(report["buckets"])  # [{range, count, win_rate, avg_r}, ...]
    print(report["brier_score"])  # 0.0 = perfect, 0.25 = coin flip
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _Observation:
    """Single confidence observation."""

    confidence: float
    won: bool
    r_multiple: float


@dataclass
class _StrategyCalibration:
    """Per-strategy calibration data."""

    observations: list[_Observation] = field(default_factory=list)
    _dirty: bool = True
    _cache: dict[str, Any] = field(default_factory=dict)


class ConfidenceCalibrator:
    """Measures signal-confidence calibration quality.

    Parameters
    ----------
    n_buckets : int
        Number of confidence bins for the calibration curve.
        Default 5 (quintiles: 0-20%, 20-40%, ..., 80-100%).
    max_observations : int
        Maximum observations per strategy before FIFO eviction.
        Default 2000.
    """

    def __init__(
        self,
        *,
        n_buckets: int = 5,
        max_observations: int = 2000,
    ) -> None:
        self._n_buckets = max(2, n_buckets)
        self._max_obs = max_observations
        self._data: dict[str, _StrategyCalibration] = defaultdict(
            _StrategyCalibration
        )

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record(
        self,
        strategy_id: str,
        confidence: float,
        won: bool,
        r_multiple: float = 0.0,
    ) -> None:
        """Record a trade outcome with its signal confidence.

        Parameters
        ----------
        strategy_id : str
            Strategy that generated the signal.
        confidence : float
            Signal confidence at the time of entry [0.0, 1.0].
        won : bool
            Whether the trade was a winner.
        r_multiple : float
            Realised R-multiple of the trade.
        """
        cal = self._data[strategy_id]
        cal.observations.append(
            _Observation(
                confidence=max(0.0, min(1.0, confidence)),
                won=won,
                r_multiple=r_multiple,
            )
        )
        # FIFO eviction
        while len(cal.observations) > self._max_obs:
            cal.observations.pop(0)
        cal._dirty = True

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def report(self, strategy_id: str) -> dict[str, Any]:
        """Generate a calibration report for a strategy.

        Returns
        -------
        dict
            ``buckets`` : list of dicts with range, count, win_rate, avg_r,
            avg_confidence, expected_win_rate.
            ``brier_score`` : overall Brier score (lower is better).
            ``calibration_error`` : mean absolute calibration error.
            ``overconfidence_ratio`` : fraction of buckets where confidence > win_rate.
            ``total_observations`` : total sample size.
        """
        cal = self._data.get(strategy_id)
        if cal is None or len(cal.observations) == 0:
            return {
                "buckets": [],
                "brier_score": 1.0,
                "calibration_error": 1.0,
                "overconfidence_ratio": 1.0,
                "total_observations": 0,
            }

        if not cal._dirty and cal._cache:
            return cal._cache

        obs = cal.observations
        n = len(obs)
        step = 1.0 / self._n_buckets

        buckets = []
        for i in range(self._n_buckets):
            lo = i * step
            hi = (i + 1) * step
            in_bucket = [
                o for o in obs if lo <= o.confidence < hi or (i == self._n_buckets - 1 and o.confidence == hi)
            ]
            count = len(in_bucket)
            if count == 0:
                buckets.append({
                    "range": f"{lo:.2f}-{hi:.2f}",
                    "count": 0,
                    "win_rate": 0.0,
                    "avg_r": 0.0,
                    "avg_confidence": (lo + hi) / 2,
                    "expected_win_rate": (lo + hi) / 2,
                })
                continue

            wins_in = sum(1 for o in in_bucket if o.won)
            win_rate = wins_in / count
            avg_r = sum(o.r_multiple for o in in_bucket) / count
            avg_conf = sum(o.confidence for o in in_bucket) / count

            buckets.append({
                "range": f"{lo:.2f}-{hi:.2f}",
                "count": count,
                "win_rate": round(win_rate, 4),
                "avg_r": round(avg_r, 4),
                "avg_confidence": round(avg_conf, 4),
                "expected_win_rate": round(avg_conf, 4),
            })

        # Brier score: mean of (confidence - outcome)^2
        brier = sum(
            (o.confidence - (1.0 if o.won else 0.0)) ** 2 for o in obs
        ) / n

        # Mean absolute calibration error across buckets
        non_empty = [b for b in buckets if b["count"] > 0]
        if non_empty:
            cal_error = sum(
                abs(b["avg_confidence"] - b["win_rate"]) for b in non_empty
            ) / len(non_empty)
        else:
            cal_error = 1.0

        # Overconfidence ratio
        overconf_buckets = [
            b for b in non_empty if b["avg_confidence"] > b["win_rate"]
        ]
        overconf_ratio = (
            len(overconf_buckets) / len(non_empty) if non_empty else 1.0
        )

        result = {
            "buckets": buckets,
            "brier_score": round(brier, 6),
            "calibration_error": round(cal_error, 6),
            "overconfidence_ratio": round(overconf_ratio, 4),
            "total_observations": n,
        }

        cal._cache = result
        cal._dirty = False
        return result

    def is_well_calibrated(
        self,
        strategy_id: str,
        *,
        max_brier: float = 0.25,
        min_observations: int = 30,
    ) -> bool:
        """Quick check: is this strategy reasonably well-calibrated?

        Returns False if insufficient data or Brier score exceeds threshold.
        """
        report = self.report(strategy_id)
        if report["total_observations"] < min_observations:
            return False
        return report["brier_score"] <= max_brier

    def get_confidence_edge(self, strategy_id: str) -> float:
        """How much does high confidence predict better outcomes?

        Returns the correlation between confidence bucket midpoint
        and win rate.  Positive = good calibration, near zero = no
        signal, negative = inverted confidence.
        """
        report = self.report(strategy_id)
        buckets = [b for b in report["buckets"] if b["count"] > 0]
        if len(buckets) < 2:
            return 0.0

        # Simple Pearson correlation between avg_confidence and win_rate
        confs = [b["avg_confidence"] for b in buckets]
        rates = [b["win_rate"] for b in buckets]
        n = len(confs)

        mean_c = sum(confs) / n
        mean_r = sum(rates) / n

        cov = sum((c - mean_c) * (r - mean_r) for c, r in zip(confs, rates))
        var_c = sum((c - mean_c) ** 2 for c in confs)
        var_r = sum((r - mean_r) ** 2 for r in rates)

        denom = math.sqrt(var_c * var_r)
        if denom == 0:
            return 0.0
        return cov / denom

    def get_all_strategy_ids(self) -> list[str]:
        """Return all strategy IDs that have calibration data."""
        return [
            sid
            for sid, cal in self._data.items()
            if len(cal.observations) > 0
        ]
