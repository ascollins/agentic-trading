"""Overtrading detector — abnormal signal frequency analysis.

Detects when a strategy is generating signals at an abnormally high
rate relative to its historical baseline.  Overtrading is one of the
most common and destructive trading behaviours; catching it early
prevents unnecessary fee burn and emotional decision-making.

Uses a z-score approach: if the recent signal rate deviates more than
``threshold_z`` standard deviations from the rolling mean, an alert
is raised.

Usage::

    detector = OvertradingDetector(lookback=50, threshold_z=2.0)
    for trade in closed_trades:
        alert = detector.record_trade("trend", trade.opened_at)
        if alert:
            print(f"OVERTRADING: {alert}")

    report = detector.report("trend")
    print(report["is_overtrading"])  # True / False
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _StrategyFrequency:
    """Per-strategy signal frequency tracking."""

    # Timestamps of recent trades
    timestamps: deque[datetime] = field(default_factory=deque)
    # Rolling interval durations (hours between trades)
    intervals: deque[float] = field(default_factory=deque)
    # Alerts issued
    alert_count: int = 0
    last_alert: datetime | None = None


class OvertradingDetector:
    """Detect abnormally high trading frequency.

    Parameters
    ----------
    lookback : int
        Number of recent intervals to maintain for baseline.
        Default 50.
    threshold_z : float
        Z-score threshold for overtrading alert.  Default 2.0
        (roughly 2 standard deviations faster than average).
    cooldown_minutes : int
        Minimum minutes between consecutive alerts.  Default 60.
    min_samples : int
        Minimum number of intervals before alerting.  Default 10.
    """

    def __init__(
        self,
        *,
        lookback: int = 50,
        threshold_z: float = 2.0,
        cooldown_minutes: int = 60,
        min_samples: int = 10,
    ) -> None:
        self._lookback = max(5, lookback)
        self._threshold_z = threshold_z
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._min_samples = max(3, min_samples)
        self._data: dict[str, _StrategyFrequency] = defaultdict(
            _StrategyFrequency
        )

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record_trade(
        self,
        strategy_id: str,
        timestamp: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Record a trade event and check for overtrading.

        Parameters
        ----------
        strategy_id : str
            Strategy that generated the signal.
        timestamp : datetime | None
            When the trade was opened.  Uses utcnow() if None.

        Returns
        -------
        dict or None
            Alert dict if overtrading detected, None otherwise.
        """
        ts = timestamp or datetime.now(timezone.utc)
        freq = self._data[strategy_id]

        # Compute interval from previous trade
        if freq.timestamps:
            prev = freq.timestamps[-1]
            interval_hours = (ts - prev).total_seconds() / 3600.0
            if interval_hours > 0:
                freq.intervals.append(interval_hours)
                while len(freq.intervals) > self._lookback:
                    freq.intervals.popleft()

        freq.timestamps.append(ts)
        while len(freq.timestamps) > self._lookback + 1:
            freq.timestamps.popleft()

        # Check for overtrading
        if len(freq.intervals) < self._min_samples:
            return None

        # Cooldown check
        if freq.last_alert is not None:
            if (ts - freq.last_alert) < self._cooldown:
                return None

        intervals = list(freq.intervals)
        mean_interval = statistics.mean(intervals)
        if len(intervals) < 2:
            return None
        std_interval = statistics.stdev(intervals)
        if std_interval == 0:
            return None

        # The latest interval
        latest = intervals[-1]

        # Z-score: how many SDs faster is this trade than average?
        # Negative z = faster than average
        z_score = (latest - mean_interval) / std_interval

        # Overtrading = significantly shorter interval (negative z below threshold)
        if z_score < -self._threshold_z:
            freq.alert_count += 1
            freq.last_alert = ts
            alert = {
                "strategy_id": strategy_id,
                "z_score": round(z_score, 4),
                "latest_interval_hours": round(latest, 4),
                "mean_interval_hours": round(mean_interval, 4),
                "std_interval_hours": round(std_interval, 4),
                "trades_per_day_current": round(24.0 / latest, 2) if latest > 0 else float("inf"),
                "trades_per_day_normal": round(24.0 / mean_interval, 2) if mean_interval > 0 else 0.0,
                "alert_count": freq.alert_count,
                "timestamp": ts.isoformat(),
            }
            logger.warning(
                "Overtrading alert: %s z=%.2f (%.1fh vs avg %.1fh)",
                strategy_id,
                z_score,
                latest,
                mean_interval,
            )
            return alert

        return None

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def report(self, strategy_id: str) -> dict[str, Any]:
        """Generate a frequency report for a strategy."""
        freq = self._data.get(strategy_id)
        if freq is None or len(freq.intervals) == 0:
            return {
                "strategy_id": strategy_id,
                "is_overtrading": False,
                "trade_count": 0,
                "intervals_tracked": 0,
                "mean_interval_hours": 0.0,
                "current_trades_per_day": 0.0,
                "normal_trades_per_day": 0.0,
                "alert_count": 0,
            }

        intervals = list(freq.intervals)
        mean_int = statistics.mean(intervals)
        std_int = statistics.stdev(intervals) if len(intervals) >= 2 else 0.0
        latest = intervals[-1]

        z = (latest - mean_int) / std_int if std_int > 0 else 0.0

        return {
            "strategy_id": strategy_id,
            "is_overtrading": z < -self._threshold_z,
            "trade_count": len(freq.timestamps),
            "intervals_tracked": len(intervals),
            "mean_interval_hours": round(mean_int, 4),
            "std_interval_hours": round(std_int, 4),
            "latest_interval_hours": round(latest, 4),
            "z_score": round(z, 4),
            "current_trades_per_day": round(24.0 / latest, 2) if latest > 0 else 0.0,
            "normal_trades_per_day": round(24.0 / mean_int, 2) if mean_int > 0 else 0.0,
            "alert_count": freq.alert_count,
        }

    def is_overtrading(self, strategy_id: str) -> bool:
        """Quick check: is this strategy currently overtrading?"""
        report = self.report(strategy_id)
        return report["is_overtrading"]

    def get_all_strategy_ids(self) -> list[str]:
        """Return all strategy IDs with frequency data."""
        return [
            sid
            for sid, freq in self._data.items()
            if len(freq.timestamps) > 0
        ]
