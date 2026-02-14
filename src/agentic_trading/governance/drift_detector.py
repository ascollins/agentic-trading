"""Live-vs-backtest metric drift detection.

Compares live strategy performance metrics against walk-forward baselines.
When drift exceeds configurable thresholds, the detector recommends
sizing reductions or strategy pauses.

Inspired by Soteria's Consistency Auditor (C10): detecting when a
deployed agent's behaviour diverges from its validated baseline.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from agentic_trading.core.config import DriftDetectorConfig
from agentic_trading.core.enums import GovernanceAction
from agentic_trading.core.events import DriftAlert

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects performance drift between live trading and backtest baselines.

    Usage::

        detector = DriftDetector(config)
        detector.set_baseline("trend_following", {
            "win_rate": 0.55,
            "sharpe": 1.8,
            "profit_factor": 1.6,
        })
        detector.update_live_metric("trend_following", "win_rate", 0.38)
        alerts = detector.check_drift("trend_following")
    """

    def __init__(self, config: DriftDetectorConfig) -> None:
        self._config = config
        self._baselines: dict[str, dict[str, float]] = {}
        self._live_metrics: dict[str, dict[str, float]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def set_baseline(
        self, strategy_id: str, metrics: dict[str, float]
    ) -> None:
        """Register backtest baseline metrics for a strategy.

        Args:
            strategy_id: Strategy identifier.
            metrics: Dict of metric_name → baseline_value.
                Only metrics in ``config.metrics_tracked`` are stored.
        """
        tracked = set(self._config.metrics_tracked)
        self._baselines[strategy_id] = {
            k: v for k, v in metrics.items() if k in tracked
        }
        logger.info(
            "Drift baseline set for %s: %s",
            strategy_id,
            self._baselines[strategy_id],
        )

    def load_baseline_from_backtest(
        self, strategy_id: str, backtest_result: Any
    ) -> None:
        """Extract baseline metrics from a backtest result object.

        Expects the result to have attributes matching tracked metric names.
        """
        metrics: dict[str, float] = {}
        for name in self._config.metrics_tracked:
            value = getattr(backtest_result, name, None)
            if value is not None:
                metrics[name] = float(value)
        if metrics:
            self.set_baseline(strategy_id, metrics)

    def has_baseline(self, strategy_id: str) -> bool:
        """Whether a baseline exists for the strategy."""
        return strategy_id in self._baselines

    # ------------------------------------------------------------------
    # Live metric updates
    # ------------------------------------------------------------------

    def update_live_metric(
        self,
        strategy_id: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Update a live metric value for a strategy."""
        if metric_name in set(self._config.metrics_tracked):
            self._live_metrics[strategy_id][metric_name] = value

    def update_live_metrics(
        self, strategy_id: str, metrics: dict[str, float]
    ) -> None:
        """Batch update live metrics."""
        for name, value in metrics.items():
            self.update_live_metric(strategy_id, name, value)

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(self, strategy_id: str) -> list[DriftAlert]:
        """Compare live metrics to baselines and return drift alerts.

        Returns:
            List of :class:`DriftAlert` for metrics exceeding thresholds.
            Empty list if no drift or no baseline set.
        """
        if strategy_id not in self._baselines:
            return []

        baseline = self._baselines[strategy_id]
        live = self._live_metrics.get(strategy_id, {})
        alerts: list[DriftAlert] = []

        for metric_name, baseline_val in baseline.items():
            live_val = live.get(metric_name)
            if live_val is None:
                continue

            deviation_pct = self._compute_deviation(baseline_val, live_val)

            if deviation_pct >= self._config.pause_threshold_pct:
                action = GovernanceAction.PAUSE
            elif deviation_pct >= self._config.deviation_threshold_pct:
                action = GovernanceAction.REDUCE_SIZE
            else:
                continue  # Within tolerance

            alert = DriftAlert(
                strategy_id=strategy_id,
                metric_name=metric_name,
                baseline_value=baseline_val,
                live_value=live_val,
                deviation_pct=round(deviation_pct, 2),
                action_taken=action,
            )
            alerts.append(alert)
            logger.warning(
                "Drift detected: %s.%s baseline=%.4f live=%.4f deviation=%.1f%% → %s",
                strategy_id,
                metric_name,
                baseline_val,
                live_val,
                deviation_pct,
                action.value,
            )

        return alerts

    def get_status(self, strategy_id: str) -> dict[str, Any]:
        """Return current drift status for a strategy."""
        baseline = self._baselines.get(strategy_id, {})
        live = self._live_metrics.get(strategy_id, {})
        status: dict[str, Any] = {
            "has_baseline": strategy_id in self._baselines,
            "metrics": {},
        }

        for metric_name in self._config.metrics_tracked:
            bl = baseline.get(metric_name)
            lv = live.get(metric_name)
            dev = (
                self._compute_deviation(bl, lv)
                if bl is not None and lv is not None
                else None
            )
            status["metrics"][metric_name] = {
                "baseline": bl,
                "live": lv,
                "deviation_pct": round(dev, 2) if dev is not None else None,
            }

        return status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_deviation(
        baseline: float, live: float
    ) -> float:
        """Compute percentage deviation from baseline.

        Returns absolute percentage deviation.
        """
        if baseline == 0:
            return 0.0 if live == 0 else 100.0
        return abs(live - baseline) / abs(baseline) * 100.0
