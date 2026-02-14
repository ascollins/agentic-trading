"""Walk-forward validation framework.

Splits data into rolling train/test windows and evaluates strategy performance
with proper temporal ordering. Detects overfitting via in-sample vs out-of-sample
performance comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """A single train/test split."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    fold_index: int


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""

    fold_index: int
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    train_max_dd: float
    test_max_dd: float
    params: dict[str, Any] = field(default_factory=dict)
    overfit_ratio: float = 0.0  # test_sharpe / train_sharpe


@dataclass
class WalkForwardReport:
    """Aggregate walk-forward results."""

    folds: list[WalkForwardResult]
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    overfit_score: float = 0.0  # 1 - avg(test/train) ratio
    degradation_pct: float = 0.0
    is_overfit: bool = False


class WalkForwardValidator:
    """Rolling walk-forward evaluation.

    Creates N folds of train/test splits with anchored or rolling windows.
    Evaluates strategy on each fold and computes overfit metrics.
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_pct: float = 0.7,
        gap_periods: int = 1,  # Gap between train and test (prevent leakage)
        anchored: bool = False,  # If True, train start is always the first date
    ) -> None:
        self._n_folds = n_folds
        self._train_pct = train_pct
        self._gap = gap_periods
        self._anchored = anchored

    def create_windows(
        self, timestamps: list[datetime]
    ) -> list[WalkForwardWindow]:
        """Create walk-forward windows from a list of timestamps."""
        n = len(timestamps)
        if n < 100:
            raise ValueError(f"Need at least 100 data points, got {n}")

        fold_size = n // self._n_folds
        windows = []

        for i in range(self._n_folds):
            if self._anchored:
                train_start_idx = 0
            else:
                train_start_idx = i * fold_size

            test_end_idx = min((i + 1) * fold_size + fold_size, n - 1)
            split_idx = train_start_idx + int(
                (test_end_idx - train_start_idx) * self._train_pct
            )

            train_end_idx = split_idx
            test_start_idx = min(split_idx + self._gap, test_end_idx)

            if test_start_idx >= test_end_idx:
                continue

            windows.append(
                WalkForwardWindow(
                    train_start=timestamps[train_start_idx],
                    train_end=timestamps[train_end_idx],
                    test_start=timestamps[test_start_idx],
                    test_end=timestamps[test_end_idx],
                    fold_index=i,
                )
            )

        return windows

    @staticmethod
    def compute_sharpe(returns: list[float], periods_per_year: int = 252) -> float:
        """Annualized Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std == 0:
            return 0.0
        return mean / std * np.sqrt(periods_per_year)

    @staticmethod
    def compute_max_drawdown(returns: list[float]) -> float:
        """Maximum drawdown from returns."""
        if not returns:
            return 0.0
        cumulative = np.cumprod(1.0 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.min(drawdowns))

    def evaluate(self, fold_results: list[WalkForwardResult]) -> WalkForwardReport:
        """Compute aggregate walk-forward metrics."""
        if not fold_results:
            return WalkForwardReport(folds=[])

        # Compute overfit ratios
        for r in fold_results:
            if r.train_sharpe > 0:
                r.overfit_ratio = r.test_sharpe / r.train_sharpe
            else:
                r.overfit_ratio = 0.0

        avg_train = float(np.mean([r.train_sharpe for r in fold_results]))
        avg_test = float(np.mean([r.test_sharpe for r in fold_results]))

        if avg_train > 0:
            overfit_score = 1.0 - (avg_test / avg_train)
            degradation = (avg_train - avg_test) / avg_train * 100
        else:
            overfit_score = 1.0
            degradation = 100.0

        return WalkForwardReport(
            folds=fold_results,
            avg_train_sharpe=round(avg_train, 4),
            avg_test_sharpe=round(avg_test, 4),
            overfit_score=round(max(0, overfit_score), 4),
            degradation_pct=round(degradation, 2),
            is_overfit=overfit_score > 0.5,  # >50% degradation = overfit
        )
