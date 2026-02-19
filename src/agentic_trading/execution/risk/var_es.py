"""Value-at-Risk and Expected Shortfall computation.

Provides historical VaR, parametric VaR, and Expected Shortfall (CVaR)
for portfolio risk assessment.  All methods are pure functions over
numpy arrays, making them safe to call from both live and backtest code.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class RiskMetrics:
    """Stateless calculator for VaR and Expected Shortfall.

    Usage::

        rm = RiskMetrics()
        var = rm.compute_var(daily_returns, confidence=0.95)
        es  = rm.compute_es(daily_returns, confidence=0.95)
    """

    # ------------------------------------------------------------------
    # Historical VaR
    # ------------------------------------------------------------------

    @staticmethod
    def compute_var(
        returns: ArrayLike,
        confidence: float = 0.95,
    ) -> float:
        """Compute historical Value-at-Risk.

        VaR is defined as the loss at the (1 - confidence) percentile of
        the empirical return distribution.  A positive VaR value means a
        loss of that magnitude is expected to be exceeded only
        ``(1 - confidence) * 100``% of the time.

        Args:
            returns: Array-like of period returns (e.g. daily log-returns).
            confidence: Confidence level in [0, 1].  Default 0.95.

        Returns:
            VaR expressed as a *positive* number representing loss.
            Returns 0.0 when the input array is empty or contains only NaN.
        """
        arr = np.asarray(returns, dtype=np.float64)
        arr = arr[~np.isnan(arr)]

        if arr.size == 0:
            logger.warning("compute_var called with empty return series")
            return 0.0

        percentile = (1.0 - confidence) * 100.0
        var_value: float = -float(np.percentile(arr, percentile))
        logger.debug(
            "Historical VaR(%.1f%%): %.6f over %d observations",
            confidence * 100,
            var_value,
            arr.size,
        )
        return var_value

    # ------------------------------------------------------------------
    # Parametric (Gaussian) VaR
    # ------------------------------------------------------------------

    @staticmethod
    def compute_parametric_var(
        mean: float,
        std: float,
        confidence: float = 0.95,
    ) -> float:
        """Compute parametric (variance-covariance) VaR assuming normal returns.

        Uses the inverse-normal to map the confidence level to a z-score
        and then computes ``-(mean + z * std)``.

        Args:
            mean: Expected return (e.g. daily mean).
            std: Standard deviation of returns.
            confidence: Confidence level in [0, 1].  Default 0.95.

        Returns:
            VaR as a positive number.  Returns 0.0 when *std* <= 0.
        """
        if std <= 0.0:
            logger.warning("compute_parametric_var: std=%.6f is non-positive", std)
            return 0.0

        from scipy.stats import norm  # lazy import -- only needed here

        z: float = norm.ppf(1.0 - confidence)
        var_value: float = -(mean + z * std)
        logger.debug(
            "Parametric VaR(%.1f%%): %.6f  (mean=%.6f, std=%.6f, z=%.4f)",
            confidence * 100,
            var_value,
            mean,
            std,
            z,
        )
        return var_value

    # ------------------------------------------------------------------
    # Expected Shortfall (CVaR)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_es(
        returns: ArrayLike,
        confidence: float = 0.95,
    ) -> float:
        """Compute Expected Shortfall (Conditional VaR).

        ES is the mean loss in the worst ``(1 - confidence)`` tail of the
        return distribution.  It is always >= VaR and captures the
        severity of tail losses better than VaR alone.

        Args:
            returns: Array-like of period returns.
            confidence: Confidence level in [0, 1].  Default 0.95.

        Returns:
            ES expressed as a *positive* number representing average
            tail loss.  Returns 0.0 when the input is empty / all NaN.
        """
        arr = np.asarray(returns, dtype=np.float64)
        arr = arr[~np.isnan(arr)]

        if arr.size == 0:
            logger.warning("compute_es called with empty return series")
            return 0.0

        percentile = (1.0 - confidence) * 100.0
        var_threshold = np.percentile(arr, percentile)
        tail = arr[arr <= var_threshold]

        if tail.size == 0:
            # Edge case: no observations below the threshold (very small sample).
            return float(-var_threshold)

        es_value: float = -float(np.mean(tail))
        logger.debug(
            "ES(%.1f%%): %.6f  (tail obs=%d, total obs=%d)",
            confidence * 100,
            es_value,
            tail.size,
            arr.size,
        )
        return es_value
