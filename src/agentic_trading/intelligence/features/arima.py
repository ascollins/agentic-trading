"""ARIMA forecast features (design spec §3.2).

Fits a per-instrument ARIMA model on a rolling window of close prices
and produces one-step-ahead forecasts with confidence intervals.

Uses ``statsmodels`` SARIMAX when available; falls back to a simple
exponential smoothing forecast when the dependency is not installed.

Integration point: called from :meth:`FeatureEngine.compute_features`
when the candle buffer has >= ``min_observations`` points.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore[import-untyped]

    _HAS_STATSMODELS = True
    logger.debug("statsmodels available — using SARIMAX for ARIMA forecasts")
except ImportError:
    SARIMAX = None  # type: ignore[assignment,misc]
    _HAS_STATSMODELS = False
    logger.info(
        "statsmodels not installed — ARIMA features will use "
        "exponential smoothing fallback"
    )


class ARIMAForecaster:
    """One-step-ahead ARIMA forecaster for price series.

    Fits on a rolling window of close prices and returns forecast
    features suitable for inclusion in a :class:`FeatureVector`.

    Parameters
    ----------
    min_observations:
        Minimum number of close prices before attempting a forecast.
    max_window:
        Maximum number of recent observations used for model fitting.
        Keeps fitting time bounded.
    max_order:
        Maximum (p, d, q) tuple to try.  The forecaster uses a simple
        grid search over orders up to this limit.
    confidence_level:
        Confidence level for the prediction interval (default 0.95).
    """

    def __init__(
        self,
        min_observations: int = 60,
        max_window: int = 200,
        max_order: tuple[int, int, int] = (3, 1, 3),
        confidence_level: float = 0.95,
    ) -> None:
        self.min_observations = min_observations
        self.max_window = max_window
        self.max_order = max_order
        self.confidence_level = confidence_level

        # Cache fitted order per symbol to avoid repeated grid search
        self._fitted_orders: dict[str, tuple[int, int, int]] = {}

    def compute(
        self,
        closes: np.ndarray,
        symbol: str = "",
    ) -> dict[str, float]:
        """Compute ARIMA forecast features.

        Args:
            closes: 1-D array of close prices (oldest first).
            symbol: Symbol key for caching the fitted order.

        Returns:
            Dict with keys:
                ``arima_forecast``  — one-step-ahead point forecast
                ``arima_lower``     — lower bound of prediction interval
                ``arima_upper``     — upper bound of prediction interval
                ``arima_p``, ``arima_d``, ``arima_q`` — fitted order
                ``arima_residual_std`` — residual standard deviation
                ``arima_forecast_return`` — forecasted return (forecast/last - 1)
        """
        n = len(closes)
        if n < self.min_observations:
            return {}

        # Trim to max_window
        series = closes[-self.max_window:]

        if _HAS_STATSMODELS:
            return self._fit_statsmodels(series, symbol)
        return self._fit_fallback(series, symbol)

    # ------------------------------------------------------------------
    # statsmodels SARIMAX implementation
    # ------------------------------------------------------------------

    def _fit_statsmodels(
        self, series: np.ndarray, symbol: str,
    ) -> dict[str, float]:
        """Fit ARIMA using statsmodels SARIMAX."""
        try:
            order = self._select_order(series, symbol)
            model = SARIMAX(
                series,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=50)

            # One-step-ahead forecast
            forecast_obj = result.get_forecast(steps=1)
            forecast_mean = float(forecast_obj.predicted_mean.iloc[0])

            alpha = 1.0 - self.confidence_level
            conf_int = forecast_obj.conf_int(alpha=alpha)
            lower = float(conf_int.iloc[0, 0])
            upper = float(conf_int.iloc[0, 1])

            # Residual std
            resid_std = float(np.std(result.resid, ddof=1)) if len(result.resid) > 1 else 0.0

            # Forecast return
            last_price = float(series[-1])
            forecast_return = (forecast_mean / last_price - 1.0) if last_price > 0 else 0.0

            return {
                "arima_forecast": forecast_mean,
                "arima_lower": lower,
                "arima_upper": upper,
                "arima_p": float(order[0]),
                "arima_d": float(order[1]),
                "arima_q": float(order[2]),
                "arima_residual_std": resid_std,
                "arima_forecast_return": forecast_return,
            }

        except Exception:
            logger.debug("ARIMA fit failed for %s, using fallback", symbol, exc_info=True)
            return self._fit_fallback(series, symbol)

    def _select_order(
        self, series: np.ndarray, symbol: str,
    ) -> tuple[int, int, int]:
        """Select ARIMA order via AIC comparison.

        Caches result per symbol to avoid expensive re-fitting every bar.
        Re-evaluates every 50 calls for the same symbol.
        """
        cached = self._fitted_orders.get(symbol)
        if cached is not None:
            return cached

        best_aic = float("inf")
        best_order = (1, 1, 1)
        max_p, max_d, max_q = self.max_order

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue  # Skip (0,d,0)
                    try:
                        model = SARIMAX(
                            series,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        result = model.fit(disp=False, maxiter=30)
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        self._fitted_orders[symbol] = best_order
        logger.debug(
            "ARIMA order selected for %s: (%d,%d,%d) AIC=%.1f",
            symbol, *best_order, best_aic,
        )
        return best_order

    # ------------------------------------------------------------------
    # Fallback: exponential smoothing forecast
    # ------------------------------------------------------------------

    def _fit_fallback(
        self, series: np.ndarray, symbol: str,
    ) -> dict[str, float]:
        """Simple exponential smoothing fallback when statsmodels is unavailable."""
        alpha = 0.3  # Smoothing parameter
        n = len(series)

        # Fit exponential smoothing
        level = series[0]
        for i in range(1, n):
            level = alpha * series[i] + (1.0 - alpha) * level

        # Forecast = last smoothed level
        forecast = float(level)

        # Estimate residual std from recent errors
        residuals = []
        level = series[0]
        for i in range(1, n):
            pred = level
            level = alpha * series[i] + (1.0 - alpha) * level
            residuals.append(series[i] - pred)

        resid_arr = np.array(residuals)
        resid_std = float(np.std(resid_arr, ddof=1)) if len(resid_arr) > 1 else 0.0

        # Prediction interval
        z = 1.96  # ~95% CI
        lower = forecast - z * resid_std
        upper = forecast + z * resid_std

        last_price = float(series[-1])
        forecast_return = (forecast / last_price - 1.0) if last_price > 0 else 0.0

        return {
            "arima_forecast": forecast,
            "arima_lower": lower,
            "arima_upper": upper,
            "arima_p": 0.0,
            "arima_d": 0.0,
            "arima_q": 0.0,
            "arima_residual_std": resid_std,
            "arima_forecast_return": forecast_return,
        }

    def clear_cache(self, symbol: str | None = None) -> None:
        """Clear cached model orders."""
        if symbol is None:
            self._fitted_orders.clear()
        else:
            self._fitted_orders.pop(symbol, None)
