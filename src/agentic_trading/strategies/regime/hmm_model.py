"""HMM-based regime detection using hmmlearn.

Fits a Gaussian HMM on returns and volatility features.
Outputs regime (trend/range) and volatility state (high/low).
"""

from __future__ import annotations

import logging

import numpy as np

from agentic_trading.core.enums import RegimeType, VolatilityRegime

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    logger.warning("hmmlearn not installed; HMM regime detection unavailable")


class HMMRegimeModel:
    """Two-state Gaussian HMM for regime detection.

    State 0: range/low-vol
    State 1: trend/high-vol
    Mapped by examining the learned means.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 50) -> None:
        self._n_states = n_states
        self._n_iter = n_iter
        self._model: "GaussianHMM | None" = None
        self._fitted = False

    def fit(self, returns: list[float], volatilities: list[float]) -> None:
        """Fit the HMM on historical data."""
        if not HAS_HMMLEARN:
            return

        X = np.column_stack([returns, volatilities])
        if len(X) < 30:
            return

        self._model = GaussianHMM(
            n_components=self._n_states,
            covariance_type="full",
            n_iter=self._n_iter,
            random_state=42,
        )
        try:
            self._model.fit(X)
            self._fitted = True
        except Exception:
            logger.exception("HMM fitting failed")
            self._fitted = False

    def predict(
        self, returns: list[float], volatilities: list[float]
    ) -> tuple[RegimeType, VolatilityRegime, float]:
        """Predict current regime from recent data.

        Returns (regime_type, vol_regime, confidence).
        Auto-fits if not yet fitted.
        """
        if not HAS_HMMLEARN:
            return RegimeType.UNKNOWN, VolatilityRegime.UNKNOWN, 0.0

        X = np.column_stack([returns, volatilities])
        if len(X) < 30:
            return RegimeType.UNKNOWN, VolatilityRegime.UNKNOWN, 0.0

        if not self._fitted:
            self.fit(returns, volatilities)
        if not self._fitted or self._model is None:
            return RegimeType.UNKNOWN, VolatilityRegime.UNKNOWN, 0.0

        try:
            # Get state probabilities for the last observation
            log_prob, state_sequence = self._model.decode(X)
            posteriors = self._model.predict_proba(X)

            current_state = state_sequence[-1]
            current_probs = posteriors[-1]
            confidence = float(current_probs[current_state])

            # Map states to regime types by examining means
            # State with higher abs(mean return) → trend
            # State with lower abs(mean return) → range
            means = self._model.means_
            abs_return_means = [abs(m[0]) for m in means]

            trend_state = int(np.argmax(abs_return_means))
            range_state = 1 - trend_state

            if current_state == trend_state:
                regime = RegimeType.TREND
            else:
                regime = RegimeType.RANGE

            # Volatility: state with higher vol mean → high
            vol_means = [m[1] for m in means]
            high_vol_state = int(np.argmax(vol_means))

            if current_state == high_vol_state:
                vol_regime = VolatilityRegime.HIGH
            else:
                vol_regime = VolatilityRegime.LOW

            return regime, vol_regime, confidence

        except Exception:
            logger.exception("HMM prediction failed")
            return RegimeType.UNKNOWN, VolatilityRegime.UNKNOWN, 0.0
