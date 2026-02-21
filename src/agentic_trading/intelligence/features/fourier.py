"""Fourier / FFT spectral analysis features (design spec §3.3).

Applies the Discrete Fourier Transform to a rolling window of close
prices and extracts the magnitude and phase of the dominant low-frequency
components.  These features capture cyclical price patterns (mean-reversion
intervals, recurring volatility cycles, seasonality).

Uses :func:`numpy.fft.rfft` — no additional dependencies required.

Integration point: called from :meth:`FeatureEngine.compute_features`
when the candle buffer has >= ``min_window`` points.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FourierExtractor:
    """Extract spectral features from a price series via FFT.

    Parameters
    ----------
    min_window:
        Minimum number of close prices before computing FFT.
    window_size:
        Number of recent observations used for the DFT.  Should be a
        power of 2 for optimal FFT performance (128 is default).
    num_components:
        Number of low-frequency components to extract (default 5).
        Components are ranked by magnitude; the top ``num_components``
        are returned.
    detrend:
        Whether to remove a linear trend before FFT (recommended for
        financial time series).
    """

    def __init__(
        self,
        min_window: int = 64,
        window_size: int = 128,
        num_components: int = 5,
        detrend: bool = True,
    ) -> None:
        self.min_window = min_window
        self.window_size = window_size
        self.num_components = num_components
        self.detrend = detrend

    def compute(self, closes: np.ndarray) -> dict[str, float]:
        """Compute FFT spectral features from close prices.

        Args:
            closes: 1-D array of close prices (oldest first).

        Returns:
            Dict with keys:
                ``fft_mag_1`` .. ``fft_mag_N``    — magnitudes of top N components
                ``fft_phase_1`` .. ``fft_phase_N``— phases (radians) of top N
                ``fft_period_1`` .. ``fft_period_N`` — period (in bars) of each
                ``fft_dominant_period``            — period of the single strongest component
                ``fft_spectral_entropy``           — entropy of the power spectrum (0 = single
                                                     frequency, high = noise)
        """
        n = len(closes)
        if n < self.min_window:
            return {}

        # Use the most recent window_size observations
        window = closes[-self.window_size:] if n >= self.window_size else closes.copy()
        window = window.astype(np.float64)
        wlen = len(window)

        # Optional detrending: remove linear trend to focus on cycles
        if self.detrend:
            window = self._detrend(window)

        # Apply Hann window to reduce spectral leakage
        hann = np.hanning(wlen)
        windowed = window * hann

        # Compute real FFT
        spectrum = np.fft.rfft(windowed)
        magnitudes = np.abs(spectrum)
        phases = np.angle(spectrum)

        # Frequency resolution
        freqs = np.fft.rfftfreq(wlen)  # cycles per sample

        # Skip DC component (index 0) and Nyquist
        if len(magnitudes) < 3:
            return {}

        mag_no_dc = magnitudes[1:]
        phase_no_dc = phases[1:]
        freq_no_dc = freqs[1:]

        features: dict[str, float] = {}

        # Find top N components by magnitude
        num_to_extract = min(self.num_components, len(mag_no_dc))
        top_indices = np.argsort(mag_no_dc)[::-1][:num_to_extract]

        for rank, idx in enumerate(top_indices, start=1):
            mag = float(mag_no_dc[idx])
            phase = float(phase_no_dc[idx])
            freq = float(freq_no_dc[idx])
            period = 1.0 / freq if freq > 0 else float("inf")

            features[f"fft_mag_{rank}"] = mag
            features[f"fft_phase_{rank}"] = phase
            features[f"fft_period_{rank}"] = period

        # Dominant period (strongest component)
        if len(top_indices) > 0:
            dom_idx = top_indices[0]
            dom_freq = float(freq_no_dc[dom_idx])
            features["fft_dominant_period"] = (
                1.0 / dom_freq if dom_freq > 0 else float("inf")
            )

        # Spectral entropy: measures how "spread out" energy is across frequencies
        total_power = float(np.sum(mag_no_dc ** 2))
        if total_power > 0:
            psd = (mag_no_dc ** 2) / total_power
            # Filter out zeros to avoid log(0)
            psd_nonzero = psd[psd > 0]
            entropy = -float(np.sum(psd_nonzero * np.log2(psd_nonzero)))
            # Normalise by max entropy (uniform distribution)
            max_entropy = np.log2(len(psd_nonzero)) if len(psd_nonzero) > 1 else 1.0
            features["fft_spectral_entropy"] = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            features["fft_spectral_entropy"] = 0.0

        return features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detrend(series: np.ndarray) -> np.ndarray:
        """Remove linear trend from a 1-D array."""
        n = len(series)
        x = np.arange(n, dtype=np.float64)
        mean_x = np.mean(x)
        mean_y = np.mean(series)
        denom = np.sum((x - mean_x) ** 2)
        if denom == 0:
            return series - mean_y
        slope = np.sum((x - mean_x) * (series - mean_y)) / denom
        intercept = mean_y - slope * mean_x
        return series - (slope * x + intercept)
