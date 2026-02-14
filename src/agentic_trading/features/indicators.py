"""Technical indicator wrappers.

Provides a uniform interface over tulipy where available, with pure-numpy
fallback implementations for every indicator.  All functions accept and
return numpy arrays.  Warmup periods that cannot be computed are filled
with ``np.nan`` so that array lengths always match the input length.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import tulipy; fall back gracefully
# ---------------------------------------------------------------------------

try:
    import tulipy as ti  # type: ignore[import-untyped]

    _HAS_TULIPY = True
    logger.debug("tulipy available - using native C implementations")
except ImportError:
    ti = None  # type: ignore[assignment]
    _HAS_TULIPY = False
    logger.info("tulipy not installed - using pure-numpy fallback indicators")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_front(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Left-pad *arr* with ``np.nan`` so its length equals *target_len*."""
    if arr.shape[0] >= target_len:
        return arr
    pad_size = target_len - arr.shape[0]
    return np.concatenate([np.full(pad_size, np.nan), arr])


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Ensure the array is contiguous float64 (required by tulipy)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


# ===================================================================
# Moving averages
# ===================================================================

def compute_ema(close_prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average.

    Args:
        close_prices: 1-D array of close prices.
        period: EMA lookback period.

    Returns:
        Array of same length as *close_prices* with leading NaN padding.
    """
    close_prices = _ensure_float64(close_prices)
    n = len(close_prices)

    if _HAS_TULIPY:
        result = ti.ema(close_prices, period=period)
        return _pad_front(result, n)

    # Pure-numpy fallback using the standard recursive formula.
    alpha = 2.0 / (period + 1)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    # Seed with SMA of first *period* values.
    ema[period - 1] = np.mean(close_prices[:period])
    for i in range(period, n):
        ema[i] = alpha * close_prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_sma(close_prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average.

    Args:
        close_prices: 1-D array of close prices.
        period: SMA lookback period.

    Returns:
        Array of same length as *close_prices* with leading NaN padding.
    """
    close_prices = _ensure_float64(close_prices)
    n = len(close_prices)

    if _HAS_TULIPY:
        result = ti.sma(close_prices, period=period)
        return _pad_front(result, n)

    sma = np.full(n, np.nan)
    if n < period:
        return sma
    cumsum = np.cumsum(close_prices)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    sma[period - 1:] = cumsum[period - 1:] / period
    return sma


# ===================================================================
# RSI
# ===================================================================

def compute_rsi(close_prices: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index (Wilder smoothing).

    Args:
        close_prices: 1-D array of close prices.
        period: RSI lookback period (typically 14).

    Returns:
        Array of same length as *close_prices* with leading NaN padding.
    """
    close_prices = _ensure_float64(close_prices)
    n = len(close_prices)

    if _HAS_TULIPY:
        result = ti.rsi(close_prices, period=period)
        return _pad_front(result, n)

    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    deltas = np.diff(close_prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed averages with simple mean of first *period* changes.
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing for the rest.
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


# ===================================================================
# Bollinger Bands
# ===================================================================

def compute_bollinger_bands(
    close: np.ndarray,
    period: int,
    std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands (upper, middle, lower).

    Args:
        close: 1-D array of close prices.
        period: Lookback period (typically 20).
        std: Number of standard deviations for the bands.

    Returns:
        Tuple of (upper, middle, lower) arrays, each same length as *close*.
    """
    close = _ensure_float64(close)
    n = len(close)

    if _HAS_TULIPY:
        bands = ti.bbands(close, period=period, stddev=std)
        # tulipy returns (lower, middle, upper) - reorder to (upper, middle, lower)
        lower = _pad_front(bands[0], n)
        middle = _pad_front(bands[1], n)
        upper = _pad_front(bands[2], n)
        return upper, middle, lower

    middle = compute_sma(close, period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = close[i - period + 1: i + 1]
        sd = np.std(window, ddof=0)
        upper[i] = middle[i] + std * sd
        lower[i] = middle[i] - std * sd

    return upper, middle, lower


# ===================================================================
# ADX (Average Directional Index)
# ===================================================================

def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Average Directional Index.

    Args:
        high: 1-D array of high prices.
        low: 1-D array of low prices.
        close: 1-D array of close prices.
        period: ADX lookback period (typically 14).

    Returns:
        Array of same length as *high* with leading NaN padding.
    """
    high = _ensure_float64(high)
    low = _ensure_float64(low)
    close = _ensure_float64(close)
    n = len(high)

    if _HAS_TULIPY:
        result = ti.adx(high, low, close, period=period)
        return _pad_front(result, n)

    # Pure-numpy Wilder-smoothed ADX implementation.
    adx_out = np.full(n, np.nan)
    if n < 2 * period:
        return adx_out

    # True Range, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h_l = high[i] - low[i]
        h_cp = abs(high[i] - close[i - 1])
        l_cp = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_cp, l_cp)

        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # Wilder smoothing for ATR, +DI, -DI
    atr_s = np.sum(tr[1: period + 1])
    plus_dm_s = np.sum(plus_dm[1: period + 1])
    minus_dm_s = np.sum(minus_dm[1: period + 1])

    dx_values: list[float] = []

    for i in range(period, n):
        if i == period:
            atr_val = atr_s
            pdm_val = plus_dm_s
            mdm_val = minus_dm_s
        else:
            atr_val = atr_val - atr_val / period + tr[i]
            pdm_val = pdm_val - pdm_val / period + plus_dm[i]
            mdm_val = mdm_val - mdm_val / period + minus_dm[i]

        if atr_val == 0:
            dx_values.append(0.0)
            continue

        plus_di = 100.0 * pdm_val / atr_val
        minus_di = 100.0 * mdm_val / atr_val
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_values.append(0.0)
        else:
            dx_values.append(100.0 * abs(plus_di - minus_di) / di_sum)

    # Smooth DX to get ADX
    if len(dx_values) < period:
        return adx_out

    adx_val = np.mean(dx_values[:period])
    adx_out[2 * period - 1] = adx_val
    for i in range(period, len(dx_values)):
        adx_val = (adx_val * (period - 1) + dx_values[i]) / period
        adx_out[period + i] = adx_val

    return adx_out


# ===================================================================
# ATR (Average True Range)
# ===================================================================

def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Average True Range (Wilder smoothing).

    Args:
        high: 1-D array of high prices.
        low: 1-D array of low prices.
        close: 1-D array of close prices.
        period: ATR lookback period (typically 14).

    Returns:
        Array of same length as *high* with leading NaN padding.
    """
    high = _ensure_float64(high)
    low = _ensure_float64(low)
    close = _ensure_float64(close)
    n = len(high)

    if _HAS_TULIPY:
        result = ti.atr(high, low, close, period=period)
        return _pad_front(result, n)

    atr_out = np.full(n, np.nan)
    if n < period + 1:
        return atr_out

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # First ATR is simple mean
    atr_val = np.mean(tr[1: period + 1])
    atr_out[period] = atr_val

    # Wilder smoothing
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        atr_out[i] = atr_val

    return atr_out


# ===================================================================
# MACD
# ===================================================================

def compute_macd(
    close: np.ndarray,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (Moving Average Convergence Divergence).

    Args:
        close: 1-D array of close prices.
        fast: Fast EMA period (typically 12).
        slow: Slow EMA period (typically 26).
        signal: Signal line EMA period (typically 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram) arrays,
        each same length as *close*.
    """
    close = _ensure_float64(close)
    n = len(close)

    if _HAS_TULIPY:
        result = ti.macd(close, short_period=fast, long_period=slow, signal_period=signal)
        macd_line = _pad_front(result[0], n)
        signal_line = _pad_front(result[1], n)
        histogram = _pad_front(result[2], n)
        return macd_line, signal_line, histogram

    fast_ema = compute_ema(close, fast)
    slow_ema = compute_ema(close, slow)
    macd_line = fast_ema - slow_ema

    # Signal line is EMA of the MACD line.  We need to compute it only
    # over the valid (non-NaN) portion.
    valid_mask = ~np.isnan(macd_line)
    valid_macd = macd_line[valid_mask]

    if len(valid_macd) < signal:
        signal_line = np.full(n, np.nan)
    else:
        raw_signal = compute_ema(valid_macd, signal)
        signal_line = np.full(n, np.nan)
        signal_line[valid_mask] = raw_signal

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# ===================================================================
# Donchian Channel
# ===================================================================

def compute_donchian(
    high: np.ndarray,
    low: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Donchian Channel (upper, lower).

    Not available in tulipy -- always uses numpy.

    Args:
        high: 1-D array of high prices.
        low: 1-D array of low prices.
        period: Lookback window (typically 20).

    Returns:
        Tuple of (upper, lower) arrays, each same length as *high*.
    """
    high = _ensure_float64(high)
    low = _ensure_float64(low)
    n = len(high)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1: i + 1])
        lower[i] = np.min(low[i - period + 1: i + 1])

    return upper, lower


# ===================================================================
# VWAP
# ===================================================================

def compute_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Volume-Weighted Average Price (cumulative, intra-session).

    Not available in tulipy -- always uses numpy.

    The standard VWAP is a cumulative running sum over the trading
    session.  For multi-day data the caller should reset / segment by
    session boundaries.

    Args:
        high: 1-D array of high prices.
        low: 1-D array of low prices.
        close: 1-D array of close prices.
        volume: 1-D array of volume.

    Returns:
        Array of same length as *high*.  Bars with zero cumulative
        volume are ``np.nan``.
    """
    high = _ensure_float64(high)
    low = _ensure_float64(low)
    close = _ensure_float64(close)
    volume = _ensure_float64(volume)

    typical_price = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)

    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    return vwap
