"""Technical indicator wrappers.

Provides a uniform interface over tulipy where available, with pure-numpy
fallback implementations for every indicator.  All functions accept and
return numpy arrays.  Warmup periods that cannot be computed are filled
with ``np.nan`` so that array lengths always match the input length.
"""

from __future__ import annotations

import logging
from datetime import datetime

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


# ===================================================================
# Stochastic Oscillator
# ===================================================================

def compute_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator (%K and %D).

    Args:
        high: 1-D array of high prices.
        low: 1-D array of low prices.
        close: 1-D array of close prices.
        k_period: Lookback period for %K (typically 14).
        d_period: Smoothing period for %D (typically 3).

    Returns:
        Tuple of (%K, %D) arrays, each same length as *close*.
    """
    high = _ensure_float64(high)
    low = _ensure_float64(low)
    close = _ensure_float64(close)
    n = len(close)

    if _HAS_TULIPY:
        try:
            result = ti.stoch(high, low, close, k_period, k_period, d_period)
            k_line = _pad_front(result[0], n)
            d_line = _pad_front(result[1], n)
            return k_line, d_line
        except Exception:
            pass  # Fall through to numpy implementation

    k_line = np.full(n, np.nan)
    if n < k_period:
        return k_line, np.full(n, np.nan)

    for i in range(k_period - 1, n):
        highest = np.max(high[i - k_period + 1: i + 1])
        lowest = np.min(low[i - k_period + 1: i + 1])
        range_ = highest - lowest
        if range_ == 0:
            k_line[i] = 50.0
        else:
            k_line[i] = 100.0 * (close[i] - lowest) / range_

    # %D is SMA of %K
    d_line = np.full(n, np.nan)
    valid_k = ~np.isnan(k_line)
    valid_indices = np.where(valid_k)[0]
    if len(valid_indices) >= d_period:
        for i in range(d_period - 1, len(valid_indices)):
            idx = valid_indices[i]
            window_indices = valid_indices[i - d_period + 1: i + 1]
            d_line[idx] = np.mean(k_line[window_indices])

    return k_line, d_line


# ===================================================================
# On-Balance Volume (OBV)
# ===================================================================

def compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume.

    Args:
        close: 1-D array of close prices.
        volume: 1-D array of volume.

    Returns:
        Array of same length as *close* with cumulative OBV values.
    """
    close = _ensure_float64(close)
    volume = _ensure_float64(volume)
    n = len(close)

    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    return obv


# ===================================================================
# Keltner Channel
# ===================================================================

def compute_keltner(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channel (upper, middle, lower).

    Args:
        high: 1-D array of high prices.
        low: 1-D array of low prices.
        close: 1-D array of close prices.
        ema_period: Period for centre EMA (typically 20).
        atr_period: Period for ATR width (typically 14).
        atr_multiplier: Multiple of ATR for channel width (typically 1.5).

    Returns:
        Tuple of (upper, middle, lower) arrays, each same length as *close*.
    """
    middle = compute_ema(close, ema_period)
    atr = compute_atr(high, low, close, atr_period)

    upper = middle + atr_multiplier * atr
    lower = middle - atr_multiplier * atr

    return upper, middle, lower


# ===================================================================
# Bollinger Band Width
# ===================================================================

def compute_bbw(
    close: np.ndarray,
    period: int = 20,
    std: float = 2.0,
) -> np.ndarray:
    """Bollinger Band Width (normalised).

    BBW = (upper - lower) / middle

    Used for squeeze detection. Low BBW = consolidation / potential breakout.

    Args:
        close: 1-D array of close prices.
        period: BB lookback period.
        std: Number of standard deviations.

    Returns:
        Array of same length as *close* with BBW values.
    """
    upper, middle, lower = compute_bollinger_bands(close, period, std)
    n = len(close)
    bbw = np.full(n, np.nan)

    valid = ~np.isnan(middle) & (middle != 0)
    bbw[valid] = (upper[valid] - lower[valid]) / middle[valid]

    return bbw


# ===================================================================
# Fibonacci Levels
# ===================================================================

def compute_fibonacci_levels(
    swing_high: float,
    swing_low: float,
) -> dict[str, float]:
    """Compute standard Fibonacci retracement levels.

    Args:
        swing_high: Recent swing high price.
        swing_low: Recent swing low price.

    Returns:
        Dict mapping level name to price.
    """
    diff = swing_high - swing_low
    return {
        "fib_0": swing_high,
        "fib_236": swing_high - 0.236 * diff,
        "fib_382": swing_high - 0.382 * diff,
        "fib_500": swing_high - 0.500 * diff,
        "fib_618": swing_high - 0.618 * diff,
        "fib_786": swing_high - 0.786 * diff,
        "fib_1000": swing_low,
    }


def compute_fibonacci_extensions(
    swing_high: float,
    swing_low: float,
    retracement_point: float,
) -> dict[str, float]:
    """Compute Fibonacci extension levels from a retracement.

    Args:
        swing_high: Swing high price of the initial move.
        swing_low: Swing low price of the initial move.
        retracement_point: Price where the retracement ended.

    Returns:
        Dict mapping extension level name to price.
    """
    diff = swing_high - swing_low
    return {
        "ext_1000": retracement_point + diff,
        "ext_1272": retracement_point + 1.272 * diff,
        "ext_1618": retracement_point + 1.618 * diff,
        "ext_2000": retracement_point + 2.0 * diff,
        "ext_2618": retracement_point + 2.618 * diff,
    }


# ===================================================================
# Ichimoku Cloud
# ===================================================================

def compute_ichimoku(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> dict[str, np.ndarray]:
    """Ichimoku Kink\u014d Hy\u014d (Cloud).

    Computes all five Ichimoku lines.  Senkou Span A and B are *not*
    displaced forward in the returned arrays (the caller can shift if
    needed for charting; for feature-vector purposes the current values
    are more useful).

    Args:
        high: 1-D array of high prices.
        low: 1-D array of low prices.
        close: 1-D array of close prices.
        tenkan_period: Conversion line period (default 9).
        kijun_period: Base line period (default 26).
        senkou_b_period: Senkou Span B period (default 52).
        displacement: Chikou Span lookback (default 26).

    Returns:
        Dict with keys ``tenkan_sen``, ``kijun_sen``, ``senkou_span_a``,
        ``senkou_span_b``, ``chikou_span``.  Each value is an ndarray of
        the same length as *close* with leading NaN padding.
    """
    high = _ensure_float64(high)
    low = _ensure_float64(low)
    close = _ensure_float64(close)
    n = len(close)

    def _midpoint(h: np.ndarray, l: np.ndarray, period: int) -> np.ndarray:
        out = np.full(n, np.nan)
        for i in range(period - 1, n):
            out[i] = (np.max(h[i - period + 1 : i + 1]) + np.min(l[i - period + 1 : i + 1])) / 2.0
        return out

    tenkan = _midpoint(high, low, tenkan_period)
    kijun = _midpoint(high, low, kijun_period)
    senkou_a = np.where(
        ~np.isnan(tenkan) & ~np.isnan(kijun),
        (tenkan + kijun) / 2.0,
        np.nan,
    )
    senkou_b = _midpoint(high, low, senkou_b_period)

    # Chikou span: close shifted back by *displacement* bars
    chikou = np.full(n, np.nan)
    if n > displacement:
        chikou[:n - displacement] = close[displacement:]

    return {
        "tenkan_sen": tenkan,
        "kijun_sen": kijun,
        "senkou_span_a": senkou_a,
        "senkou_span_b": senkou_b,
        "chikou_span": chikou,
    }


# ===================================================================
# HyperWave Momentum Oscillator
# ===================================================================

def compute_hyperwave(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fast_period: int = 10,
    slow_period: int = 34,
    signal_period: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HyperWave momentum oscillator.

    A wave-based momentum indicator that combines True Range momentum
    with dual-EMA smoothing to produce a zero-centred oscillator line,
    a signal line, and a histogram.  Conceptually similar to MACD but
    operates on normalised momentum (close change / ATR) to be
    comparable across instruments and timeframes.

    Args:
        high: 1-D array of high prices.
        low: 1-D array of low prices.
        close: 1-D array of close prices.
        fast_period: Fast EMA period (default 10).
        slow_period: Slow EMA period (default 34).
        signal_period: Signal-line EMA period (default 5).

    Returns:
        Tuple of (wave, signal, histogram) arrays, each same length as
        *close*.
    """
    high = _ensure_float64(high)
    low = _ensure_float64(low)
    close = _ensure_float64(close)
    n = len(close)

    # Normalised momentum: price change / ATR (bounded ±∞ but typically ±3)
    atr = compute_atr(high, low, close, period=max(fast_period, 14))
    momentum = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            momentum[i] = (close[i] - close[i - 1]) / atr[i]

    # Fill leading NaN so EMA can operate on valid portion
    valid_mask = ~np.isnan(momentum)
    valid_momentum = momentum[valid_mask]

    if len(valid_momentum) < slow_period:
        empty = np.full(n, np.nan)
        return empty, empty.copy(), empty.copy()

    fast_ema = compute_ema(valid_momentum, fast_period)
    slow_ema = compute_ema(valid_momentum, slow_period)
    wave_raw = fast_ema - slow_ema

    # Signal line
    valid_wave = wave_raw[~np.isnan(wave_raw)]
    if len(valid_wave) < signal_period:
        empty = np.full(n, np.nan)
        return empty, empty.copy(), empty.copy()

    signal_raw = compute_ema(valid_wave, signal_period)

    # Map back to full-length arrays
    wave = np.full(n, np.nan)
    signal = np.full(n, np.nan)

    valid_indices = np.where(valid_mask)[0]
    wave[valid_indices] = wave_raw

    wave_valid_indices = valid_indices[~np.isnan(wave_raw)]
    if len(wave_valid_indices) >= len(signal_raw):
        signal[wave_valid_indices[-len(signal_raw):]] = signal_raw

    histogram = wave - signal

    return wave, signal, histogram


# ===================================================================
# Rate of Change (ROC)
# ===================================================================

def compute_roc(close: np.ndarray, period: int) -> np.ndarray:
    """Rate of Change: (close - close[n]) / close[n] * 100.

    Args:
        close: 1-D array of close prices.
        period: Lookback period.

    Returns:
        Array of same length as *close* with leading NaN padding.
    """
    close = _ensure_float64(close)
    n = len(close)
    roc = np.full(n, np.nan)

    if n <= period:
        return roc

    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0

    return roc


# ===================================================================
# Volume Delta (buy/sell pressure approximation)
# ===================================================================

def compute_volume_delta(
    open_prices: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    cumulative_period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Approximate volume delta from OHLCV candle data.

    Uses the sign of ``(close - open)`` to assign each bar's volume
    as buy-dominant or sell-dominant.  This is a proxy for true
    order-flow data (which requires tick-level trade data).

    Args:
        open_prices: 1-D array of open prices.
        close: 1-D array of close prices.
        volume: 1-D array of volume.
        cumulative_period: Rolling window for cumulative delta and ratio.

    Returns:
        Tuple of four arrays (same length as *close*):

        - **delta**: Per-bar volume delta (positive = buy-dominant).
        - **cumulative**: Rolling sum of delta over *cumulative_period*.
        - **ratio**: Rolling buy_volume / sell_volume over *cumulative_period*.
        - **trend**: Slope of cumulative delta (+1/0/-1 encoded).
    """
    open_prices = _ensure_float64(open_prices)
    close = _ensure_float64(close)
    volume = _ensure_float64(volume)
    n = len(close)

    # Per-bar delta: positive if close > open, negative otherwise
    direction = np.sign(close - open_prices)
    # If close == open (doji), split volume 50/50 → zero delta
    delta = volume * direction

    # Buy and sell volumes
    buy_vol = np.where(direction > 0, volume, 0.0)
    sell_vol = np.where(direction < 0, volume, 0.0)
    # Doji bars: split evenly
    doji_mask = direction == 0
    buy_vol[doji_mask] = volume[doji_mask] * 0.5
    sell_vol[doji_mask] = volume[doji_mask] * 0.5

    # Rolling cumulative delta
    cumulative = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    trend = np.full(n, 0.0)

    if n >= cumulative_period:
        for i in range(cumulative_period - 1, n):
            window = delta[i - cumulative_period + 1: i + 1]
            cumulative[i] = float(np.sum(window))

            buy_sum = float(np.sum(buy_vol[i - cumulative_period + 1: i + 1]))
            sell_sum = float(np.sum(sell_vol[i - cumulative_period + 1: i + 1]))
            if sell_sum > 0:
                ratio[i] = buy_sum / sell_sum
            elif buy_sum > 0:
                ratio[i] = 10.0  # Capped maximum
            else:
                ratio[i] = 1.0

        # Trend: simple regression slope of cumulative over last N bars
        valid_cum = cumulative[~np.isnan(cumulative)]
        if len(valid_cum) >= 5:
            recent = valid_cum[-min(10, len(valid_cum)):]
            x = np.arange(len(recent), dtype=np.float64)
            mean_x = np.mean(x)
            mean_y = np.mean(recent)
            denom = np.sum((x - mean_x) ** 2)
            if denom > 0:
                slope = float(np.sum((x - mean_x) * (recent - mean_y)) / denom)
                if slope > 0:
                    trend[-1] = 1.0
                elif slope < 0:
                    trend[-1] = -1.0

    return delta, cumulative, ratio, trend


# ===================================================================
# Session / Previous Period High-Low Levels
# ===================================================================

_SESSION_HOURS: dict[str, tuple[int, int]] = {
    "asia": (0, 8),       # 00:00-08:00 UTC
    "london": (8, 16),    # 08:00-16:00 UTC
    "new_york": (13, 21), # 13:00-21:00 UTC
}


def compute_session_levels(
    timestamps: list[datetime],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> dict[str, float]:
    """Compute previous session / day / week high-low levels.

    Returns a dict of features for the *latest* bar including
    previous-day, previous-week, and per-session high/low/close values.

    Args:
        timestamps: List of UTC-aware datetime objects (one per bar).
        highs: 1-D array of high prices.
        lows: 1-D array of low prices.
        closes: 1-D array of close prices.

    Returns:
        Dict with keys like ``prev_day_high``, ``prev_week_low``,
        ``prev_asia_high``, etc.  Values are ``nan`` when insufficient
        data.
    """
    from datetime import timezone  # local import to avoid circular

    result: dict[str, float] = {}
    n = len(timestamps)
    if n < 2:
        for key in (
            "prev_day_high", "prev_day_low", "prev_day_close",
            "prev_week_high", "prev_week_low", "prev_week_close",
        ):
            result[key] = float("nan")
        for sess in _SESSION_HOURS:
            result[f"prev_{sess}_high"] = float("nan")
            result[f"prev_{sess}_low"] = float("nan")
        return result

    current_ts = timestamps[-1]
    current_date = current_ts.date()
    current_week = current_date.isocalendar()[1]

    # --- Previous day ---
    prev_day_high = float("nan")
    prev_day_low = float("nan")
    prev_day_close = float("nan")
    prev_day_found = False
    for i in range(n - 2, -1, -1):
        bar_date = timestamps[i].date()
        if bar_date < current_date:
            if not prev_day_found:
                prev_day_high = float(highs[i])
                prev_day_low = float(lows[i])
                prev_day_close = float(closes[i])
                prev_day_found = True
                prev_day_date = bar_date
            elif bar_date == prev_day_date:
                prev_day_high = max(prev_day_high, float(highs[i]))
                prev_day_low = min(prev_day_low, float(lows[i]))
            else:
                break

    result["prev_day_high"] = prev_day_high
    result["prev_day_low"] = prev_day_low
    result["prev_day_close"] = prev_day_close

    # --- Previous week ---
    prev_week_high = float("nan")
    prev_week_low = float("nan")
    prev_week_close = float("nan")
    prev_week_found = False
    for i in range(n - 2, -1, -1):
        bar_week = timestamps[i].date().isocalendar()[1]
        bar_year = timestamps[i].date().isocalendar()[0]
        cur_year = current_date.isocalendar()[0]
        if bar_week < current_week or bar_year < cur_year:
            if not prev_week_found:
                prev_week_high = float(highs[i])
                prev_week_low = float(lows[i])
                prev_week_close = float(closes[i])
                prev_week_found = True
                prev_week_num = bar_week
                prev_week_year = bar_year
            elif bar_week == prev_week_num and bar_year == prev_week_year:
                prev_week_high = max(prev_week_high, float(highs[i]))
                prev_week_low = min(prev_week_low, float(lows[i]))
            else:
                break

    result["prev_week_high"] = prev_week_high
    result["prev_week_low"] = prev_week_low
    result["prev_week_close"] = prev_week_close

    # --- Previous session levels ---
    for sess_name, (start_h, end_h) in _SESSION_HOURS.items():
        sess_high = float("nan")
        sess_low = float("nan")
        found = False
        for i in range(n - 2, -1, -1):
            h = timestamps[i].hour
            in_session = start_h <= h < end_h
            bar_date = timestamps[i].date()
            # Must be from a previous session (either earlier today or yesterday)
            is_past = bar_date < current_date or (
                bar_date == current_date and h < timestamps[-1].hour
            )
            if in_session and is_past:
                if not found:
                    sess_high = float(highs[i])
                    sess_low = float(lows[i])
                    found = True
                    sess_date = bar_date
                elif bar_date == sess_date:
                    sess_high = max(sess_high, float(highs[i]))
                    sess_low = min(sess_low, float(lows[i]))
                else:
                    break
        result[f"prev_{sess_name}_high"] = sess_high
        result[f"prev_{sess_name}_low"] = sess_low

    return result
