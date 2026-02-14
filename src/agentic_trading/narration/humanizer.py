"""Transform raw strategy rationale into plain-English narration reasons.

Strategies produce rationale strings like ``"EMA12 > EMA26 | HTF aligned
bullish | ADX=27.3 | ATR=0.0045"`` — this module converts each fragment
into language suitable for the Bloomberg Presenter and the narration
service, stripping numeric jargon.
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------
# Pattern → human-readable mapping
# ------------------------------------------------------------------

_EMA_CROSS_UP = re.compile(r"EMA\d+\s*>\s*EMA\d+", re.IGNORECASE)
_EMA_CROSS_DOWN = re.compile(r"EMA\d+\s*<\s*EMA\d+", re.IGNORECASE)
_SMA_CROSS_UP = re.compile(r"SMA\d+\s*>\s*SMA\d+", re.IGNORECASE)
_SMA_CROSS_DOWN = re.compile(r"SMA\d+\s*<\s*SMA\d+", re.IGNORECASE)
_HTF_BULLISH = re.compile(r"HTF\s+aligned\s+bullish", re.IGNORECASE)
_HTF_BEARISH = re.compile(r"HTF\s+aligned\s+bearish", re.IGNORECASE)
_HTF_NOT_ALIGNED = re.compile(r"HTF\s+not\s+aligned", re.IGNORECASE)
_ADX_VALUE = re.compile(r"ADX\s*=\s*([\d.]+)", re.IGNORECASE)
_ATR_VALUE = re.compile(r"ATR\s*=\s*([\d.]+)", re.IGNORECASE)
_RANGE_REGIME = re.compile(r"range\s+regime", re.IGNORECASE)
_BB_DEVIATION = re.compile(r"price\s+(below|above)\s+lower\s+BB\s+by\s+([\d.]+)%", re.IGNORECASE)
_RSI_VALUE = re.compile(r"RSI\s*=\s*([\d.]+)", re.IGNORECASE)
_INDICATOR_EQUALS = re.compile(r"[A-Za-z_]{2,}\d*\s*=\s*[\d.]+")


def humanize_rationale(
    rationale: str,
    features_used: dict[str, float] | None = None,
) -> list[str]:
    """Convert a pipe-delimited rationale string into plain-English reasons.

    Parameters
    ----------
    rationale:
        Raw strategy rationale, e.g. ``"EMA12 > EMA26 | ADX=27.3"``.
    features_used:
        Optional feature dict for additional context.

    Returns
    -------
    list[str]
        Clean, human-readable reason strings.
    """
    if not rationale:
        return []

    parts = [p.strip() for p in rationale.split("|")]
    reasons: list[str] = []

    for part in parts:
        human = _humanize_part(part, features_used)
        if human:
            reasons.append(human)

    return reasons if reasons else ["Technical conditions favour this trade"]


def _humanize_part(
    part: str,
    features: dict[str, float] | None = None,
) -> str | None:
    """Convert a single rationale fragment to plain English.

    Returns None if the fragment should be dropped (e.g. ATR sizing data).
    """
    # EMA crossover
    if _EMA_CROSS_UP.search(part) or _SMA_CROSS_UP.search(part):
        return "Short-term momentum is bullish"
    if _EMA_CROSS_DOWN.search(part) or _SMA_CROSS_DOWN.search(part):
        return "Short-term momentum is bearish"

    # Higher timeframe alignment
    if _HTF_BULLISH.search(part):
        return "Higher timeframe trend confirms the direction"
    if _HTF_BEARISH.search(part):
        return "Higher timeframe trend confirms the direction"
    if _HTF_NOT_ALIGNED.search(part):
        return "Higher timeframe is not aligned, reducing conviction"

    # ADX — trend strength
    m = _ADX_VALUE.search(part)
    if m:
        val = float(m.group(1))
        if val < 20:
            return "Trend strength is weak"
        elif val < 40:
            return "Trend strength is moderate"
        else:
            return "Trend strength is strong"

    # ATR — internal sizing data, drop it
    if _ATR_VALUE.search(part):
        return None

    # Range regime
    if _RANGE_REGIME.search(part):
        return "Market is range-bound, reducing conviction"

    # Bollinger Band deviation
    m = _BB_DEVIATION.search(part)
    if m:
        direction = m.group(1)
        pct = float(m.group(2))
        if pct > 2.0:
            return f"Price is significantly {direction} the lower band"
        return f"Price is near the lower band"

    # RSI
    m = _RSI_VALUE.search(part)
    if m:
        val = float(m.group(1))
        if val < 30:
            return "Momentum is deeply oversold"
        elif val > 70:
            return "Momentum is overbought"
        else:
            return "Momentum is neutral"

    # Catch-all: if it looks like INDICATOR=NUMBER, drop it
    if _INDICATOR_EQUALS.fullmatch(part.strip()):
        return None

    # If it's already readable (no numeric jargon), keep it
    if not _INDICATOR_EQUALS.search(part):
        return part

    # Has embedded indicator values — use generic fallback
    return "Technical conditions support this view"
