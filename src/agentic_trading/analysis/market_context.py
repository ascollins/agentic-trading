"""Market context and macro regime assessment.

Provides a structured framework for categorising the macro environment
and its impact on crypto trading conditions, inspired by the
intermarket/macro analysis reference.

Five regimes are defined, each with a description, crypto impact
assessment, and positioning guidance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Current macro market context assessment."""

    risk_regime: str = "neutral"  # risk_on / risk_off / neutral
    dollar_trend: str = "neutral"  # strengthening / weakening / neutral
    yield_environment: str = "neutral"  # rising / falling / neutral
    liquidity_conditions: str = "normal"  # tightening / easing / normal
    crypto_specific: dict[str, Any] = field(default_factory=dict)
    regime_key: str = "neutral_ranging"
    summary: str = ""
    impact_on_crypto: str = ""
    positioning_guidance: str = ""


# Predefined macro regime descriptions
MACRO_REGIMES: dict[str, dict[str, str]] = {
    "risk_on_easing": {
        "description": "Central banks easing, risk assets favoured",
        "crypto_impact": "Strongly bullish — full exposure, favour high-beta altcoins",
        "positioning": "Full position sizing, wider trailing stops, trend-follow aggressively",
    },
    "risk_on_neutral": {
        "description": "Positive risk appetite, neutral monetary policy",
        "crypto_impact": "Moderately bullish — standard exposure, BTC-heavy",
        "positioning": "Standard position sizing, focus on higher-quality setups",
    },
    "risk_off_tightening": {
        "description": "Central banks tightening, risk aversion rising",
        "crypto_impact": "Bearish — reduced exposure, BTC-only or stablecoin heavy",
        "positioning": "50% or less of normal sizing, tight stops, counter-trend only at major support",
    },
    "risk_off_crisis": {
        "description": "Active risk-off event (geopolitical, financial)",
        "crypto_impact": "Highly bearish short-term, potential BTC safe-haven bid medium-term",
        "positioning": "Minimal exposure, cash preservation, probe long only at extreme levels",
    },
    "neutral_ranging": {
        "description": "No clear macro driver, markets range-bound",
        "crypto_impact": "Neutral — crypto-specific narratives dominate",
        "positioning": "Selective entries, mean-reversion setups, smaller sizing",
    },
}


def assess_macro_regime(
    context_inputs: dict[str, Any],
) -> MarketContext:
    """Assess macro market context from available inputs.

    Args:
        context_inputs: Dict with optional keys:

            - ``dxy_trend``: ``"up"`` / ``"down"`` / ``"flat"``
            - ``yields_10y_trend``: ``"up"`` / ``"down"`` / ``"flat"``
            - ``sp500_vs_200sma``: float ratio (>1 = above)
            - ``funding_rates_avg``: float (e.g. 0.01 for 0.01%/8h)
            - ``stablecoin_supply_trend``: ``"up"`` / ``"down"`` / ``"flat"``
            - ``btc_dominance``: float 0–100

    Returns:
        :class:`MarketContext` assessment.
    """
    dxy = context_inputs.get("dxy_trend", "flat")
    yields = context_inputs.get("yields_10y_trend", "flat")
    sp500_ratio = context_inputs.get("sp500_vs_200sma", 1.0)
    funding = context_inputs.get("funding_rates_avg", 0.0)
    stablecoin = context_inputs.get("stablecoin_supply_trend", "flat")

    # Risk regime
    if sp500_ratio > 1.02:
        risk = "risk_on"
    elif sp500_ratio < 0.98:
        risk = "risk_off"
    else:
        risk = "neutral"

    # Dollar trend
    dollar = (
        "strengthening"
        if dxy == "up"
        else "weakening"
        if dxy == "down"
        else "neutral"
    )

    # Yield environment
    yield_env = (
        "rising"
        if yields == "up"
        else "falling"
        if yields == "down"
        else "neutral"
    )

    # Liquidity conditions
    liquidity = (
        "easing"
        if stablecoin == "up"
        else "tightening"
        if stablecoin == "down"
        else "normal"
    )

    # Crypto-specific context
    crypto_ctx: dict[str, Any] = {}
    if funding > 0.01:
        crypto_ctx["funding_sentiment"] = "overheated"
    elif funding < -0.005:
        crypto_ctx["funding_sentiment"] = "fearful"
    else:
        crypto_ctx["funding_sentiment"] = "neutral"

    btc_dom = context_inputs.get("btc_dominance")
    if btc_dom is not None:
        crypto_ctx["btc_dominance"] = btc_dom
        if btc_dom > 55:
            crypto_ctx["rotation"] = "btc_dominant"
        elif btc_dom < 45:
            crypto_ctx["rotation"] = "alt_season"
        else:
            crypto_ctx["rotation"] = "balanced"

    # Map to regime
    if risk == "risk_on" and liquidity == "easing":
        regime_key = "risk_on_easing"
    elif risk == "risk_on":
        regime_key = "risk_on_neutral"
    elif risk == "risk_off" and dollar == "strengthening":
        regime_key = "risk_off_tightening"
    elif risk == "risk_off":
        regime_key = "risk_off_crisis"
    else:
        regime_key = "neutral_ranging"

    regime_info = MACRO_REGIMES.get(regime_key, MACRO_REGIMES["neutral_ranging"])

    return MarketContext(
        risk_regime=risk,
        dollar_trend=dollar,
        yield_environment=yield_env,
        liquidity_conditions=liquidity,
        crypto_specific=crypto_ctx,
        regime_key=regime_key,
        summary=regime_info["description"],
        impact_on_crypto=regime_info["crypto_impact"],
        positioning_guidance=regime_info["positioning"],
    )
