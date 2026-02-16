"""SMC Feature Computer — aggregates all SMC detections into flat feature dict.

Computes Smart Money Concepts features from a candle buffer and returns them
as a flat dictionary that can be merged into the FeatureVector.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agentic_trading.core.models import Candle

from .liquidity_sweeps import SweepType, detect_liquidity_sweeps
from .order_blocks import (
    BlockType,
    GapType,
    detect_fvgs,
    detect_order_blocks,
)
from .price_location import compute_dealing_range, classify_price_location
from .structure_breaks import (
    BreakDirection,
    BreakType,
    detect_all_structure_breaks,
)
from .swing_detection import (
    classify_structure,
    compute_swing_bias,
    detect_all_swings,
)

logger = logging.getLogger(__name__)


class SMCFeatureComputer:
    """Computes Smart Money Concepts features from a candle buffer.

    Returns a flat dict of feature keys prefixed with ``smc_``.
    These integrate with the existing FeatureVector pipeline.

    Usage::

        computer = SMCFeatureComputer()
        features = computer.compute(candles)
        # features = {"smc_swing_bias": 0.6, "smc_ob_count_bullish": 3, ...}
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        displacement_mult: float = 2.0,
        min_candles: int = 50,
    ) -> None:
        self._swing_lookback = swing_lookback
        self._displacement_mult = displacement_mult
        self._min_candles = min_candles

    def compute(self, candles: list[Candle]) -> dict[str, float]:
        """Compute all SMC features from a candle buffer.

        Args:
            candles: List of Candle objects, oldest first.

        Returns:
            Dict of SMC feature keys → float values.
        """
        if len(candles) < self._min_candles:
            return self._empty_features()

        # Extract OHLCV arrays
        opens = np.array([c.open for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)
        closes = np.array([c.close for c in candles], dtype=np.float64)
        volumes = np.array([c.volume for c in candles], dtype=np.float64)

        # Compute ATR for displacement detection
        atr = self._compute_atr(highs, lows, closes, period=14)

        features: dict[str, float] = {}

        # --- Swing detection ---
        swings = detect_all_swings(highs, lows, lookback=self._swing_lookback)
        classifications = classify_structure(swings)
        swing_bias = compute_swing_bias(classifications)

        features["smc_swing_bias"] = round(swing_bias, 4)
        features["smc_swing_count"] = float(len(swings))

        # Count recent structure labels (last 10 classifications)
        recent = classifications[-10:] if classifications else []
        features["smc_hh_count"] = float(
            sum(1 for c in recent if c.label.value == "HH")
        )
        features["smc_hl_count"] = float(
            sum(1 for c in recent if c.label.value == "HL")
        )
        features["smc_lh_count"] = float(
            sum(1 for c in recent if c.label.value == "LH")
        )
        features["smc_ll_count"] = float(
            sum(1 for c in recent if c.label.value == "LL")
        )

        # --- Order blocks ---
        obs = detect_order_blocks(
            opens, highs, lows, closes, volumes, atr,
            displacement_mult=self._displacement_mult,
        )
        bullish_obs = [ob for ob in obs if ob.block_type == BlockType.BULLISH]
        bearish_obs = [ob for ob in obs if ob.block_type == BlockType.BEARISH]
        unmitigated_bullish = [ob for ob in bullish_obs if not ob.is_mitigated]
        unmitigated_bearish = [ob for ob in bearish_obs if not ob.is_mitigated]

        features["smc_ob_count_bullish"] = float(len(bullish_obs))
        features["smc_ob_count_bearish"] = float(len(bearish_obs))
        features["smc_ob_unmitigated_bullish"] = float(len(unmitigated_bullish))
        features["smc_ob_unmitigated_bearish"] = float(len(unmitigated_bearish))

        # Distance to nearest unmitigated OB
        current_price = float(closes[-1])
        features["smc_nearest_demand_distance"] = self._nearest_ob_distance(
            unmitigated_bullish, current_price, "demand"
        )
        features["smc_nearest_supply_distance"] = self._nearest_ob_distance(
            unmitigated_bearish, current_price, "supply"
        )

        # --- Fair Value Gaps ---
        fvgs = detect_fvgs(highs, lows, closes)
        unfilled_bullish = [
            g for g in fvgs if g.gap_type == GapType.BULLISH and not g.is_filled
        ]
        unfilled_bearish = [
            g for g in fvgs if g.gap_type == GapType.BEARISH and not g.is_filled
        ]

        features["smc_fvg_count_bullish"] = float(len(unfilled_bullish))
        features["smc_fvg_count_bearish"] = float(len(unfilled_bearish))
        features["smc_fvg_count_total"] = float(
            len(unfilled_bullish) + len(unfilled_bearish)
        )

        # --- Structure breaks ---
        all_breaks = detect_all_structure_breaks(swings, closes)
        recent_breaks = all_breaks[-5:] if all_breaks else []

        bos_bullish = sum(
            1 for b in recent_breaks
            if b.break_type == BreakType.BOS
            and b.direction == BreakDirection.BULLISH
        )
        bos_bearish = sum(
            1 for b in recent_breaks
            if b.break_type == BreakType.BOS
            and b.direction == BreakDirection.BEARISH
        )
        choch_bullish = sum(
            1 for b in recent_breaks
            if b.break_type == BreakType.CHOCH
            and b.direction == BreakDirection.BULLISH
        )
        choch_bearish = sum(
            1 for b in recent_breaks
            if b.break_type == BreakType.CHOCH
            and b.direction == BreakDirection.BEARISH
        )

        features["smc_bos_bullish"] = float(bos_bullish)
        features["smc_bos_bearish"] = float(bos_bearish)
        features["smc_choch_bullish"] = float(choch_bullish)
        features["smc_choch_bearish"] = float(choch_bearish)

        # Last break direction: +1 bullish, -1 bearish, 0 none
        if all_breaks:
            last_break = all_breaks[-1]
            features["smc_last_break_direction"] = (
                1.0 if last_break.direction == BreakDirection.BULLISH else -1.0
            )
            features["smc_last_break_is_choch"] = (
                1.0 if last_break.break_type == BreakType.CHOCH else 0.0
            )
        else:
            features["smc_last_break_direction"] = 0.0
            features["smc_last_break_is_choch"] = 0.0

        # --- Liquidity sweeps ---
        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings, atr=atr,
        )

        recent_sweeps = sweeps[-10:] if sweeps else []
        bsl_count = sum(1 for s in recent_sweeps if s.sweep_type == SweepType.BSL)
        ssl_count = sum(1 for s in recent_sweeps if s.sweep_type == SweepType.SSL)
        bsl_confirmed = sum(
            1 for s in recent_sweeps
            if s.sweep_type == SweepType.BSL and s.reversal_confirmed
        )
        ssl_confirmed = sum(
            1 for s in recent_sweeps
            if s.sweep_type == SweepType.SSL and s.reversal_confirmed
        )

        features["smc_bsl_count"] = float(bsl_count)
        features["smc_ssl_count"] = float(ssl_count)
        features["smc_bsl_confirmed_count"] = float(bsl_confirmed)
        features["smc_ssl_confirmed_count"] = float(ssl_confirmed)

        if sweeps:
            last_sweep = sweeps[-1]
            features["smc_last_sweep_type"] = (
                1.0 if last_sweep.sweep_type == SweepType.BSL else -1.0
            )
            features["smc_last_sweep_bars_ago"] = float(
                len(closes) - 1 - last_sweep.index
            )
            features["smc_last_sweep_penetration"] = last_sweep.penetration_pct
            features["smc_sweep_reversal_confirmed"] = (
                1.0 if last_sweep.reversal_confirmed else 0.0
            )
        else:
            features["smc_last_sweep_type"] = 0.0
            features["smc_last_sweep_bars_ago"] = 0.0
            features["smc_last_sweep_penetration"] = 0.0
            features["smc_sweep_reversal_confirmed"] = 0.0

        # --- Price location (premium / discount zones) ---
        last_atr = float(atr[-1]) if not np.isnan(atr[-1]) else None
        dealing_range = compute_dealing_range(swings, atr_value=last_atr)

        if dealing_range is not None:
            location = classify_price_location(
                current_price, dealing_range[0], dealing_range[1],
            )
            zone_encoding = {
                "deep_discount": -2.0,
                "discount": -1.0,
                "equilibrium": 0.0,
                "premium": 1.0,
                "deep_premium": 2.0,
            }
            features["smc_equilibrium"] = location.equilibrium
            features["smc_dealing_range_high"] = location.dealing_range_high
            features["smc_dealing_range_low"] = location.dealing_range_low
            features["smc_price_zone"] = zone_encoding.get(
                location.zone.value, 0.0,
            )
            features["smc_deviation_from_eq"] = location.deviation_pct
            features["smc_range_position"] = location.range_position_pct
            features["smc_in_ote"] = 1.0 if location.in_ote else 0.0
            # OTE alignment: +1 if OTE supports longs, -1 if shorts
            if location.in_ote:
                features["smc_ote_alignment"] = (
                    1.0 if location.range_position_pct < 0.5 else -1.0
                )
            else:
                features["smc_ote_alignment"] = 0.0
        else:
            features["smc_equilibrium"] = 0.0
            features["smc_dealing_range_high"] = 0.0
            features["smc_dealing_range_low"] = 0.0
            features["smc_price_zone"] = 0.0
            features["smc_deviation_from_eq"] = 0.0
            features["smc_range_position"] = 0.0
            features["smc_in_ote"] = 0.0
            features["smc_ote_alignment"] = 0.0

        # --- SMC Confluence Score (0-14 scale from reference docs) ---
        features["smc_confluence_score"] = self._compute_confluence_score(
            swing_bias, unmitigated_bullish, unmitigated_bearish,
            unfilled_bullish, unfilled_bearish, recent_breaks, current_price,
            sweeps=sweeps, in_ote=features.get("smc_in_ote", 0.0),
        )

        return features

    @staticmethod
    def _empty_features() -> dict[str, float]:
        """Return zeroed SMC features when insufficient data."""
        return {
            "smc_swing_bias": 0.0,
            "smc_swing_count": 0.0,
            "smc_hh_count": 0.0,
            "smc_hl_count": 0.0,
            "smc_lh_count": 0.0,
            "smc_ll_count": 0.0,
            "smc_ob_count_bullish": 0.0,
            "smc_ob_count_bearish": 0.0,
            "smc_ob_unmitigated_bullish": 0.0,
            "smc_ob_unmitigated_bearish": 0.0,
            "smc_nearest_demand_distance": 0.0,
            "smc_nearest_supply_distance": 0.0,
            "smc_fvg_count_bullish": 0.0,
            "smc_fvg_count_bearish": 0.0,
            "smc_fvg_count_total": 0.0,
            "smc_bos_bullish": 0.0,
            "smc_bos_bearish": 0.0,
            "smc_choch_bullish": 0.0,
            "smc_choch_bearish": 0.0,
            "smc_last_break_direction": 0.0,
            "smc_last_break_is_choch": 0.0,
            # Liquidity sweeps
            "smc_bsl_count": 0.0,
            "smc_ssl_count": 0.0,
            "smc_bsl_confirmed_count": 0.0,
            "smc_ssl_confirmed_count": 0.0,
            "smc_last_sweep_type": 0.0,
            "smc_last_sweep_bars_ago": 0.0,
            "smc_last_sweep_penetration": 0.0,
            "smc_sweep_reversal_confirmed": 0.0,
            # Price location
            "smc_equilibrium": 0.0,
            "smc_dealing_range_high": 0.0,
            "smc_dealing_range_low": 0.0,
            "smc_price_zone": 0.0,
            "smc_deviation_from_eq": 0.0,
            "smc_range_position": 0.0,
            "smc_in_ote": 0.0,
            "smc_ote_alignment": 0.0,
            # Confluence
            "smc_confluence_score": 0.0,
        }

    @staticmethod
    def _compute_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Compute ATR for the candle buffer."""
        n = len(closes)
        atr = np.full(n, np.nan)
        if n < period + 1:
            return atr

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        # Prepend NaN for index 0
        tr = np.concatenate([[np.nan], tr])

        # Simple moving average for initial ATR
        for i in range(period, n):
            atr[i] = float(np.nanmean(tr[i - period + 1 : i + 1]))

        return atr

    @staticmethod
    def _nearest_ob_distance(
        obs: list, current_price: float, zone_type: str
    ) -> float:
        """Compute distance to the nearest unmitigated OB as % of price."""
        if not obs or current_price <= 0:
            return 0.0

        distances = []
        for ob in obs:
            mid = (ob.ob_high + ob.ob_low) / 2.0
            dist = abs(current_price - mid) / current_price
            distances.append(dist)

        return round(min(distances), 6) if distances else 0.0

    @staticmethod
    def _compute_confluence_score(
        swing_bias: float,
        bullish_obs: list,
        bearish_obs: list,
        bullish_fvgs: list,
        bearish_fvgs: list,
        recent_breaks: list,
        current_price: float,
        sweeps: list | None = None,
        in_ote: float = 0.0,
    ) -> float:
        """Compute SMC confluence score (0-14 scale).

        Scoring:
        - HTF bias aligned: 0-3 points (swing bias magnitude)
        - Unmitigated OB at entry: 0-2 points
        - FVG overlapping entry zone: 0-2 points
        - Liquidity swept: 0-3 points (actual BSL/SSL sweeps)
        - OTE alignment: 0-2 points (price in Optimal Trade Entry zone)
        - Structure breaks: 0-2 points (BOS/CHoCH confirmation)
        """
        score = 0.0

        # Swing bias alignment (proxy for HTF bias): 0-3 points
        score += min(3.0, abs(swing_bias) * 3.0)

        # Unmitigated order blocks present: 0-2 points
        ob_count = len(bullish_obs) + len(bearish_obs)
        score += min(2.0, ob_count * 0.5)

        # Fair value gaps present: 0-2 points
        fvg_count = len(bullish_fvgs) + len(bearish_fvgs)
        score += min(2.0, fvg_count * 0.4)

        # Liquidity sweeps: 0-3 points (actual sweep detection)
        if sweeps:
            confirmed_sweeps = sum(1 for s in sweeps if s.reversal_confirmed)
            unconfirmed_sweeps = len(sweeps) - confirmed_sweeps
            score += min(3.0, confirmed_sweeps * 1.5 + unconfirmed_sweeps * 0.5)
        else:
            # Fallback: use structure breaks as proxy
            from .structure_breaks import BreakType

            choch_count = sum(
                1 for b in recent_breaks if b.break_type == BreakType.CHOCH
            )
            bos_count = sum(
                1 for b in recent_breaks if b.break_type == BreakType.BOS
            )
            score += min(3.0, choch_count * 1.5 + bos_count * 0.5)

        # OTE alignment: 0-2 points
        if in_ote > 0:
            score += 2.0

        return round(min(14.0, score), 2)
