"""Multi-timeframe SMC confluence scorer.

Scores how well Smart Money Concepts signals align across multiple
timeframes.  Consumes prefixed features from
:class:`~agentic_trading.features.multi_timeframe.MultiTimeframeAligner`
(e.g., ``"4h_smc_swing_bias"``) and produces a unified confluence
assessment.

Usage::

    scorer = SMCConfluenceScorer()
    result = scorer.score("BTC/USDT", aligned_features)
    print(result.total_confluence_points, result.htf_bias)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentic_trading.core.enums import MarketStructureBias, Timeframe

logger = logging.getLogger(__name__)

# Timeframe hierarchy for HTF analysis (highest to lowest)
_TF_HIERARCHY: list[Timeframe] = [
    Timeframe.D1,
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
]

# Weight per position in hierarchy (higher TF = more weight)
_TF_WEIGHTS: dict[Timeframe, float] = {
    Timeframe.D1: 6.0,
    Timeframe.H4: 5.0,
    Timeframe.H1: 4.0,
    Timeframe.M15: 3.0,
    Timeframe.M5: 2.0,
    Timeframe.M1: 1.0,
}


@dataclass
class SMCTimeframeSummary:
    """SMC-specific assessment for a single timeframe."""

    timeframe: Timeframe
    swing_bias: float = 0.0  # smc_swing_bias [-1, +1]
    trend_label: MarketStructureBias = MarketStructureBias.UNCLEAR

    # Structure breaks
    last_break_type: str = "none"  # "BOS" / "CHoCH" / "none"
    last_break_direction: str = "none"  # "bullish" / "bearish" / "none"
    bos_bullish: int = 0
    bos_bearish: int = 0
    choch_bullish: int = 0
    choch_bearish: int = 0

    # Order blocks & FVGs
    unmitigated_demand_zones: int = 0
    unmitigated_supply_zones: int = 0
    unfilled_bullish_fvgs: int = 0
    unfilled_bearish_fvgs: int = 0
    nearest_demand_distance: float = 0.0
    nearest_supply_distance: float = 0.0

    # Price location
    price_zone: str = "unknown"  # "deep_discount" .. "deep_premium"
    equilibrium: float = 0.0
    in_ote: bool = False

    # Liquidity sweeps
    bsl_sweeps: int = 0
    ssl_sweeps: int = 0
    last_sweep_type: str = "none"  # "BSL" / "SSL" / "none"

    # Confluence
    confluence_score: float = 0.0

    # Observations
    key_observations: list[str] = field(default_factory=list)


@dataclass
class SMCConfluenceResult:
    """Cross-timeframe SMC confluence assessment."""

    symbol: str
    htf_bias: MarketStructureBias = MarketStructureBias.UNCLEAR
    ltf_confirmation: bool = False
    bias_alignment_score: float = 0.0  # 0-1 SMC-specific alignment
    structure_alignment: str = "unclear"  # "aligned" / "divergent" / "transitioning"

    timeframe_summaries: list[SMCTimeframeSummary] = field(default_factory=list)

    # Confluence factors
    htf_ob_at_ltf_entry: bool = False  # HTF OB zone coincides with LTF price
    fvg_confluence: bool = False  # FVGs align across TFs
    liquidity_swept: bool = False  # Recent sweep on any TF

    # Scoring
    total_confluence_points: float = 0.0  # 0-14 scale
    confluence_factors: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    # Key levels from SMC
    nearest_demand_price: float | None = None
    nearest_supply_price: float | None = None
    equilibrium_price: float | None = None
    dealing_range_high: float | None = None
    dealing_range_low: float | None = None


class SMCConfluenceScorer:
    """Scores SMC signal alignment across multiple timeframes.

    Consumes prefixed features from ``MultiTimeframeAligner``
    (e.g., ``"4h_smc_swing_bias"``) and produces a unified SMC
    confluence assessment.

    Scoring system (0-14 total):

    - HTF bias alignment:       0-3 points (swing_bias on highest TF)
    - Structure alignment:      0-2 points (HTF + LTF BOS/CHoCH agree)
    - Order block confluence:   0-2 points (HTF OB zone near LTF price)
    - FVG confluence:           0-2 points (unfilled FVGs align across TFs)
    - Liquidity sweep:          0-3 points (confirmed sweep with reversal)
    - OTE alignment:            0-2 points (price in OTE zone with bias)
    """

    def __init__(
        self,
        htf_timeframes: list[Timeframe] | None = None,
        ltf_timeframes: list[Timeframe] | None = None,
        ob_proximity_threshold: float = 0.02,
    ) -> None:
        self._htf_timeframes = htf_timeframes or [Timeframe.D1, Timeframe.H4]
        self._ltf_timeframes = ltf_timeframes or [Timeframe.H1, Timeframe.M15, Timeframe.M5]
        self._ob_proximity = ob_proximity_threshold

    def score(
        self,
        symbol: str,
        aligned_features: dict[str, float],
        available_timeframes: list[Timeframe] | None = None,
    ) -> SMCConfluenceResult:
        """Produce a cross-timeframe SMC confluence assessment.

        Args:
            symbol: Trading pair.
            aligned_features: Prefixed feature dict from
                ``MultiTimeframeAligner`` (e.g., ``"4h_smc_swing_bias"``).
                Also accepts non-prefixed features (single-TF mode).
            available_timeframes: Which timeframes have data.  If *None*,
                inferred from feature key prefixes.

        Returns:
            :class:`SMCConfluenceResult` with per-timeframe summaries,
            cross-TF confluence scoring, and key levels.
        """
        if available_timeframes is None:
            available_timeframes = self._infer_timeframes(aligned_features)

        # If no timeframe prefixes detected, treat as single-TF
        if not available_timeframes:
            single_summary = self._extract_smc_summary(
                "", Timeframe.M1, aligned_features,
            )
            return SMCConfluenceResult(
                symbol=symbol,
                htf_bias=single_summary.trend_label,
                timeframe_summaries=[single_summary],
                total_confluence_points=single_summary.confluence_score,
            )

        # Build per-TF summaries
        summaries: list[SMCTimeframeSummary] = []
        for tf in _TF_HIERARCHY:
            if tf not in available_timeframes:
                continue
            prefix = tf.value
            summary = self._extract_smc_summary(prefix, tf, aligned_features)
            summaries.append(summary)

        if not summaries:
            return SMCConfluenceResult(symbol=symbol)

        # Score confluence
        total_points, factors, conflicts = self._score_confluence(
            summaries, aligned_features,
        )

        # Determine HTF bias from the highest available TF
        htf_summary = summaries[0]
        htf_bias = htf_summary.trend_label

        # LTF confirmation: does the lowest TF agree with HTF?
        ltf_summary = summaries[-1] if len(summaries) > 1 else summaries[0]
        ltf_confirmation = (
            htf_bias == ltf_summary.trend_label
            and htf_bias != MarketStructureBias.UNCLEAR
        )

        # Bias alignment score (0-1)
        alignment = self._compute_alignment(summaries)

        # Structure alignment label
        structure_alignment = self._classify_structure_alignment(summaries)

        # OB confluence: does HTF have OBs near LTF price?
        htf_ob = (
            htf_summary.nearest_demand_distance < self._ob_proximity
            or htf_summary.nearest_supply_distance < self._ob_proximity
        )

        # FVG confluence: multiple TFs have unfilled FVGs
        tfs_with_fvg = sum(
            1 for s in summaries
            if s.unfilled_bullish_fvgs > 0 or s.unfilled_bearish_fvgs > 0
        )
        fvg_confluence = tfs_with_fvg >= 2

        # Liquidity swept on any TF
        liquidity_swept = any(
            s.bsl_sweeps > 0 or s.ssl_sweeps > 0 for s in summaries
        )

        # Key levels from the highest TF
        result = SMCConfluenceResult(
            symbol=symbol,
            htf_bias=htf_bias,
            ltf_confirmation=ltf_confirmation,
            bias_alignment_score=round(alignment, 2),
            structure_alignment=structure_alignment,
            timeframe_summaries=summaries,
            htf_ob_at_ltf_entry=htf_ob,
            fvg_confluence=fvg_confluence,
            liquidity_swept=liquidity_swept,
            total_confluence_points=round(total_points, 2),
            confluence_factors=factors,
            conflicts=conflicts,
        )

        # Extract key levels from highest TF summary
        if htf_summary.equilibrium > 0:
            result.equilibrium_price = htf_summary.equilibrium

        htf_prefix = htf_summary.timeframe.value
        dr_high = aligned_features.get(f"{htf_prefix}_smc_dealing_range_high", 0.0)
        dr_low = aligned_features.get(f"{htf_prefix}_smc_dealing_range_low", 0.0)
        if dr_high > 0:
            result.dealing_range_high = dr_high
        if dr_low > 0:
            result.dealing_range_low = dr_low

        return result

    def _extract_smc_summary(
        self,
        prefix: str,
        tf: Timeframe,
        features: dict[str, float],
    ) -> SMCTimeframeSummary:
        """Extract SMC summary for one timeframe from prefixed features."""
        p = f"{prefix}_" if prefix else ""

        def _get(key: str, default: float = 0.0) -> float:
            return features.get(f"{p}{key}", features.get(key, default))

        swing_bias = _get("smc_swing_bias")
        if swing_bias > 0.3:
            trend = MarketStructureBias.BULLISH
        elif swing_bias < -0.3:
            trend = MarketStructureBias.BEARISH
        elif abs(swing_bias) < 0.1:
            trend = MarketStructureBias.NEUTRAL
        else:
            trend = MarketStructureBias.UNCLEAR

        # Last break
        break_dir = _get("smc_last_break_direction")
        is_choch = _get("smc_last_break_is_choch")
        if break_dir > 0:
            last_break_direction = "bullish"
        elif break_dir < 0:
            last_break_direction = "bearish"
        else:
            last_break_direction = "none"

        last_break_type = "CHoCH" if is_choch > 0.5 else ("BOS" if break_dir != 0 else "none")

        # Price zone
        zone_val = _get("smc_price_zone")
        zone_map = {
            -2.0: "deep_discount",
            -1.0: "discount",
            0.0: "equilibrium",
            1.0: "premium",
            2.0: "deep_premium",
        }
        price_zone = zone_map.get(zone_val, "unknown")

        # Sweep type
        sweep_val = _get("smc_last_sweep_type")
        if sweep_val > 0:
            last_sweep = "BSL"
        elif sweep_val < 0:
            last_sweep = "SSL"
        else:
            last_sweep = "none"

        observations: list[str] = []
        if trend == MarketStructureBias.BULLISH:
            observations.append(f"Bullish bias (swing={swing_bias:.2f})")
        elif trend == MarketStructureBias.BEARISH:
            observations.append(f"Bearish bias (swing={swing_bias:.2f})")

        if price_zone != "unknown" and price_zone != "equilibrium":
            observations.append(f"Price in {price_zone.replace('_', ' ')}")

        if _get("smc_in_ote") > 0:
            observations.append("Price in OTE zone")

        if last_break_type == "CHoCH":
            observations.append(f"{last_break_direction.capitalize()} CHoCH detected")

        if last_sweep != "none":
            observations.append(f"Recent {last_sweep} sweep")

        ob_demand = int(_get("smc_ob_unmitigated_bullish"))
        ob_supply = int(_get("smc_ob_unmitigated_bearish"))
        if ob_demand > 0:
            observations.append(f"{ob_demand} unmitigated demand zone(s)")
        if ob_supply > 0:
            observations.append(f"{ob_supply} unmitigated supply zone(s)")

        return SMCTimeframeSummary(
            timeframe=tf,
            swing_bias=swing_bias,
            trend_label=trend,
            last_break_type=last_break_type,
            last_break_direction=last_break_direction,
            bos_bullish=int(_get("smc_bos_bullish")),
            bos_bearish=int(_get("smc_bos_bearish")),
            choch_bullish=int(_get("smc_choch_bullish")),
            choch_bearish=int(_get("smc_choch_bearish")),
            unmitigated_demand_zones=ob_demand,
            unmitigated_supply_zones=ob_supply,
            unfilled_bullish_fvgs=int(_get("smc_fvg_count_bullish")),
            unfilled_bearish_fvgs=int(_get("smc_fvg_count_bearish")),
            nearest_demand_distance=_get("smc_nearest_demand_distance"),
            nearest_supply_distance=_get("smc_nearest_supply_distance"),
            price_zone=price_zone,
            equilibrium=_get("smc_equilibrium"),
            in_ote=_get("smc_in_ote") > 0,
            bsl_sweeps=int(_get("smc_bsl_count")),
            ssl_sweeps=int(_get("smc_ssl_count")),
            last_sweep_type=last_sweep,
            confluence_score=_get("smc_confluence_score"),
            key_observations=observations,
        )

    def _score_confluence(
        self,
        summaries: list[SMCTimeframeSummary],
        features: dict[str, float],
    ) -> tuple[float, list[str], list[str]]:
        """Score cross-TF SMC confluence.

        Returns:
            Tuple of (total_points, factor_names, conflict_names).
        """
        score = 0.0
        factors: list[str] = []
        conflicts: list[str] = []

        if not summaries:
            return score, factors, conflicts

        htf = summaries[0]
        ltf = summaries[-1] if len(summaries) > 1 else summaries[0]

        # 1. HTF bias alignment: 0-3 points
        htf_pts = min(3.0, abs(htf.swing_bias) * 3.0)
        if htf_pts > 0:
            score += htf_pts
            factors.append(
                f"HTF {htf.timeframe.value} bias "
                f"{'bullish' if htf.swing_bias > 0 else 'bearish'} "
                f"({htf_pts:.1f}pts)"
            )

        # 2. Structure alignment: 0-2 points
        if len(summaries) >= 2:
            if (
                htf.trend_label == ltf.trend_label
                and htf.trend_label != MarketStructureBias.UNCLEAR
            ):
                score += 2.0
                factors.append(
                    f"Structure aligned: {htf.timeframe.value} and "
                    f"{ltf.timeframe.value} both {htf.trend_label.value} (2pts)"
                )
            elif (
                htf.trend_label != MarketStructureBias.UNCLEAR
                and ltf.trend_label != MarketStructureBias.UNCLEAR
                and htf.trend_label != ltf.trend_label
            ):
                conflicts.append(
                    f"{htf.timeframe.value} {htf.trend_label.value} "
                    f"vs {ltf.timeframe.value} {ltf.trend_label.value}"
                )

        # 3. OB confluence: 0-2 points
        ob_pts = 0.0
        if htf.nearest_demand_distance > 0 and htf.nearest_demand_distance < self._ob_proximity:
            ob_pts += 1.0
        if htf.nearest_supply_distance > 0 and htf.nearest_supply_distance < self._ob_proximity:
            ob_pts += 1.0
        if ob_pts > 0:
            score += ob_pts
            factors.append(f"HTF OB zone near price ({ob_pts:.0f}pts)")

        # 4. FVG confluence: 0-2 points
        tfs_with_bullish_fvg = sum(1 for s in summaries if s.unfilled_bullish_fvgs > 0)
        tfs_with_bearish_fvg = sum(1 for s in summaries if s.unfilled_bearish_fvgs > 0)
        fvg_pts = 0.0
        if tfs_with_bullish_fvg >= 2:
            fvg_pts += 1.0
        if tfs_with_bearish_fvg >= 2:
            fvg_pts += 1.0
        fvg_pts = min(2.0, fvg_pts)
        if fvg_pts > 0:
            score += fvg_pts
            factors.append(f"FVG confluence across {max(tfs_with_bullish_fvg, tfs_with_bearish_fvg)} TFs ({fvg_pts:.0f}pts)")

        # 5. Liquidity sweep: 0-3 points
        confirmed_sweeps = sum(
            s.bsl_sweeps + s.ssl_sweeps for s in summaries
        )
        if confirmed_sweeps > 0:
            sweep_pts = min(3.0, confirmed_sweeps * 1.0)
            score += sweep_pts
            factors.append(f"Liquidity swept ({confirmed_sweeps} sweeps, {sweep_pts:.1f}pts)")

        # 6. OTE alignment: 0-2 points
        ote_count = sum(1 for s in summaries if s.in_ote)
        if ote_count > 0:
            ote_pts = min(2.0, ote_count * 1.0)
            score += ote_pts
            factors.append(f"Price in OTE on {ote_count} TF(s) ({ote_pts:.0f}pts)")

        return min(14.0, score), factors, conflicts

    @staticmethod
    def _compute_alignment(summaries: list[SMCTimeframeSummary]) -> float:
        """Compute alignment score from bias across timeframes.

        Returns 1.0 if all TFs agree, 0.0 if fully mixed.
        """
        if not summaries:
            return 0.0

        biases = [s.swing_bias for s in summaries if abs(s.swing_bias) > 0.1]
        if not biases:
            return 0.5  # All neutral

        all_positive = all(b > 0 for b in biases)
        all_negative = all(b < 0 for b in biases)
        if all_positive or all_negative:
            return 1.0

        # Partial alignment: use weighted agreement
        abs_sum = sum(abs(b) for b in biases)
        if abs_sum == 0:
            return 0.5
        return abs(sum(biases)) / abs_sum

    @staticmethod
    def _classify_structure_alignment(
        summaries: list[SMCTimeframeSummary],
    ) -> str:
        """Classify the structural relationship between timeframes.

        Returns:
            ``"aligned"`` if all non-unclear TFs agree,
            ``"divergent"`` if HTF and LTF disagree,
            ``"transitioning"`` if any TF has a recent CHoCH.
        """
        if not summaries:
            return "unclear"

        # Check for CHoCH on any TF (transitioning)
        has_choch = any(
            s.choch_bullish > 0 or s.choch_bearish > 0 for s in summaries
        )

        biases = [
            s.trend_label for s in summaries
            if s.trend_label != MarketStructureBias.UNCLEAR
        ]

        if not biases:
            return "unclear"

        all_same = len(set(biases)) == 1
        if all_same and not has_choch:
            return "aligned"
        if has_choch:
            return "transitioning"
        return "divergent"

    @staticmethod
    def _infer_timeframes(features: dict[str, float]) -> list[Timeframe]:
        """Infer available timeframes from prefixed feature keys."""
        tf_values = {tf.value: tf for tf in Timeframe}
        found: set[Timeframe] = set()
        for key in features:
            parts = key.split("_", 1)
            if len(parts) > 1 and parts[0] in tf_values:
                found.add(tf_values[parts[0]])
        return sorted(found, key=lambda tf: tf.minutes, reverse=True)
