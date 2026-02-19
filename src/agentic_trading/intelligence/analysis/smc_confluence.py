"""Multi-timeframe SMC confluence scorer.

Scores how well Smart Money Concepts signals align across multiple
timeframes.  Consumes prefixed features from
:class:`~agentic_trading.features.multi_timeframe.MultiTimeframeAligner`
(e.g., ``"4h_smc_swing_bias"``) and produces a unified confluence
assessment.

In addition to pure price-action SMC signals, the scorer integrates
auxiliary data domains when available in the feature dict:

* **Funding rate** — crowding / sentiment assessment.
* **Open interest** — fresh-money confirmation.
* **Orderbook depth** — bid/ask wall support.
* **Volume delta** — buy/sell pressure confirmation.
* **BTC correlation** — cross-asset risk advisory.

Usage::

    scorer = SMCConfluenceScorer()
    result = scorer.score("BTC/USDT", aligned_features)
    print(result.total_confluence_points, result.htf_bias)
"""

from __future__ import annotations

import logging
import math
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


# -----------------------------------------------------------------------
# Auxiliary data assessments (funding, OI, orderbook, volume, correlation)
# -----------------------------------------------------------------------

@dataclass
class FundingAssessment:
    """Funding rate crowding / sentiment assessment."""

    rate: float = 0.0
    z_score: float = 0.0
    sentiment: str = "NEUTRAL"  # NEUTRAL / SLIGHTLY_BULLISH / SLIGHTLY_BEARISH / OVERHEATED_LONG / OVERHEATED_SHORT
    is_crowded: bool = False
    description: str = ""


@dataclass
class OIAssessment:
    """Open interest confirmation assessment."""

    current: float = 0.0
    change_pct_24h: float = 0.0
    trend: int = 0  # +1 increasing, -1 decreasing, 0 stable
    interpretation: str = "STABLE"  # FRESH_MONEY_ENTERING / UNWINDING / STABLE
    description: str = ""


@dataclass
class OrderbookAssessment:
    """Orderbook depth / bid-ask wall assessment."""

    imbalance: float = 1.0  # bid_depth / ask_depth (>1 = bid-heavy)
    spread_pct: float = 0.0
    bid_wall_price: float = 0.0
    bid_wall_size: float = 0.0
    bid_wall_persistence: float = 0.0
    ask_wall_price: float = 0.0
    ask_wall_size: float = 0.0
    ask_wall_persistence: float = 0.0
    description: str = ""


@dataclass
class VolumeDeltaAssessment:
    """Volume delta (buy/sell pressure) assessment."""

    delta: float = 0.0  # positive = buy-dominant, negative = sell-dominant
    cumulative: float = 0.0
    ratio: float = 1.0  # buy_vol / sell_vol
    trend: str = "FLAT"  # INCREASING / DECREASING / FLAT
    description: str = ""


@dataclass
class CorrelationAssessment:
    """Cross-asset BTC correlation assessment."""

    btc_correlation: float = 0.0
    independence_level: str = "UNKNOWN"  # HIGH / MODERATE / LOW / UNKNOWN
    description: str = ""


# -----------------------------------------------------------------------
# Main confluence result
# -----------------------------------------------------------------------

# Maximum confluence points (expanded from 14 to 20 for auxiliary data).
MAX_CONFLUENCE_POINTS: float = 20.0


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
    total_confluence_points: float = 0.0  # 0-20 scale
    max_confluence_points: float = MAX_CONFLUENCE_POINTS
    confluence_factors: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    # Key levels from SMC
    nearest_demand_price: float | None = None
    nearest_supply_price: float | None = None
    equilibrium_price: float | None = None
    dealing_range_high: float | None = None
    dealing_range_low: float | None = None

    # Auxiliary data assessments
    funding_assessment: FundingAssessment | None = None
    oi_assessment: OIAssessment | None = None
    orderbook_assessment: OrderbookAssessment | None = None
    volume_delta_assessment: VolumeDeltaAssessment | None = None
    correlation_assessment: CorrelationAssessment | None = None


class SMCConfluenceScorer:
    """Scores SMC signal alignment across multiple timeframes.

    Consumes prefixed features from ``MultiTimeframeAligner``
    (e.g., ``"4h_smc_swing_bias"``) and produces a unified SMC
    confluence assessment.

    Scoring system (0-20 total):

    **Price-action SMC (0-14):**

    - HTF bias alignment:       0-3 points (swing_bias on highest TF)
    - Structure alignment:      0-2 points (HTF + LTF BOS/CHoCH agree)
    - Order block confluence:   0-2 points (HTF OB zone near LTF price)
    - FVG confluence:           0-2 points (unfilled FVGs align across TFs)
    - Liquidity sweep:          0-3 points (confirmed sweep with reversal)
    - OTE alignment:            0-2 points (price in OTE zone with bias)

    **Auxiliary data (0-6):**

    - Funding rate:             0-2 points (neutral/favorable funding)
    - Open interest:            0-2 points (OI confirms bias direction)
    - Orderbook depth:          0-1 point  (orderbook supports bias)
    - Volume delta:             0-1 point  (volume confirms bias)
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

        # --- Auxiliary data assessments (non-SMC) ---
        funding = self._assess_funding(aligned_features, htf_bias)
        oi = self._assess_open_interest(aligned_features, htf_bias)
        orderbook = self._assess_orderbook(aligned_features, htf_bias)
        volume_delta = self._assess_volume_delta(aligned_features, htf_bias)
        correlation = self._assess_correlation(aligned_features)

        # Score auxiliary confluence
        aux_points, aux_factors, aux_conflicts = self._score_auxiliary(
            htf_bias, funding, oi, orderbook, volume_delta,
        )

        total_points += aux_points
        factors.extend(aux_factors)
        conflicts.extend(aux_conflicts)

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
            total_confluence_points=round(min(MAX_CONFLUENCE_POINTS, total_points), 2),
            confluence_factors=factors,
            conflicts=conflicts,
            funding_assessment=funding,
            oi_assessment=oi,
            orderbook_assessment=orderbook,
            volume_delta_assessment=volume_delta,
            correlation_assessment=correlation,
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

        # Cap at 14 for the SMC-only portion; auxiliary adds up to 6 more.
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

    # ------------------------------------------------------------------
    # Auxiliary data assessments
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_funding(
        features: dict[str, float],
        htf_bias: MarketStructureBias,
    ) -> FundingAssessment | None:
        """Assess funding rate for crowding / sentiment.

        Looks for ``funding_rate`` and ``funding_zscore`` keys in *features*.
        Returns ``None`` when funding data is unavailable.
        """
        rate = features.get("funding_rate", float("nan"))
        z_score = features.get("funding_zscore", float("nan"))

        if math.isnan(rate):
            return None

        z = z_score if not math.isnan(z_score) else 0.0

        # Classify sentiment
        if abs(z) > 2.0:
            sentiment = "OVERHEATED_LONG" if z > 0 else "OVERHEATED_SHORT"
            is_crowded = True
        elif abs(z) > 1.0:
            sentiment = "SLIGHTLY_BULLISH" if z > 0 else "SLIGHTLY_BEARISH"
            is_crowded = False
        else:
            sentiment = "NEUTRAL"
            is_crowded = False

        rate_pct = rate * 100  # as percentage
        desc = f"{rate_pct:+.4f}% ({sentiment.replace('_', ' ').title()})"
        if is_crowded:
            desc += " - Overcrowded"
        elif abs(z) < 1.0:
            desc += " - Not overcrowded on either side"

        return FundingAssessment(
            rate=rate,
            z_score=z,
            sentiment=sentiment,
            is_crowded=is_crowded,
            description=desc,
        )

    @staticmethod
    def _assess_open_interest(
        features: dict[str, float],
        htf_bias: MarketStructureBias,
    ) -> OIAssessment | None:
        """Assess open interest for fresh-money confirmation.

        Looks for ``oi_current``, ``oi_change_pct_24h``, ``oi_trend`` keys.
        Returns ``None`` when OI data is unavailable.
        """
        oi_current = features.get("oi_current", float("nan"))
        if math.isnan(oi_current):
            return None

        change_pct = features.get("oi_change_pct_24h", 0.0)
        trend = int(features.get("oi_trend", 0))

        # Interpret
        if change_pct > 1.0:
            interpretation = "FRESH_MONEY_ENTERING"
        elif change_pct < -1.0:
            interpretation = "UNWINDING"
        else:
            interpretation = "STABLE"

        # Description
        if change_pct > 0:
            desc = f"+{change_pct:.1f}% increase in 24h - {interpretation.replace('_', ' ').title()}"
        elif change_pct < 0:
            desc = f"{change_pct:.1f}% decrease in 24h - {interpretation.replace('_', ' ').title()}"
        else:
            desc = "Stable"

        return OIAssessment(
            current=oi_current,
            change_pct_24h=change_pct,
            trend=trend,
            interpretation=interpretation,
            description=desc,
        )

    @staticmethod
    def _assess_orderbook(
        features: dict[str, float],
        htf_bias: MarketStructureBias,
    ) -> OrderbookAssessment | None:
        """Assess orderbook depth and bid/ask walls.

        Looks for ``ob_imbalance``, ``ob_spread_pct``, ``ob_bid_wall_*``,
        ``ob_ask_wall_*`` keys.  Returns ``None`` when no orderbook data
        is present.
        """
        imbalance = features.get("ob_imbalance", float("nan"))
        if math.isnan(imbalance):
            return None

        spread_pct = features.get("ob_spread_pct", 0.0)
        bid_wall_price = features.get("ob_bid_wall_price", 0.0)
        bid_wall_size = features.get("ob_bid_wall_size", 0.0)
        bid_wall_persist = features.get("ob_bid_wall_persistence", 0.0)
        ask_wall_price = features.get("ob_ask_wall_price", 0.0)
        ask_wall_size = features.get("ob_ask_wall_size", 0.0)
        ask_wall_persist = features.get("ob_ask_wall_persistence", 0.0)

        parts: list[str] = []
        if imbalance > 1.5:
            parts.append(f"Bid-heavy imbalance ({imbalance:.1f}x)")
        elif imbalance < 0.67:
            parts.append(f"Ask-heavy imbalance ({imbalance:.1f}x)")
        else:
            parts.append(f"Balanced orderbook ({imbalance:.1f}x)")

        if bid_wall_size > 0 and bid_wall_price > 0:
            persist_str = f", {bid_wall_persist:.0%} persistence" if bid_wall_persist > 0 else ""
            parts.append(
                f"Bid wall at ${bid_wall_price:,.2f} "
                f"({bid_wall_size:,.0f}{persist_str})"
            )
        if ask_wall_size > 0 and ask_wall_price > 0:
            persist_str = f", {ask_wall_persist:.0%} persistence" if ask_wall_persist > 0 else ""
            parts.append(
                f"Ask wall at ${ask_wall_price:,.2f} "
                f"({ask_wall_size:,.0f}{persist_str})"
            )

        return OrderbookAssessment(
            imbalance=imbalance,
            spread_pct=spread_pct,
            bid_wall_price=bid_wall_price,
            bid_wall_size=bid_wall_size,
            bid_wall_persistence=bid_wall_persist,
            ask_wall_price=ask_wall_price,
            ask_wall_size=ask_wall_size,
            ask_wall_persistence=ask_wall_persist,
            description=" | ".join(parts),
        )

    @staticmethod
    def _assess_volume_delta(
        features: dict[str, float],
        htf_bias: MarketStructureBias,
    ) -> VolumeDeltaAssessment | None:
        """Assess volume delta for buy/sell pressure confirmation.

        Looks for ``volume_delta``, ``volume_delta_cumulative``,
        ``volume_delta_ratio``, ``volume_delta_trend`` keys.
        Returns ``None`` when volume delta data is unavailable.
        """
        delta = features.get("volume_delta", float("nan"))
        if math.isnan(delta):
            return None

        cumulative = features.get("volume_delta_cumulative", 0.0)
        ratio = features.get("volume_delta_ratio", 1.0)
        trend_val = features.get("volume_delta_trend", 0.0)

        if trend_val > 0.5:
            trend = "INCREASING"
        elif trend_val < -0.5:
            trend = "DECREASING"
        else:
            trend = "FLAT"

        # Description
        if delta > 0:
            desc = "Bullish volume pressure"
        elif delta < 0:
            desc = "Bearish volume pressure"
        else:
            desc = "Mixed volume delta"

        if trend != "FLAT":
            desc += f", {trend.lower()} trend"

        return VolumeDeltaAssessment(
            delta=delta,
            cumulative=cumulative,
            ratio=ratio if not math.isnan(ratio) else 1.0,
            trend=trend,
            description=desc,
        )

    @staticmethod
    def _assess_correlation(
        features: dict[str, float],
    ) -> CorrelationAssessment | None:
        """Assess BTC correlation for cross-asset risk advisory.

        Looks for ``btc_correlation`` key.  Returns ``None`` when not
        available (e.g. for BTC/USDT itself).
        """
        corr = features.get("btc_correlation", float("nan"))
        if math.isnan(corr):
            return None

        abs_corr = abs(corr)
        if abs_corr > 0.7:
            level = "HIGH"
            desc = "Highly correlated with BTC - track BTC closely"
        elif abs_corr > 0.3:
            level = "MODERATE"
            desc = "Moderate BTC correlation - can move semi-independently"
        else:
            level = "LOW"
            desc = "Low BTC correlation - moves independently"

        return CorrelationAssessment(
            btc_correlation=corr,
            independence_level=level,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Auxiliary confluence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_auxiliary(
        htf_bias: MarketStructureBias,
        funding: FundingAssessment | None,
        oi: OIAssessment | None,
        orderbook: OrderbookAssessment | None,
        volume_delta: VolumeDeltaAssessment | None,
    ) -> tuple[float, list[str], list[str]]:
        """Score auxiliary (non-SMC) confluence factors.

        Returns:
            Tuple of (points, factor_names, conflict_names).
            Maximum 6 points (funding 2, OI 2, orderbook 1, vol delta 1).
        """
        score = 0.0
        factors: list[str] = []
        conflicts: list[str] = []

        is_bullish = htf_bias == MarketStructureBias.BULLISH
        is_bearish = htf_bias == MarketStructureBias.BEARISH
        has_direction = is_bullish or is_bearish

        # --- Funding rate: 0-2 points ---
        if funding is not None and has_direction:
            if funding.is_crowded:
                # Crowded in same direction as bias = negative
                crowded_bullish = funding.sentiment == "OVERHEATED_LONG"
                if (is_bullish and crowded_bullish) or (is_bearish and not crowded_bullish):
                    score -= 1.0
                    conflicts.append(
                        f"Funding overcrowded in trade direction ({funding.sentiment})"
                    )
                else:
                    # Crowded against bias direction = favorable
                    score += 2.0
                    factors.append(
                        f"Funding overcrowded against position ({funding.sentiment}, 2pts)"
                    )
            elif funding.sentiment == "NEUTRAL":
                score += 1.0
                factors.append("Funding neutral - not overcrowded (1pt)")
            else:
                # Slightly directional
                slightly_bullish = funding.sentiment == "SLIGHTLY_BULLISH"
                if (is_bullish and not slightly_bullish) or (is_bearish and slightly_bullish):
                    # Funding leans favorably
                    score += 1.5
                    factors.append(
                        f"Funding leans favorable ({funding.sentiment}, 1.5pts)"
                    )
                else:
                    score += 0.5
                    factors.append(
                        f"Funding slightly against ({funding.sentiment}, 0.5pts)"
                    )

        # --- Open interest: 0-2 points ---
        if oi is not None and has_direction:
            if oi.interpretation == "FRESH_MONEY_ENTERING":
                # OI increasing - confirms directional conviction
                score += 2.0
                factors.append(
                    f"OI increasing ({oi.change_pct_24h:+.1f}%) - fresh money entering (2pts)"
                )
            elif oi.interpretation == "UNWINDING":
                conflicts.append(
                    f"OI decreasing ({oi.change_pct_24h:+.1f}%) - positions unwinding"
                )
            else:
                score += 0.5
                factors.append("OI stable (0.5pts)")

        # --- Orderbook depth: 0-1 point ---
        if orderbook is not None and has_direction:
            if is_bullish and orderbook.imbalance > 1.3:
                score += 1.0
                factors.append(
                    f"Orderbook bid-heavy ({orderbook.imbalance:.1f}x, 1pt)"
                )
            elif is_bearish and orderbook.imbalance < 0.77:
                score += 1.0
                factors.append(
                    f"Orderbook ask-heavy ({orderbook.imbalance:.1f}x, 1pt)"
                )
            elif (is_bullish and orderbook.imbalance < 0.77) or (
                is_bearish and orderbook.imbalance > 1.3
            ):
                conflicts.append(
                    f"Orderbook against bias ({orderbook.imbalance:.1f}x imbalance)"
                )

        # --- Volume delta: 0-1 point ---
        if volume_delta is not None and has_direction:
            if (is_bullish and volume_delta.delta > 0) or (
                is_bearish and volume_delta.delta < 0
            ):
                score += 1.0
                factors.append(f"Volume delta confirms bias ({volume_delta.trend}, 1pt)")
            elif (is_bullish and volume_delta.delta < 0) or (
                is_bearish and volume_delta.delta > 0
            ):
                conflicts.append(
                    f"Volume delta diverges from bias ({volume_delta.trend})"
                )

        return max(0.0, score), factors, conflicts

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
