"""Higher-timeframe market structure analyzer.

Implements the multi-timeframe analysis protocol where higher-timeframe
structure always takes precedence for directional bias, while lower
timeframes provide timing precision.

Consumes the same ``dict[str, float]`` that
:meth:`~agentic_trading.features.multi_timeframe.MultiTimeframeAligner.get_aligned_features`
produces (prefixed feature keys like ``"4h_ema_21"``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentic_trading.core.enums import (
    MarketStructureBias,
    RegimeType,
    Timeframe,
    VolatilityRegime,
)

logger = logging.getLogger(__name__)

# Timeframe hierarchy for HTF analysis (highest to lowest)
_HTF_HIERARCHY = [
    Timeframe.D1,
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
]


@dataclass
class TimeframeSummary:
    """Structure assessment for a single timeframe."""

    timeframe: Timeframe
    bias: MarketStructureBias = MarketStructureBias.UNCLEAR
    trend_strength: float = 0.0  # ADX or equivalent, 0–100
    momentum: float = 0.0  # RSI deviation from 50
    volatility_state: str = "normal"  # expanding / contracting / normal
    key_observations: list[str] = field(default_factory=list)


@dataclass
class HTFAssessment:
    """Complete multi-timeframe structure assessment."""

    symbol: str
    overall_bias: MarketStructureBias = MarketStructureBias.UNCLEAR
    bias_alignment_score: float = 0.0  # 0–1, how aligned the timeframes are
    timeframe_summaries: list[TimeframeSummary] = field(default_factory=list)
    regime_suggestion: RegimeType = RegimeType.UNKNOWN
    volatility_assessment: VolatilityRegime = VolatilityRegime.UNKNOWN
    trade_recommendation: str = ""
    confluences: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


class HTFAnalyzer:
    """Analyzes multi-timeframe features to assess market structure.

    Higher timeframes receive greater weight for directional bias.
    Lower timeframes provide timing and entry precision.

    Usage::

        analyzer = HTFAnalyzer()
        assessment = analyzer.analyze("BTC/USDT", aligned_features)
    """

    def __init__(
        self,
        trend_ema_fast: int = 21,
        trend_ema_slow: int = 50,
        structure_ema: int = 200,
        adx_trend_threshold: float = 25.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> None:
        self._ema_fast = trend_ema_fast
        self._ema_slow = trend_ema_slow
        self._structure_ema = structure_ema
        self._adx_threshold = adx_trend_threshold
        self._rsi_ob = rsi_overbought
        self._rsi_os = rsi_oversold

    def analyze(
        self,
        symbol: str,
        aligned_features: dict[str, float],
        available_timeframes: list[Timeframe] | None = None,
    ) -> HTFAssessment:
        """Produce an HTF assessment from aligned feature vectors.

        Args:
            symbol: Trading pair.
            aligned_features: Prefixed feature dict from
                ``MultiTimeframeAligner`` (e.g. ``"4h_ema_21"``).
            available_timeframes: Which timeframes have data.  If *None*,
                inferred from feature key prefixes.

        Returns:
            :class:`HTFAssessment` with per-timeframe summaries and
            overall weighted bias.
        """
        if available_timeframes is None:
            available_timeframes = self._infer_timeframes(aligned_features)

        summaries: list[TimeframeSummary] = []
        bias_scores: list[float] = []  # +1 bullish, -1 bearish

        for rank, tf in enumerate(_HTF_HIERARCHY):
            if tf not in available_timeframes:
                continue

            prefix = tf.value
            summary = self._analyze_timeframe(prefix, tf, aligned_features)
            summaries.append(summary)

            # HTF gets more weight (lower rank = higher in hierarchy)
            weight = float(len(_HTF_HIERARCHY) - rank)
            if summary.bias == MarketStructureBias.BULLISH:
                bias_scores.append(weight)
            elif summary.bias == MarketStructureBias.BEARISH:
                bias_scores.append(-weight)
            else:
                bias_scores.append(0.0)

        # Overall bias from weighted scores
        total_weight = sum(
            float(len(_HTF_HIERARCHY) - i)
            for i, tf in enumerate(_HTF_HIERARCHY)
            if tf in available_timeframes
        )

        net_score = sum(bias_scores) / total_weight if total_weight > 0 else 0

        if net_score > 0.3:
            overall = MarketStructureBias.BULLISH
        elif net_score < -0.3:
            overall = MarketStructureBias.BEARISH
        elif abs(net_score) < 0.1:
            overall = MarketStructureBias.NEUTRAL
        else:
            overall = MarketStructureBias.UNCLEAR

        # Alignment score: 1.0 if all TFs agree, lower if mixed
        alignment = self._compute_alignment(bias_scores)

        # Confluences & conflicts
        confluences, conflicts = self._find_confluence_conflicts(summaries)

        # Regime suggestion from highest available TF
        regime = RegimeType.UNKNOWN
        if summaries:
            htf_adx = summaries[0].trend_strength
            if htf_adx > self._adx_threshold:
                regime = RegimeType.TREND
            elif htf_adx < 20:
                regime = RegimeType.RANGE

        # Volatility from highest available TF
        vol = VolatilityRegime.UNKNOWN
        if summaries:
            vs = summaries[0].volatility_state
            if vs == "expanding":
                vol = VolatilityRegime.HIGH
            elif vs == "contracting":
                vol = VolatilityRegime.LOW

        return HTFAssessment(
            symbol=symbol,
            overall_bias=overall,
            bias_alignment_score=round(alignment, 2),
            timeframe_summaries=summaries,
            regime_suggestion=regime,
            volatility_assessment=vol,
            confluences=confluences,
            conflicts=conflicts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_timeframe(
        self,
        prefix: str,
        tf: Timeframe,
        features: dict[str, float],
    ) -> TimeframeSummary:
        """Analyze a single timeframe from prefixed features."""
        close = features.get(f"{prefix}_close", 0)
        ema_fast = features.get(f"{prefix}_ema_{self._ema_fast}", 0)
        ema_slow = features.get(f"{prefix}_ema_{self._ema_slow}", 0)
        ema_200 = features.get(f"{prefix}_ema_{self._structure_ema}")
        sma_200 = features.get(f"{prefix}_sma_200")
        adx = features.get(f"{prefix}_adx_14", 0)
        rsi = features.get(f"{prefix}_rsi_14", 50)
        atr_pct = features.get(f"{prefix}_atr_14_pct", 0)

        observations: list[str] = []
        bias = MarketStructureBias.UNCLEAR

        # EMA cross bias
        if ema_fast > 0 and ema_slow > 0:
            if ema_fast > ema_slow:
                bias = MarketStructureBias.BULLISH
                observations.append(
                    f"EMA{self._ema_fast} > EMA{self._ema_slow}"
                )
            else:
                bias = MarketStructureBias.BEARISH
                observations.append(
                    f"EMA{self._ema_fast} < EMA{self._ema_slow}"
                )

        # Price vs structure EMA (200)
        structure_val = ema_200 or sma_200
        if structure_val and close > 0:
            if close > structure_val:
                observations.append("Price above 200 MA (bullish structure)")
            else:
                observations.append("Price below 200 MA (bearish structure)")

        # RSI observations
        if rsi > self._rsi_ob:
            observations.append(f"RSI {rsi:.0f} overbought")
        elif rsi < self._rsi_os:
            observations.append(f"RSI {rsi:.0f} oversold")

        # Trend strength
        if adx > self._adx_threshold:
            observations.append(f"ADX {adx:.0f} strong trend")

        # Volatility state
        vol_state = "normal"
        if atr_pct > 5.0:
            vol_state = "expanding"
        elif atr_pct < 1.0:
            vol_state = "contracting"

        return TimeframeSummary(
            timeframe=tf,
            bias=bias,
            trend_strength=adx,
            momentum=rsi - 50.0,
            volatility_state=vol_state,
            key_observations=observations,
        )

    @staticmethod
    def _infer_timeframes(features: dict[str, float]) -> list[Timeframe]:
        """Infer available timeframes from prefixed feature keys."""
        tf_values = {tf.value: tf for tf in Timeframe}
        found: set[Timeframe] = set()
        for key in features:
            prefix = key.split("_", 1)[0]
            if prefix in tf_values:
                found.add(tf_values[prefix])
        return sorted(found, key=lambda tf: tf.minutes)

    @staticmethod
    def _compute_alignment(bias_scores: list[float]) -> float:
        """Compute alignment score from bias scores."""
        if not bias_scores:
            return 0.0

        non_zero = [s for s in bias_scores if s != 0]
        if not non_zero:
            return 0.5  # All neutral

        all_positive = all(s >= 0 for s in bias_scores)
        all_negative = all(s <= 0 for s in bias_scores)
        if all_positive or all_negative:
            return 1.0

        abs_sum = sum(abs(s) for s in non_zero)
        if abs_sum == 0:
            return 0.5
        return abs(sum(non_zero)) / abs_sum

    @staticmethod
    def _find_confluence_conflicts(
        summaries: list[TimeframeSummary],
    ) -> tuple[list[str], list[str]]:
        """Identify where timeframes agree or disagree."""
        confluences: list[str] = []
        conflicts: list[str] = []

        biases = [
            (s.timeframe.value, s.bias)
            for s in summaries
            if s.bias != MarketStructureBias.UNCLEAR
        ]

        for i in range(len(biases)):
            for j in range(i + 1, len(biases)):
                tf_a, bias_a = biases[i]
                tf_b, bias_b = biases[j]
                if bias_a == bias_b:
                    confluences.append(
                        f"{tf_a} and {tf_b} aligned {bias_a.value}"
                    )
                else:
                    conflicts.append(
                        f"{tf_a} {bias_a.value} vs {tf_b} {bias_b.value}"
                    )

        return confluences, conflicts
