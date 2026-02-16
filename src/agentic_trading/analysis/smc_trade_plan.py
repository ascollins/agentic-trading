"""SMC-driven trade plan generator and narrative formatter.

Combines all SMC analysis — swing structure, order blocks, FVGs,
liquidity sweeps, premium/discount zones, and multi-TF confluence —
into a structured :class:`~agentic_trading.analysis.trade_plan.TradePlan`
and a human-readable narrative report matching institutional SMC analysis
output (multi-timeframe view, trade verdict, entry/SL/TP, invalidation).

Usage::

    generator = SMCTradePlanGenerator()
    report = generator.generate_report(
        symbol="BTC/USDT",
        current_price=612.1,
        aligned_features=features,
    )
    plan = generator.generate_trade_plan(report)
    text = generator.format_report(report, plan)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentic_trading.analysis.rr_calculator import calculate_rr
from agentic_trading.analysis.smc_confluence import (
    SMCConfluenceResult,
    SMCConfluenceScorer,
    SMCTimeframeSummary,
)
from agentic_trading.analysis.trade_plan import EntryZone, TargetSpec, TradePlan
from agentic_trading.core.enums import (
    ConvictionLevel,
    MarketStructureBias,
    SetupGrade,
    SignalDirection,
    Timeframe,
)

logger = logging.getLogger(__name__)


@dataclass
class InvalidationCondition:
    """A condition that would invalidate the trade setup."""

    description: str
    trigger_price: float | None = None
    trigger_condition: str = ""


@dataclass
class SMCAnalysisReport:
    """Complete SMC analysis report for a symbol.

    Contains all information needed to render the multi-timeframe
    SMC analysis as text, drive a :class:`TradePlan`, or feed
    narration services.
    """

    symbol: str
    current_price: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Per-timeframe analysis
    htf_analysis: SMCTimeframeSummary | None = None
    ltf_analysis: SMCTimeframeSummary | None = None
    htf_timeframe: Timeframe = Timeframe.H4
    ltf_timeframe: Timeframe = Timeframe.H1

    # Overall assessment
    overall_bias: MarketStructureBias = MarketStructureBias.UNCLEAR
    confluence_result: SMCConfluenceResult | None = None

    # Structure narratives (human-readable)
    htf_structure_narrative: str = ""
    ltf_structure_narrative: str = ""

    # Key levels
    key_support_level: float | None = None
    key_support_description: str = ""
    key_resistance_level: float | None = None
    key_resistance_description: str = ""
    equilibrium_price: float | None = None
    dealing_range_high: float | None = None
    dealing_range_low: float | None = None

    # Verdict
    trade_verdict: str = ""
    setup_direction: SignalDirection = SignalDirection.FLAT
    conviction: ConvictionLevel = ConvictionLevel.LOW

    # Invalidation
    invalidation_conditions: list[InvalidationCondition] = field(
        default_factory=list,
    )


class SMCTradePlanGenerator:
    """Generates structured trade plans from SMC analysis.

    Combines:

    - :class:`SMCConfluenceScorer` output (multi-TF alignment)
    - R:R calculator (:func:`~agentic_trading.analysis.rr_calculator.calculate_rr`)
    - Key levels from SMC feature dict

    Produces:

    - :class:`SMCAnalysisReport` (data model)
    - :class:`TradePlan` (existing model, filled from SMC data)
    - Formatted text output (Bloomberg-presenter-style narrative)
    """

    def __init__(
        self,
        smc_confluence_scorer: SMCConfluenceScorer | None = None,
        default_risk_pct: float = 0.01,
        min_rr_ratio: float = 2.0,
        min_confluence_score: float = 7.0,
    ) -> None:
        self._scorer = smc_confluence_scorer or SMCConfluenceScorer()
        self._default_risk_pct = default_risk_pct
        self._min_rr = min_rr_ratio
        self._min_confluence = min_confluence_score

    def generate_report(
        self,
        symbol: str,
        current_price: float,
        aligned_features: dict[str, float],
        available_timeframes: list[Timeframe] | None = None,
    ) -> SMCAnalysisReport:
        """Generate a complete SMC analysis report.

        Args:
            symbol: Trading pair (e.g., ``"BTC/USDT"``).
            current_price: Current market price.
            aligned_features: Prefixed feature dict from
                ``MultiTimeframeAligner`` or single-TF features.
            available_timeframes: Available timeframes.

        Returns:
            :class:`SMCAnalysisReport` with all analysis data.
        """
        confluence = self._scorer.score(
            symbol, aligned_features, available_timeframes,
        )

        report = SMCAnalysisReport(
            symbol=symbol,
            current_price=current_price,
            overall_bias=confluence.htf_bias,
            confluence_result=confluence,
        )

        # Assign per-TF summaries
        if confluence.timeframe_summaries:
            report.htf_analysis = confluence.timeframe_summaries[0]
            report.htf_timeframe = report.htf_analysis.timeframe

            if len(confluence.timeframe_summaries) > 1:
                report.ltf_analysis = confluence.timeframe_summaries[-1]
                report.ltf_timeframe = report.ltf_analysis.timeframe

        # Build structure narratives
        if report.htf_analysis:
            report.htf_structure_narrative = self._build_structure_narrative(
                report.htf_analysis,
            )
        if report.ltf_analysis:
            report.ltf_structure_narrative = self._build_structure_narrative(
                report.ltf_analysis,
            )

        # Extract key levels
        if confluence.equilibrium_price is not None:
            report.equilibrium_price = confluence.equilibrium_price
        if confluence.dealing_range_high is not None:
            report.dealing_range_high = confluence.dealing_range_high
        if confluence.dealing_range_low is not None:
            report.dealing_range_low = confluence.dealing_range_low

        # Key support/resistance from OB distances and FVGs
        self._extract_key_levels(report, aligned_features, confluence)

        # Determine trade verdict
        verdict, direction = self._determine_trade_verdict(report, confluence)
        report.trade_verdict = verdict
        report.setup_direction = direction

        # Conviction from confluence score
        report.conviction = self._score_to_conviction(
            confluence.total_confluence_points,
        )

        # Invalidation conditions
        report.invalidation_conditions = self._build_invalidation_conditions(
            direction, report, aligned_features,
        )

        return report

    def generate_trade_plan(
        self,
        report: SMCAnalysisReport,
        risk_pct: float | None = None,
    ) -> TradePlan | None:
        """Convert an :class:`SMCAnalysisReport` into a :class:`TradePlan`.

        Returns *None* if the confluence score is below the minimum
        threshold or if a valid entry/SL/target set cannot be constructed.

        Args:
            report: Previously generated SMC analysis report.
            risk_pct: Risk per trade as a decimal (default: 1%).

        Returns:
            A populated :class:`TradePlan` or *None*.
        """
        if report.setup_direction == SignalDirection.FLAT:
            return None

        confluence = report.confluence_result
        if confluence is None:
            return None

        if confluence.total_confluence_points < self._min_confluence:
            logger.debug(
                "Confluence score %.1f below threshold %.1f for %s",
                confluence.total_confluence_points,
                self._min_confluence,
                report.symbol,
            )
            return None

        risk = risk_pct or self._default_risk_pct
        direction = report.setup_direction

        # Determine entry zone
        entry = self._determine_entry_zone(direction, report)
        if entry is None:
            return None

        # Determine stop loss
        stop_loss = self._determine_stop_loss(direction, report)
        if stop_loss is None:
            return None

        # Determine targets
        targets = self._determine_targets(
            direction, entry.primary_entry, stop_loss, report,
        )

        if not targets:
            return None

        # Compute R:R
        rr = calculate_rr(
            entry=entry.primary_entry,
            stop_loss=stop_loss,
            targets=[t.price for t in targets],
            direction=direction,
        )

        if rr.blended_rr < self._min_rr:
            logger.debug(
                "Blended R:R %.2f below minimum %.2f for %s",
                rr.blended_rr, self._min_rr, report.symbol,
            )
            return None

        # Build key levels dict
        key_levels: dict[str, float] = {}
        if report.equilibrium_price is not None:
            key_levels["equilibrium"] = report.equilibrium_price
        if report.dealing_range_high is not None:
            key_levels["dealing_range_high"] = report.dealing_range_high
        if report.dealing_range_low is not None:
            key_levels["dealing_range_low"] = report.dealing_range_low
        if report.key_support_level is not None:
            key_levels["key_support"] = report.key_support_level
        if report.key_resistance_level is not None:
            key_levels["key_resistance"] = report.key_resistance_level

        # Build invalidation notes
        invalidation_notes = "; ".join(
            ic.description for ic in report.invalidation_conditions
        )

        # Map setup grade from R:R result
        setup_grade = rr.setup_grade

        # HTF bias
        htf_bias = report.overall_bias

        plan = TradePlan(
            strategy_id="smc_analysis",
            symbol=report.symbol,
            timestamp=report.timestamp,
            direction=direction,
            conviction=report.conviction,
            setup_grade=setup_grade,
            confidence=min(1.0, confluence.total_confluence_points / 14.0),
            htf_bias=htf_bias,
            htf_timeframe=report.htf_timeframe,
            trade_timeframe=report.ltf_timeframe,
            ltf_trigger_timeframe=report.ltf_timeframe,
            structure_notes=report.htf_structure_narrative,
            entry=entry,
            stop_loss=stop_loss,
            risk_pct=risk,
            invalidation_notes=invalidation_notes,
            targets=[
                TargetSpec(
                    price=t.price,
                    rr_ratio=t.rr_ratio,
                    scale_out_pct=t.scale_out_pct,
                    rationale=t.rationale,
                )
                for t in targets
            ],
            blended_rr=rr.blended_rr,
            expected_r=rr.expected_r,
            key_levels=key_levels,
            rationale=report.trade_verdict,
            edge_description=(
                f"SMC confluence {confluence.total_confluence_points:.1f}/14 | "
                f"{confluence.structure_alignment} structure"
            ),
        )

        return plan

    def format_report(
        self,
        report: SMCAnalysisReport,
        plan: TradePlan | None = None,
    ) -> str:
        """Format the analysis as a multi-timeframe SMC narrative.

        Produces output matching the institutional analysis format:
        HTF analysis, LTF confirmation, trade verdict, entry/SL/TP,
        and invalidation conditions.
        """
        lines: list[str] = []

        # Header
        lines.append(
            f"{report.symbol} Technical Analysis - Multi-Timeframe SMC View"
        )
        lines.append(f"Current Price: ${report.current_price:,.2f}")
        lines.append("")

        # HTF Analysis
        if report.htf_analysis:
            htf = report.htf_analysis
            bias_emoji = self._bias_emoji(htf.trend_label)
            lines.append(
                f"{report.htf_timeframe.value.upper()} Analysis (HTF Bias)"
            )
            lines.append(f"Trend: {htf.trend_label.value.upper()} {bias_emoji}")

            # Price location
            if htf.price_zone != "unknown":
                eq_str = f" vs equilibrium ${htf.equilibrium:,.2f}" if htf.equilibrium > 0 else ""
                zone_label = htf.price_zone.replace("_", " ").upper()
                lines.append(f"  Price Location: {zone_label}{eq_str}")

            # Structure narrative
            if report.htf_structure_narrative:
                lines.append(f"  Recent structure: {report.htf_structure_narrative}")

            # Key support
            if report.key_support_level is not None:
                lines.append(
                    f"  Key support: ${report.key_support_level:,.2f}"
                    f" ({report.key_support_description})"
                )

            # Sweep observations
            if htf.last_sweep_type != "none":
                lines.append(
                    f"  Recent: {htf.last_sweep_type} sweep detected"
                )

            # Additional observations
            for obs in htf.key_observations:
                if "bias" not in obs.lower() and "price in" not in obs.lower():
                    lines.append(f"  {obs}")

            lines.append("")

        # LTF Analysis
        if report.ltf_analysis and report.ltf_analysis != report.htf_analysis:
            ltf = report.ltf_analysis
            bias_emoji = self._bias_emoji(ltf.trend_label)
            lines.append(
                f"{report.ltf_timeframe.value.upper()} Analysis (LTF Confirmation)"
            )
            lines.append(f"Trend: {ltf.trend_label.value.upper()} {bias_emoji}")

            if ltf.price_zone != "unknown":
                eq_str = f" vs equilibrium ${ltf.equilibrium:,.2f}" if ltf.equilibrium > 0 else ""
                zone_label = ltf.price_zone.replace("_", " ").upper()
                htf_align = ""
                if (
                    report.htf_analysis
                    and ltf.trend_label == report.htf_analysis.trend_label
                    and ltf.trend_label != MarketStructureBias.UNCLEAR
                ):
                    htf_align = f" - aligned with {report.htf_timeframe.value.upper()}"
                lines.append(f"  Price Location: {zone_label}{eq_str}{htf_align}")

            if report.ltf_structure_narrative:
                lines.append(f"  Recent structure: {report.ltf_structure_narrative}")

            lines.append("")

        # Trade Verdict
        lines.append(f"Trade Verdict: {report.trade_verdict}")
        lines.append("")

        # Trade Plan
        if plan is not None:
            direction_label = "BULLISH" if plan.direction == SignalDirection.LONG else "BEARISH"
            lines.append(f"{direction_label} Setup:")

            # Entry
            if plan.entry.entry_low is not None and plan.entry.entry_high is not None:
                lines.append(
                    f"  Entry Zone: ${plan.entry.entry_low:,.2f} - "
                    f"${plan.entry.entry_high:,.2f}"
                )
            else:
                lines.append(f"  Entry: ${plan.entry.primary_entry:,.2f}")

            # Stop loss
            inv_note = f" ({plan.invalidation_notes})" if plan.invalidation_notes else ""
            lines.append(f"  SL: ${plan.stop_loss:,.2f}{inv_note}")

            # Targets
            for i, target in enumerate(plan.targets, 1):
                rationale = f" ({target.rationale})" if target.rationale else ""
                lines.append(
                    f"  TP{i}: ${target.price:,.2f}{rationale}"
                )

            # R:R
            lines.append(f"  R:R: {plan.blended_rr:.1f}")
            lines.append(f"  Setup Grade: {plan.setup_grade.value}")
            lines.append(f"  Conviction: {plan.conviction.value.upper()}")
            lines.append("")

        # Invalidation
        if report.invalidation_conditions:
            lines.append("Invalidation:")
            for ic in report.invalidation_conditions:
                price_str = f" (${ic.trigger_price:,.2f})" if ic.trigger_price else ""
                lines.append(f"  - {ic.description}{price_str}")
            lines.append("")

        # Confluence factors
        if report.confluence_result and report.confluence_result.confluence_factors:
            lines.append(
                f"Confluence Score: "
                f"{report.confluence_result.total_confluence_points:.1f}/14"
            )
            for factor in report.confluence_result.confluence_factors:
                lines.append(f"  + {factor}")

            if report.confluence_result.conflicts:
                for conflict in report.confluence_result.conflicts:
                    lines.append(f"  ! {conflict}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_entry_zone(
        self,
        direction: SignalDirection,
        report: SMCAnalysisReport,
    ) -> EntryZone | None:
        """Determine entry zone from SMC levels."""
        price = report.current_price

        if direction == SignalDirection.LONG:
            # Entry near demand zone / current price in discount
            primary = price
            if report.key_support_level is not None:
                # Entry between current price and support
                entry_low = report.key_support_level
                entry_high = price
                primary = (entry_low + entry_high) / 2
            else:
                entry_low = price * 0.995
                entry_high = price * 1.005
        elif direction == SignalDirection.SHORT:
            primary = price
            if report.key_resistance_level is not None:
                entry_low = price
                entry_high = report.key_resistance_level
                primary = (entry_low + entry_high) / 2
            else:
                entry_low = price * 0.995
                entry_high = price * 1.005
        else:
            return None

        return EntryZone(
            primary_entry=round(primary, 8),
            entry_low=round(entry_low, 8),
            entry_high=round(entry_high, 8),
        )

    def _determine_stop_loss(
        self,
        direction: SignalDirection,
        report: SMCAnalysisReport,
    ) -> float | None:
        """Determine stop loss from SMC levels."""
        if direction == SignalDirection.LONG:
            # Below key support or dealing range low
            if report.key_support_level is not None:
                return round(report.key_support_level * 0.998, 8)
            if report.dealing_range_low is not None:
                return round(report.dealing_range_low * 0.998, 8)
            # Fallback: 2% below current price
            return round(report.current_price * 0.98, 8)

        elif direction == SignalDirection.SHORT:
            if report.key_resistance_level is not None:
                return round(report.key_resistance_level * 1.002, 8)
            if report.dealing_range_high is not None:
                return round(report.dealing_range_high * 1.002, 8)
            return round(report.current_price * 1.02, 8)

        return None

    def _determine_targets(
        self,
        direction: SignalDirection,
        entry: float,
        stop_loss: float,
        report: SMCAnalysisReport,
    ) -> list[TargetSpec]:
        """Determine take-profit targets from SMC levels."""
        targets: list[TargetSpec] = []
        risk = abs(entry - stop_loss)
        if risk <= 0:
            return targets

        if direction == SignalDirection.LONG:
            # TP1: Equilibrium or 1:1 R:R
            if report.equilibrium_price and report.equilibrium_price > entry:
                targets.append(TargetSpec(
                    price=round(report.equilibrium_price, 8),
                    scale_out_pct=0.4,
                    rationale="equilibrium",
                ))
            else:
                targets.append(TargetSpec(
                    price=round(entry + risk, 8),
                    scale_out_pct=0.4,
                    rationale="1:1 R:R",
                ))

            # TP2: Key resistance or 2:1 R:R
            if report.key_resistance_level and report.key_resistance_level > entry:
                targets.append(TargetSpec(
                    price=round(report.key_resistance_level, 8),
                    scale_out_pct=0.3,
                    rationale="key resistance",
                ))
            else:
                targets.append(TargetSpec(
                    price=round(entry + risk * 2, 8),
                    scale_out_pct=0.3,
                    rationale="2:1 R:R",
                ))

            # TP3: Dealing range high or 3:1 R:R
            if report.dealing_range_high and report.dealing_range_high > entry:
                targets.append(TargetSpec(
                    price=round(report.dealing_range_high, 8),
                    scale_out_pct=0.3,
                    rationale="dealing range high",
                ))
            else:
                targets.append(TargetSpec(
                    price=round(entry + risk * 3, 8),
                    scale_out_pct=0.3,
                    rationale="3:1 R:R",
                ))

        elif direction == SignalDirection.SHORT:
            # Mirror for shorts
            if report.equilibrium_price and report.equilibrium_price < entry:
                targets.append(TargetSpec(
                    price=round(report.equilibrium_price, 8),
                    scale_out_pct=0.4,
                    rationale="equilibrium",
                ))
            else:
                targets.append(TargetSpec(
                    price=round(entry - risk, 8),
                    scale_out_pct=0.4,
                    rationale="1:1 R:R",
                ))

            if report.key_support_level and report.key_support_level < entry:
                targets.append(TargetSpec(
                    price=round(report.key_support_level, 8),
                    scale_out_pct=0.3,
                    rationale="key support",
                ))
            else:
                targets.append(TargetSpec(
                    price=round(entry - risk * 2, 8),
                    scale_out_pct=0.3,
                    rationale="2:1 R:R",
                ))

            if report.dealing_range_low and report.dealing_range_low < entry:
                targets.append(TargetSpec(
                    price=round(report.dealing_range_low, 8),
                    scale_out_pct=0.3,
                    rationale="dealing range low",
                ))
            else:
                targets.append(TargetSpec(
                    price=round(entry - risk * 3, 8),
                    scale_out_pct=0.3,
                    rationale="3:1 R:R",
                ))

        return targets

    @staticmethod
    def _build_structure_narrative(
        summary: SMCTimeframeSummary,
    ) -> str:
        """Build a human-readable structure narrative.

        Example: ``"Bearish BOS x2 -> Bullish CHoCH -> Bullish BOS"``
        """
        parts: list[str] = []

        # BOS events
        if summary.bos_bearish > 0:
            count_str = f" x{summary.bos_bearish}" if summary.bos_bearish > 1 else ""
            parts.append(f"Bearish BOS{count_str}")
        if summary.bos_bullish > 0:
            count_str = f" x{summary.bos_bullish}" if summary.bos_bullish > 1 else ""
            parts.append(f"Bullish BOS{count_str}")

        # CHoCH events
        if summary.choch_bearish > 0:
            parts.append("Bearish CHoCH")
        if summary.choch_bullish > 0:
            parts.append("Bullish CHoCH")

        # Last break as the most recent event
        if summary.last_break_type != "none":
            last = f"{summary.last_break_direction.capitalize()} {summary.last_break_type}"
            if parts and parts[-1] != last:
                parts.append(last)

        if not parts:
            return "No significant structure breaks"

        return " -> ".join(parts)

    def _determine_trade_verdict(
        self,
        report: SMCAnalysisReport,
        confluence: SMCConfluenceResult,
    ) -> tuple[str, SignalDirection]:
        """Determine the trade verdict and direction from the analysis."""
        score = confluence.total_confluence_points
        bias = confluence.htf_bias
        aligned = confluence.ltf_confirmation

        if bias == MarketStructureBias.UNCLEAR or score < 4.0:
            return "NO TRADE - Insufficient confluence", SignalDirection.FLAT

        if bias == MarketStructureBias.BULLISH:
            if aligned and score >= self._min_confluence:
                return "POTENTIAL LONG SETUP (High Confluence)", SignalDirection.LONG
            elif aligned:
                return "POTENTIAL LONG SETUP (Moderate Confluence)", SignalDirection.LONG
            else:
                return (
                    "WATCH LONG - HTF bullish but LTF unconfirmed",
                    SignalDirection.FLAT,
                )

        if bias == MarketStructureBias.BEARISH:
            if aligned and score >= self._min_confluence:
                return "POTENTIAL SHORT SETUP (High Confluence)", SignalDirection.SHORT
            elif aligned:
                return "POTENTIAL SHORT SETUP (Moderate Confluence)", SignalDirection.SHORT
            else:
                return (
                    "WATCH SHORT - HTF bearish but LTF unconfirmed",
                    SignalDirection.FLAT,
                )

        # NEUTRAL
        return "NO TRADE - Neutral market structure", SignalDirection.FLAT

    def _extract_key_levels(
        self,
        report: SMCAnalysisReport,
        features: dict[str, float],
        confluence: SMCConfluenceResult,
    ) -> None:
        """Extract key support/resistance levels from features."""
        # Try HTF timeframe prefix first
        htf_prefix = report.htf_timeframe.value

        # Support: nearest demand (unmitigated bullish OB) or dealing range low
        demand_dist = features.get(
            f"{htf_prefix}_smc_nearest_demand_distance",
            features.get("smc_nearest_demand_distance", 0.0),
        )
        if demand_dist > 0 and report.current_price > 0:
            support = report.current_price * (1.0 - demand_dist)
            report.key_support_level = round(support, 8)
            report.key_support_description = "unmitigated demand zone"

        # If no demand zone, use dealing range low
        if report.key_support_level is None and report.dealing_range_low is not None:
            report.key_support_level = report.dealing_range_low
            report.key_support_description = "dealing range low"

        # Resistance: nearest supply or dealing range high
        supply_dist = features.get(
            f"{htf_prefix}_smc_nearest_supply_distance",
            features.get("smc_nearest_supply_distance", 0.0),
        )
        if supply_dist > 0 and report.current_price > 0:
            resistance = report.current_price * (1.0 + supply_dist)
            report.key_resistance_level = round(resistance, 8)
            report.key_resistance_description = "unmitigated supply zone"

        if report.key_resistance_level is None and report.dealing_range_high is not None:
            report.key_resistance_level = report.dealing_range_high
            report.key_resistance_description = "dealing range high"

    @staticmethod
    def _score_to_conviction(score: float) -> ConvictionLevel:
        """Map confluence score to conviction level."""
        if score >= 10.0:
            return ConvictionLevel.HIGH
        if score >= 7.0:
            return ConvictionLevel.MODERATE
        return ConvictionLevel.LOW

    def _build_invalidation_conditions(
        self,
        direction: SignalDirection,
        report: SMCAnalysisReport,
        features: dict[str, float],
    ) -> list[InvalidationCondition]:
        """Build invalidation conditions for the trade setup."""
        conditions: list[InvalidationCondition] = []

        if direction == SignalDirection.LONG:
            # Price breaks below key support
            if report.key_support_level is not None:
                conditions.append(InvalidationCondition(
                    description=f"Price breaks below {report.key_support_description}",
                    trigger_price=report.key_support_level,
                    trigger_condition="close_below",
                ))

            # LTF bearish BOS
            if report.ltf_analysis and (
                report.ltf_analysis.bos_bearish > 0
                or report.ltf_analysis.choch_bearish > 0
            ):
                conditions.append(InvalidationCondition(
                    description=(
                        f"{report.ltf_timeframe.value.upper()} forms new "
                        f"bearish structure break"
                    ),
                ))

            # Both TFs flip bearish
            conditions.append(InvalidationCondition(
                description="Both timeframes flip to bearish structure",
            ))

        elif direction == SignalDirection.SHORT:
            if report.key_resistance_level is not None:
                conditions.append(InvalidationCondition(
                    description=f"Price breaks above {report.key_resistance_description}",
                    trigger_price=report.key_resistance_level,
                    trigger_condition="close_above",
                ))

            if report.ltf_analysis and (
                report.ltf_analysis.bos_bullish > 0
                or report.ltf_analysis.choch_bullish > 0
            ):
                conditions.append(InvalidationCondition(
                    description=(
                        f"{report.ltf_timeframe.value.upper()} forms new "
                        f"bullish structure break"
                    ),
                ))

            conditions.append(InvalidationCondition(
                description="Both timeframes flip to bullish structure",
            ))

        return conditions

    @staticmethod
    def _bias_emoji(bias: MarketStructureBias) -> str:
        """Return emoji indicator for bias."""
        if bias == MarketStructureBias.BULLISH:
            return "+"
        if bias == MarketStructureBias.BEARISH:
            return "-"
        return "?"
