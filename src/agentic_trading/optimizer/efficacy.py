"""Efficacy analysis engine — diagnoses why trades are losing.

Implements the diagnostic priority order:
1. Costs / execution (suspect first)
2. Exit geometry
3. Regime mismatch
4. Risk / sizing
5. Signal edge (suspect last)

All functions are pure — no I/O, no side effects.  The EfficacyAgent
calls this module and handles persistence and event publishing.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from agentic_trading.backtester.results import TradeDetail

from .efficacy_models import (
    DRIVER_PRIORITY,
    DataIntegrityResult,
    EfficacyReport,
    LossDriverBreakdown,
    LossDriverCategory,
    SegmentAnalysis,
)

logger = logging.getLogger(__name__)

# Minimum trades per segment for statistical validity
MIN_TRADES_DEFAULT = 50


class EfficacyAnalyzer:
    """Stateless analyser that diagnoses loss drivers from trade details.

    Usage::

        analyzer = EfficacyAnalyzer(min_trades=50)
        report = analyzer.analyze(trade_details, strategy_id="bb_squeeze")
    """

    def __init__(self, min_trades: int = MIN_TRADES_DEFAULT) -> None:
        self._min_trades = min_trades

    def analyze(
        self,
        trades: list[TradeDetail],
        strategy_id: str = "",
    ) -> EfficacyReport:
        """Run the full efficacy diagnostic pipeline.

        Returns an EfficacyReport with data integrity result,
        segment breakdowns, loss driver analysis, and recommendations.
        """
        report = EfficacyReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy_id=strategy_id,
        )

        # Phase 0: Data integrity
        report.data_integrity = self._check_data_integrity(trades)
        if not report.data_integrity.passed:
            report.recommendations.append(
                "DATA INTEGRITY FAILURE: Fix trade data capture before analysis. "
                f"Issues: {'; '.join(report.data_integrity.issues)}"
            )
            return report

        # Aggregate stats
        report.total_trades = len(trades)
        report.min_trades_met = len(trades) >= self._min_trades

        returns = [t.return_pct for t in trades]
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r < 0]

        report.win_rate = len(winners) / len(returns) if returns else 0.0
        report.avg_return = sum(returns) / len(returns) if returns else 0.0
        report.total_return = sum(returns)

        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0
        report.profit_factor = (
            round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf")
        )

        # Phase 1: Segment & Diagnose
        report.segments = self._build_segments(trades)
        report.loss_drivers = self._diagnose_loss_drivers(trades)

        # Phase 2: Generate recommendations
        report.recommendations = self._generate_recommendations(
            report.loss_drivers, report.segments, trades
        )

        return report

    # ------------------------------------------------------------------
    # Phase 0: Data Integrity
    # ------------------------------------------------------------------

    def _check_data_integrity(
        self, trades: list[TradeDetail]
    ) -> DataIntegrityResult:
        """Verify trade data is complete and usable."""
        result = DataIntegrityResult(total_trades=len(trades))

        if len(trades) == 0:
            result.passed = False
            result.issues.append("No trades to analyse")
            return result

        # Check for zero/NaN prices
        bad_prices = sum(
            1
            for t in trades
            if t.entry_price <= 0
            or t.exit_price <= 0
            or math.isnan(t.entry_price)
            or math.isnan(t.exit_price)
        )
        if bad_prices > 0:
            result.has_prices = False
            result.issues.append(
                f"{bad_prices}/{len(trades)} trades have zero/NaN prices"
            )

        # Check fees populated
        fee_trades = sum(1 for t in trades if t.fee_paid > 0)
        if fee_trades == 0:
            result.has_fees = False
            result.issues.append("No trades have fee data — cost analysis unreliable")

        # Check MAE/MFE populated
        mae_nonzero = sum(1 for t in trades if t.mae_pct != 0.0)
        mfe_nonzero = sum(1 for t in trades if t.mfe_pct != 0.0)
        if mae_nonzero == 0 and mfe_nonzero == 0:
            result.has_mae_mfe = False
            result.issues.append(
                "MAE/MFE all zero — exit geometry analysis will be limited"
            )

        # Check timestamps
        ts_missing = sum(1 for t in trades if not t.entry_time or not t.exit_time)
        if ts_missing > len(trades) * 0.5:
            result.has_timestamps = False
            result.issues.append(
                f"{ts_missing}/{len(trades)} trades missing timestamps"
            )

        # Still pass if we have issues but data is usable
        result.passed = result.has_prices and len(trades) > 0
        return result

    # ------------------------------------------------------------------
    # Phase 1: Segmentation
    # ------------------------------------------------------------------

    def _build_segments(
        self, trades: list[TradeDetail]
    ) -> dict[str, SegmentAnalysis]:
        """Build segment analyses by symbol, direction, strategy, exit_reason."""
        segments: dict[str, SegmentAnalysis] = {}

        # By symbol
        by_symbol: dict[str, list[TradeDetail]] = defaultdict(list)
        for t in trades:
            by_symbol[t.symbol].append(t)
        for symbol, group in by_symbol.items():
            key = f"symbol:{symbol}"
            segments[key] = self._compute_segment(key, "symbol", group)

        # By direction
        by_dir: dict[str, list[TradeDetail]] = defaultdict(list)
        for t in trades:
            by_dir[t.direction].append(t)
        for direction, group in by_dir.items():
            key = f"direction:{direction}"
            segments[key] = self._compute_segment(key, "direction", group)

        # By strategy
        by_strat: dict[str, list[TradeDetail]] = defaultdict(list)
        for t in trades:
            by_strat[t.strategy_id].append(t)
        for strat, group in by_strat.items():
            key = f"strategy:{strat}"
            segments[key] = self._compute_segment(key, "strategy", group)

        # By exit reason
        by_exit: dict[str, list[TradeDetail]] = defaultdict(list)
        for t in trades:
            by_exit[t.exit_reason].append(t)
        for reason, group in by_exit.items():
            key = f"exit:{reason}"
            segments[key] = self._compute_segment(key, "exit_reason", group)

        return segments

    def _compute_segment(
        self,
        name: str,
        segment_type: str,
        trades: list[TradeDetail],
    ) -> SegmentAnalysis:
        """Compute aggregate metrics for a trade segment."""
        n = len(trades)
        if n == 0:
            return SegmentAnalysis(segment_name=name, segment_type=segment_type)

        returns = [t.return_pct for t in trades]
        winners = [t for t in trades if t.return_pct > 0]
        losers = [t for t in trades if t.return_pct < 0]

        gross_profit = sum(t.return_pct for t in winners)
        gross_loss = abs(sum(t.return_pct for t in losers))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Management efficiency: for winners, how much of MFE was captured
        mgmt_effs = []
        for t in winners:
            if t.mfe_pct > 0:
                mgmt_effs.append(t.return_pct / t.mfe_pct)
        mgmt_eff = sum(mgmt_effs) / len(mgmt_effs) if mgmt_effs else 0.0

        # Fee drag
        fee_drags = []
        for t in trades:
            notional = t.entry_price * t.qty
            if notional > 0:
                fee_drags.append(t.fee_paid / notional)
        fee_drag = sum(fee_drags) / len(fee_drags) if fee_drags else 0.0

        stop_hits = sum(1 for t in trades if t.exit_reason == "stop_loss")

        return SegmentAnalysis(
            segment_name=name,
            segment_type=segment_type,
            trade_count=n,
            win_rate=len(winners) / n,
            avg_return=sum(returns) / n,
            total_return=sum(returns),
            profit_factor=round(pf, 4),
            avg_hold_seconds=sum(t.hold_seconds for t in trades) / n,
            avg_mae_pct=sum(t.mae_pct for t in trades) / n,
            avg_mfe_pct=sum(t.mfe_pct for t in trades) / n,
            management_efficiency=round(mgmt_eff, 4),
            fee_drag_pct=round(fee_drag, 6),
            stop_hit_rate=round(stop_hits / n, 4),
        )

    # ------------------------------------------------------------------
    # Phase 1: Loss Driver Diagnosis
    # ------------------------------------------------------------------

    def _diagnose_loss_drivers(
        self, trades: list[TradeDetail]
    ) -> list[LossDriverBreakdown]:
        """Diagnose loss drivers in priority order."""
        drivers: list[LossDriverBreakdown] = []

        total_losses = abs(sum(t.return_pct for t in trades if t.return_pct < 0))

        # 1. Execution / Cost analysis (suspect first)
        drivers.append(self._diagnose_costs(trades, total_losses))

        # 2. Exit geometry
        drivers.append(self._diagnose_exit_geometry(trades, total_losses))

        # 3. Regime mismatch
        drivers.append(self._diagnose_regime_mismatch(trades, total_losses))

        # 4. Risk / sizing
        drivers.append(self._diagnose_risk_sizing(trades, total_losses))

        # 5. Signal edge (suspect last)
        drivers.append(self._diagnose_signal_edge(trades, total_losses))

        # Sort by impact (highest total_pnl_impact first)
        drivers.sort(key=lambda d: abs(d.total_pnl_impact), reverse=True)

        return drivers

    def _diagnose_costs(
        self, trades: list[TradeDetail], total_losses: float
    ) -> LossDriverBreakdown:
        """Diagnose execution cost as a loss driver.

        Checks whether fee drag is a material fraction of average loss.
        """
        fee_drags = []
        notional_weighted_fees = 0.0
        total_notional = 0.0

        for t in trades:
            notional = t.entry_price * t.qty
            if notional > 0:
                fee_drags.append(t.fee_paid / notional)
                notional_weighted_fees += t.fee_paid
                total_notional += notional

        avg_fee_drag = (
            sum(fee_drags) / len(fee_drags) if fee_drags else 0.0
        )
        total_fee_pct = (
            notional_weighted_fees / total_notional if total_notional > 0 else 0.0
        )

        # How much of total losses are explained by fees?
        losers = [t for t in trades if t.return_pct < 0]
        avg_loss_mag = (
            abs(sum(t.return_pct for t in losers)) / len(losers)
            if losers
            else 0.0
        )
        fee_as_loss_share = (
            avg_fee_drag / avg_loss_mag if avg_loss_mag > 0 else 0.0
        )

        # Severity
        if fee_as_loss_share > 0.5:
            severity = "critical"
            diagnosis = (
                f"Fee drag ({avg_fee_drag:.4%}) accounts for {fee_as_loss_share:.0%} "
                f"of average loss magnitude ({avg_loss_mag:.4%}). "
                "Costs are a dominant loss driver — consider reducing trade frequency "
                "or negotiating lower fees."
            )
        elif fee_as_loss_share > 0.2:
            severity = "warning"
            diagnosis = (
                f"Fee drag ({avg_fee_drag:.4%}) is {fee_as_loss_share:.0%} of avg loss. "
                "Material cost impact — review trade frequency."
            )
        else:
            severity = "info"
            diagnosis = (
                f"Fee drag ({avg_fee_drag:.4%}) is {fee_as_loss_share:.0%} of avg loss. "
                "Costs are not the primary loss driver."
            )

        # Total PnL impact of fees
        total_fee_impact = total_fee_pct * len(trades)

        return LossDriverBreakdown(
            category=LossDriverCategory.EXECUTION_COST,
            trade_count=len(trades),
            avg_loss_pct=avg_fee_drag,
            total_pnl_impact=-total_fee_impact,
            share_of_total_losses=(
                total_fee_impact / total_losses if total_losses > 0 else 0.0
            ),
            diagnosis=diagnosis,
            severity=severity,
        )

    def _diagnose_exit_geometry(
        self, trades: list[TradeDetail], total_losses: float
    ) -> LossDriverBreakdown:
        """Diagnose exit timing as a loss driver.

        Checks management efficiency (MFE capture) and stop-loss hit rate.
        """
        winners = [t for t in trades if t.return_pct > 0]
        losers = [t for t in trades if t.return_pct < 0]

        # Management efficiency for winners: actual / MFE
        mgmt_effs = []
        for t in winners:
            if t.mfe_pct > 0:
                mgmt_effs.append(t.return_pct / t.mfe_pct)
        avg_mgmt_eff = sum(mgmt_effs) / len(mgmt_effs) if mgmt_effs else 0.0

        # Stop-loss hit rate
        stop_hits = sum(1 for t in trades if t.exit_reason == "stop_loss")
        stop_rate = stop_hits / len(trades) if trades else 0.0

        # Lost profit: how much MFE was left on table (winners)
        lost_profit = 0.0
        for t in winners:
            if t.mfe_pct > t.return_pct:
                lost_profit += t.mfe_pct - t.return_pct

        # Losers that had positive MFE (could have exited at profit)
        losers_with_mfe = [t for t in losers if t.mfe_pct > 0]
        missed_exit_rate = (
            len(losers_with_mfe) / len(losers) if losers else 0.0
        )

        total_missed = sum(t.mfe_pct for t in losers_with_mfe)

        total_impact = lost_profit + total_missed

        if avg_mgmt_eff < 0.3 or missed_exit_rate > 0.5:
            severity = "critical"
            diagnosis = (
                f"Exit geometry is poor: management efficiency {avg_mgmt_eff:.1%}, "
                f"stop-hit rate {stop_rate:.1%}, "
                f"{missed_exit_rate:.0%} of losers had positive MFE "
                f"(could have exited at profit). "
                "Consider tighter trailing stops or earlier profit-taking."
            )
        elif avg_mgmt_eff < 0.5 or missed_exit_rate > 0.3:
            severity = "warning"
            diagnosis = (
                f"Exit timing leaves profit on table: mgmt eff {avg_mgmt_eff:.1%}, "
                f"{missed_exit_rate:.0%} of losers saw profit first. "
                "Trailing stops or partial exits may help."
            )
        else:
            severity = "info"
            diagnosis = (
                f"Exit geometry acceptable: mgmt eff {avg_mgmt_eff:.1%}, "
                f"stop rate {stop_rate:.1%}."
            )

        return LossDriverBreakdown(
            category=LossDriverCategory.EXIT_GEOMETRY,
            trade_count=len(trades),
            avg_loss_pct=round(1.0 - avg_mgmt_eff, 6),
            total_pnl_impact=-total_impact,
            share_of_total_losses=(
                total_impact / total_losses if total_losses > 0 else 0.0
            ),
            diagnosis=diagnosis,
            severity=severity,
        )

    def _diagnose_regime_mismatch(
        self, trades: list[TradeDetail], total_losses: float
    ) -> LossDriverBreakdown:
        """Diagnose regime mismatch by comparing segments.

        Checks for:
        - Symbols dragging performance
        - Long/short asymmetry
        - Time-based regime shifts (first half vs second half)
        """
        issues: list[str] = []
        impact = 0.0

        # --- Symbol drag ---
        by_symbol: dict[str, list[TradeDetail]] = defaultdict(list)
        for t in trades:
            by_symbol[t.symbol].append(t)

        total_avg = (
            sum(t.return_pct for t in trades) / len(trades) if trades else 0.0
        )

        worst_symbols: list[tuple[str, float]] = []
        for symbol, group in by_symbol.items():
            if len(group) < 5:
                continue
            avg_ret = sum(t.return_pct for t in group) / len(group)
            if avg_ret < total_avg - 0.001:  # Notably worse than average
                worst_symbols.append((symbol, avg_ret))
                impact += abs(sum(t.return_pct for t in group))

        if worst_symbols:
            worst_symbols.sort(key=lambda x: x[1])
            names = [f"{s} ({r:.4%})" for s, r in worst_symbols[:3]]
            issues.append(f"Underperforming symbols: {', '.join(names)}")

        # --- Direction asymmetry ---
        by_dir: dict[str, list[TradeDetail]] = defaultdict(list)
        for t in trades:
            by_dir[t.direction].append(t)

        for direction, group in by_dir.items():
            if len(group) < 10:
                continue
            dir_avg = sum(t.return_pct for t in group) / len(group)
            dir_wr = sum(1 for t in group if t.return_pct > 0) / len(group)
            if dir_wr < 0.35:
                issues.append(
                    f"{direction} trades underperform: WR={dir_wr:.1%}, "
                    f"avg={dir_avg:.4%}"
                )

        # --- Time-based shift ---
        sorted_trades = sorted(trades, key=lambda t: t.entry_time)
        mid = len(sorted_trades) // 2
        if mid > 10:
            first_half = sorted_trades[:mid]
            second_half = sorted_trades[mid:]
            first_avg = sum(t.return_pct for t in first_half) / len(first_half)
            second_avg = sum(t.return_pct for t in second_half) / len(second_half)
            if abs(first_avg - second_avg) > 0.002:
                issues.append(
                    f"Regime shift detected: 1st half avg={first_avg:.4%}, "
                    f"2nd half avg={second_avg:.4%}"
                )

        if not issues:
            severity = "info"
            diagnosis = "No significant regime mismatch detected."
        elif len(issues) >= 2:
            severity = "critical"
            diagnosis = "Multiple regime issues: " + "; ".join(issues)
        else:
            severity = "warning"
            diagnosis = issues[0]

        return LossDriverBreakdown(
            category=LossDriverCategory.REGIME_MISMATCH,
            trade_count=len(trades),
            avg_loss_pct=0.0,
            total_pnl_impact=-impact,
            share_of_total_losses=(
                impact / total_losses if total_losses > 0 else 0.0
            ),
            diagnosis=diagnosis,
            severity=severity,
        )

    def _diagnose_risk_sizing(
        self, trades: list[TradeDetail], total_losses: float
    ) -> LossDriverBreakdown:
        """Diagnose risk/sizing issues.

        Checks for oversized losing trades and position concentration.
        """
        if not trades:
            return LossDriverBreakdown(
                category=LossDriverCategory.RISK_SIZING,
                diagnosis="No trades to analyse.",
                severity="info",
            )

        losses = [t for t in trades if t.return_pct < 0]
        if not losses:
            return LossDriverBreakdown(
                category=LossDriverCategory.RISK_SIZING,
                diagnosis="No losing trades.",
                severity="info",
            )

        # Check for fat-tail losses (losses > 3x average loss)
        avg_loss = abs(sum(t.return_pct for t in losses)) / len(losses)
        fat_tails = [t for t in losses if abs(t.return_pct) > avg_loss * 3]
        fat_tail_impact = abs(sum(t.return_pct for t in fat_tails))

        # Check hold time vs loss: are long-held trades losing more?
        long_holds = [t for t in trades if t.hold_seconds > 7200]  # > 2h
        short_holds = [t for t in trades if 0 < t.hold_seconds <= 7200]

        long_avg = (
            sum(t.return_pct for t in long_holds) / len(long_holds)
            if long_holds
            else 0.0
        )
        short_avg = (
            sum(t.return_pct for t in short_holds) / len(short_holds)
            if short_holds
            else 0.0
        )

        issues: list[str] = []
        if fat_tails:
            issues.append(
                f"{len(fat_tails)} fat-tail losses (> 3x avg) account for "
                f"{fat_tail_impact:.4%} total impact"
            )
        if long_holds and short_holds and long_avg < short_avg - 0.002:
            issues.append(
                f"Longer holds ({long_avg:.4%}) underperform shorter ({short_avg:.4%})"
            )

        if fat_tails and len(fat_tails) > len(losses) * 0.1:
            severity = "critical"
        elif issues:
            severity = "warning"
        else:
            severity = "info"

        diagnosis = "; ".join(issues) if issues else "Sizing appears reasonable."

        return LossDriverBreakdown(
            category=LossDriverCategory.RISK_SIZING,
            trade_count=len(losses),
            avg_loss_pct=avg_loss,
            total_pnl_impact=-fat_tail_impact,
            share_of_total_losses=(
                fat_tail_impact / total_losses if total_losses > 0 else 0.0
            ),
            diagnosis=diagnosis,
            severity=severity,
        )

    def _diagnose_signal_edge(
        self, trades: list[TradeDetail], total_losses: float
    ) -> LossDriverBreakdown:
        """Diagnose signal edge (suspect LAST).

        Compares actual win rate against a random baseline to determine
        whether the signal has genuine predictive power.
        """
        if not trades or len(trades) < 20:
            return LossDriverBreakdown(
                category=LossDriverCategory.SIGNAL_EDGE,
                trade_count=len(trades),
                diagnosis="Insufficient trades for edge analysis.",
                severity="info",
            )

        actual_wr = sum(1 for t in trades if t.return_pct > 0) / len(trades)

        # Random baseline: shuffle returns and measure win rate
        # This approximates what random entry with same exit logic yields
        returns = [t.return_pct for t in trades]
        rng = random.Random(42)
        shuffle_wrs = []
        for _ in range(100):
            shuffled = returns.copy()
            rng.shuffle(shuffled)
            shuffle_wr = sum(1 for r in shuffled if r > 0) / len(shuffled)
            shuffle_wrs.append(shuffle_wr)

        random_wr = sum(shuffle_wrs) / len(shuffle_wrs)

        # Edge = actual WR - random baseline WR
        edge = actual_wr - random_wr

        # Also check per-strategy edge
        by_strat: dict[str, list[TradeDetail]] = defaultdict(list)
        for t in trades:
            by_strat[t.strategy_id].append(t)

        zero_edge_strats = []
        for strat, group in by_strat.items():
            if len(group) < 10:
                continue
            strat_wr = sum(1 for t in group if t.return_pct > 0) / len(group)
            if strat_wr < 0.4:
                zero_edge_strats.append(f"{strat} (WR={strat_wr:.1%})")

        total_impact = abs(sum(t.return_pct for t in trades if t.return_pct < 0))

        if abs(edge) < 0.02 and actual_wr < 0.45:
            severity = "critical"
            diagnosis = (
                f"Signal shows NO edge: actual WR={actual_wr:.1%}, "
                f"random baseline={random_wr:.1%}, edge={edge:+.1%}. "
                "Strategy entries are no better than random."
            )
            if zero_edge_strats:
                diagnosis += (
                    f" Weakest strategies: {', '.join(zero_edge_strats)}."
                )
        elif actual_wr < 0.45:
            severity = "warning"
            diagnosis = (
                f"Weak signal edge: actual WR={actual_wr:.1%}, "
                f"random={random_wr:.1%}, edge={edge:+.1%}."
            )
        else:
            severity = "info"
            diagnosis = (
                f"Signal has edge: actual WR={actual_wr:.1%}, "
                f"random={random_wr:.1%}, edge={edge:+.1%}."
            )

        return LossDriverBreakdown(
            category=LossDriverCategory.SIGNAL_EDGE,
            trade_count=len(trades),
            avg_loss_pct=0.0,
            total_pnl_impact=-total_impact if severity == "critical" else 0.0,
            share_of_total_losses=(
                1.0 if severity == "critical" else 0.0
            ),
            diagnosis=diagnosis,
            severity=severity,
        )

    # ------------------------------------------------------------------
    # Phase 2: Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        drivers: list[LossDriverBreakdown],
        segments: dict[str, SegmentAnalysis],
        trades: list[TradeDetail],
    ) -> list[str]:
        """Generate prioritised, actionable recommendations.

        Follows the user's mandate: costs → exits → regime → signal.
        """
        recs: list[str] = []

        # Sort drivers by priority order, then by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        priority_map = {cat: i for i, cat in enumerate(DRIVER_PRIORITY)}

        sorted_drivers = sorted(
            drivers,
            key=lambda d: (
                severity_order.get(d.severity, 9),
                priority_map.get(d.category, 9),
            ),
        )

        for driver in sorted_drivers:
            if driver.severity == "critical":
                recs.append(
                    f"[CRITICAL] {driver.category.value}: {driver.diagnosis}"
                )
            elif driver.severity == "warning":
                recs.append(
                    f"[WARNING] {driver.category.value}: {driver.diagnosis}"
                )

        # Add specific actionable items based on segment data
        for key, seg in segments.items():
            if seg.segment_type == "exit_reason" and seg.segment_name == "exit:stop_loss":
                if seg.trade_count > 0 and seg.stop_hit_rate > 0.6:
                    recs.append(
                        f"High stop-loss hit rate ({seg.stop_hit_rate:.0%}) — "
                        "consider widening stops or reducing position size."
                    )
            if seg.segment_type == "symbol" and seg.trade_count >= 10:
                if seg.win_rate < 0.3:
                    recs.append(
                        f"Consider disabling {seg.segment_name}: "
                        f"WR={seg.win_rate:.0%} across {seg.trade_count} trades."
                    )

        if not recs:
            recs.append("No critical issues found. Strategy performance is acceptable.")

        return recs
