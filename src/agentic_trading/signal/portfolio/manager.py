"""Portfolio manager: aggregates signals into target positions.

Takes signals from multiple strategies, applies sizing,
resolves conflicts, and outputs TargetPosition events.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Callable

from agentic_trading.core.enums import Side, SignalDirection
from agentic_trading.core.events import Signal, TargetPosition
from agentic_trading.core.interfaces import PortfolioState, TradingContext
from agentic_trading.core.models import Instrument

from .sizing import (
    fixed_fractional_size,
    liquidity_adjusted_size,
    stop_loss_based_size,
    volatility_adjusted_size,
)

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Aggregates strategy signals into target positions."""

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_gross_exposure_pct: float = 1.0,
        sizing_multiplier: float = 1.0,
        governance_sizing_fn: Callable[[str], float] | None = None,
    ) -> None:
        self._max_position_pct = max_position_pct
        self._max_gross_exposure = max_gross_exposure_pct
        self._sizing_multiplier = sizing_multiplier
        self._governance_sizing_fn = governance_sizing_fn
        self._pending_signals: list[Signal] = []

    def on_signal(self, signal: Signal) -> None:
        """Collect signals for aggregation."""
        self._pending_signals.append(signal)

    def generate_targets(
        self,
        ctx: TradingContext,
        capital: float,
    ) -> list[TargetPosition]:
        """Process pending signals and generate target positions.

        Resolves conflicts (multiple strategies, same symbol) by
        confidence-weighted voting.  Position-aware: suppresses signals
        that duplicate or conflict with existing exchange positions.
        """
        if not self._pending_signals:
            return []

        total_signals = len(self._pending_signals)

        # Group signals by symbol
        by_symbol: dict[str, list[Signal]] = {}
        for sig in self._pending_signals:
            by_symbol.setdefault(sig.symbol, []).append(sig)

        targets = []
        for symbol, signals in by_symbol.items():
            target = self._resolve_signals(symbol, signals, ctx, capital)
            if target is not None:
                targets.append(target)

        self._pending_signals.clear()

        # Batch summary log
        if targets:
            logger.info(
                "Portfolio coordinator: %d signals → %d targets. %s",
                total_signals,
                len(targets),
                ", ".join(f"{t.symbol} {t.side.value}" for t in targets),
            )
        elif total_signals > 0:
            logger.info(
                "Portfolio coordinator: %d signals → 0 targets "
                "(all suppressed or cancelled)",
                total_signals,
            )

        return targets

    def _resolve_signals(
        self,
        symbol: str,
        signals: list[Signal],
        ctx: TradingContext,
        capital: float,
    ) -> TargetPosition | None:
        """Resolve multiple signals for the same symbol.

        Uses confidence-weighted voting + position-awareness:
        - Sum confidence × direction for each signal
        - Net direction determines final side
        - Suppress if existing position already matches or conflicts
        - Average confidence determines size
        """
        if not signals:
            return None

        # Confidence-weighted direction sum
        net_score = 0.0
        total_confidence = 0.0
        reasons = []

        for sig in signals:
            weight = sig.confidence
            if sig.direction == SignalDirection.LONG:
                net_score += weight
            elif sig.direction == SignalDirection.SHORT:
                net_score -= weight
            total_confidence += weight
            reasons.append(f"{sig.strategy_id}: {sig.direction.value} ({sig.confidence:.2f})")

        if abs(net_score) < 0.1:
            logger.info(
                "Signal conflict cancelled for %s: net_score=%.3f "
                "— conflicting strategies neutralized. Signals: %s",
                symbol, net_score, " | ".join(reasons),
            )
            return None  # Conflicting signals cancel out

        # Determine winning direction and side
        if net_score > 0:
            side = Side.BUY
            winning_direction = "long"
        else:
            side = Side.SELL
            winning_direction = "short"

        # ---- Position-aware suppression ----
        existing_pos = None
        if ctx.portfolio_state:
            existing_pos = ctx.portfolio_state.get_position(symbol)

        if existing_pos is not None and existing_pos.is_open:
            existing_direction = (
                existing_pos.side.value
                if hasattr(existing_pos.side, "value")
                else str(existing_pos.side)
            )

            # Already positioned in the same direction → suppress (no stacking)
            if existing_direction == winning_direction:
                logger.info(
                    "Suppressed %s signal for %s: already %s (qty=%s). "
                    "Signals: %s",
                    winning_direction.upper(), symbol,
                    existing_direction, existing_pos.qty,
                    " | ".join(reasons),
                )
                return None

            # Opposite direction → suppress entry.  Strategies must fire
            # FLAT to close first, then a new entry on the next candle.
            # PreTradeChecker also catches this, but suppressing here
            # avoids needless sizing work and order churn.
            if existing_direction != winning_direction:
                logger.warning(
                    "Suppressed %s signal for %s: existing position is %s "
                    "(qty=%s). Use FLAT to close first. Signals: %s",
                    winning_direction.upper(), symbol,
                    existing_direction, existing_pos.qty,
                    " | ".join(reasons),
                )
                return None

        # ---- End position-aware suppression ----

        avg_confidence = total_confidence / len(signals) if signals else 0
        instrument = ctx.get_instrument(symbol)

        # Size the position
        qty = self._compute_size(
            signals, instrument, capital, avg_confidence
        )

        if qty <= Decimal("0"):
            best_rc = max(signals, key=lambda s: s.confidence).risk_constraints if signals else {}
            logger.warning(
                "No target for %s: qty=0 (price_in_rc=%s, instrument=%s, close=%s)",
                symbol,
                best_rc.get("price", "missing"),
                instrument is not None,
                max(signals, key=lambda s: s.confidence).features_used.get("close", "missing") if signals else "?",
            )
            return None

        # Use the highest-confidence signal as the "best" signal — this is
        # the same signal used by _compute_size for risk_constraints, and its
        # trace_id is the key under which the signal was cached by the fill
        # handler.  Forwarding it ensures the fill handler can look up the
        # original Signal and retrieve its TP/SL values.
        best_signal = max(signals, key=lambda s: s.confidence)

        # Extract price estimate for downstream allocator
        rc = best_signal.risk_constraints
        price_est = float(rc.get("price", 0))
        if price_est <= 0 and best_signal.features_used:
            price_est = float(best_signal.features_used.get("close", 0))

        logger.info(
            "Target: %s %s qty=%s (confidence=%.2f, capital=%.0f, trace=%s)",
            side.value, symbol, qty, avg_confidence, capital,
            best_signal.trace_id[:8],
        )

        return TargetPosition(
            strategy_id=best_signal.strategy_id,
            symbol=symbol,
            target_qty=qty,
            side=side,
            reason=" | ".join(reasons),
            urgency=min(1.0, avg_confidence),
            trace_id=best_signal.trace_id,
            price_estimate=price_est,
        )

    def _compute_size(
        self,
        signals: list[Signal],
        instrument: Instrument | None,
        capital: float,
        confidence: float,
    ) -> Decimal:
        """Compute position size based on signal risk constraints."""
        if not signals:
            return Decimal("0")

        best_signal = max(signals, key=lambda s: s.confidence)
        rc = best_signal.risk_constraints

        sizing_method = rc.get("sizing_method", "fixed_fractional")
        atr = rc.get("atr", 0)
        price = float(
            rc.get("price", 0)
            or (instrument.tick_size if instrument else 0)
        )

        # Use candle close from features if available
        if price <= 0 and best_signal.features_used:
            price = best_signal.features_used.get("close", 0)
        if price <= 0:
            return Decimal("0")

        risk_pct = min(self._max_position_pct, confidence * 0.05)

        if sizing_method == "volatility_adjusted" and atr > 0:
            qty = volatility_adjusted_size(
                capital=capital,
                risk_per_trade_pct=risk_pct,
                atr=atr,
                price=price,
                instrument=instrument,
            )
        elif sizing_method == "stop_loss_based":
            entry_price = float(rc.get("entry", price))
            stop_loss_price = float(rc.get("stop_loss", 0))
            risk_pct_override = float(rc.get("risk_pct", risk_pct))
            if stop_loss_price > 0 and entry_price > 0:
                qty = stop_loss_based_size(
                    capital=capital,
                    risk_per_trade_pct=risk_pct_override,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    instrument=instrument,
                )
            else:
                qty = fixed_fractional_size(
                    capital=capital,
                    fraction=risk_pct,
                    price=price,
                    instrument=instrument,
                )
        elif sizing_method == "liquidity_adjusted":
            base = fixed_fractional_size(
                capital=capital,
                fraction=risk_pct,
                price=price,
                instrument=instrument,
            )
            liq_score = rc.get("liquidity_score", 1.0)
            qty = liquidity_adjusted_size(base, liq_score, instrument=instrument)
        else:
            qty = fixed_fractional_size(
                capital=capital,
                fraction=risk_pct,
                price=price,
                instrument=instrument,
            )

        # Cap notional value at max_position_pct of capital.
        # Volatility-adjusted sizing with tiny ATR (e.g. 1-minute candles)
        # can produce enormous positions that exceed available capital.
        max_notional = capital * self._max_position_pct
        notional = float(qty) * price
        if notional > max_notional and notional > 0:
            scale = max_notional / notional
            qty = Decimal(str(float(qty) * scale))
            logger.info(
                "Notional cap applied for %s: %.0f → %.0f (max %.0f, %.0f%%)",
                best_signal.symbol,
                notional,
                float(qty) * price,
                max_notional,
                self._max_position_pct * 100,
            )

        # Apply sizing multiplier (from safe_mode / regime policy)
        qty = Decimal(str(float(qty) * self._sizing_multiplier))

        # Apply governance sizing (maturity cap × health multiplier)
        if self._governance_sizing_fn is not None:
            gov_mult = self._governance_sizing_fn(best_signal.strategy_id)
            qty = Decimal(str(float(qty) * gov_mult))

        if instrument:
            qty = instrument.round_qty(qty)

        return qty

    def set_sizing_multiplier(self, mult: float) -> None:
        """Update sizing multiplier (called by regime policy)."""
        self._sizing_multiplier = max(0.0, min(2.0, mult))
