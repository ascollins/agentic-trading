"""Portfolio manager: aggregates signals into target positions.

Takes signals from multiple strategies, applies sizing,
resolves conflicts, and outputs TargetPosition events.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

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
    ) -> None:
        self._max_position_pct = max_position_pct
        self._max_gross_exposure = max_gross_exposure_pct
        self._sizing_multiplier = sizing_multiplier
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
        confidence-weighted voting.
        """
        if not self._pending_signals:
            return []

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
        return targets

    def _resolve_signals(
        self,
        symbol: str,
        signals: list[Signal],
        ctx: TradingContext,
        capital: float,
    ) -> TargetPosition | None:
        """Resolve multiple signals for the same symbol.

        Uses confidence-weighted voting:
        - Sum confidence * direction for each signal
        - Net direction determines final side
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
            return None  # Conflicting signals cancel out

        # Determine direction and side
        if net_score > 0:
            side = Side.BUY
        else:
            side = Side.SELL

        avg_confidence = total_confidence / len(signals) if signals else 0
        instrument = ctx.get_instrument(symbol)

        # Size the position
        qty = self._compute_size(
            signals, instrument, capital, avg_confidence
        )

        if qty <= Decimal("0"):
            return None

        return TargetPosition(
            strategy_id=signals[0].strategy_id,  # Primary strategy
            symbol=symbol,
            target_qty=qty,
            side=side,
            reason=" | ".join(reasons),
            urgency=min(1.0, avg_confidence),
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

        # Apply sizing multiplier (from regime policy)
        qty = Decimal(str(float(qty) * self._sizing_multiplier))

        if instrument:
            qty = instrument.round_qty(qty)

        return qty

    def set_sizing_multiplier(self, mult: float) -> None:
        """Update sizing multiplier (called by regime policy)."""
        self._sizing_multiplier = max(0.0, min(2.0, mult))
