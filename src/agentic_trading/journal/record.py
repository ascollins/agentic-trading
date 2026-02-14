"""Consolidated trade record — the core data model.

A TradeRecord captures the full lifecycle of a single trade from
signal generation through entry, management, and exit.  Every fill,
mark-to-market sample, and governance decision is attached.

Inspired by Edgewonk's unified trade view that groups partial fills
and scale-in/scale-out legs into a single record with computed
analytics (R-multiple, MAE, MFE, management efficiency).
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal


class TradePhase(str, enum.Enum):
    """Lifecycle phase of a trade record."""

    PENDING = "pending"        # Signal generated, order not yet filled
    OPEN = "open"              # Position active (at least one fill received)
    CLOSED = "closed"          # Position fully exited
    CANCELLED = "cancelled"    # Order rejected or cancelled before fill


class TradeOutcome(str, enum.Enum):
    """Win / loss / break-even classification."""

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class FillLeg:
    """A single fill contributing to the trade."""

    fill_id: str
    order_id: str
    side: str          # "buy" or "sell"
    price: Decimal
    qty: Decimal
    fee: Decimal
    fee_currency: str
    is_maker: bool
    timestamp: datetime


@dataclass
class MarkSample:
    """Point-in-time mark-to-market sample for MAE / MFE computation."""

    timestamp: datetime
    mark_price: Decimal
    unrealized_pnl: Decimal


@dataclass
class TradeRecord:
    """Consolidated record for one logical trade.

    Groups all entry fills, exit fills, and intermediate mark-to-market
    samples into a single unit.  Computes derived analytics on demand.

    Parameters
    ----------
    trade_id : str
        Unique identifier (UUID by default).
    trace_id : str
        Correlation ID linking back to the originating event chain.
    strategy_id : str
        Which strategy originated the signal.
    symbol : str
        Unified market symbol (e.g. ``"BTC/USDT"``).
    direction : str
        ``"long"`` or ``"short"``.
    """

    # Identity
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    strategy_id: str = ""
    symbol: str = ""
    exchange: str = ""
    direction: str = ""  # "long" or "short"

    # Signal context (captured at entry)
    signal_confidence: float = 0.0
    signal_rationale: str = ""
    signal_features: dict = field(default_factory=dict)
    signal_timestamp: datetime | None = None

    # Risk context
    initial_risk_price: Decimal | None = None   # Stop-loss price from signal
    initial_risk_amount: Decimal | None = None  # Position risk in quote currency
    planned_target_price: Decimal | None = None  # Take-profit target

    # Lifecycle
    phase: TradePhase = TradePhase.PENDING
    entry_fills: list[FillLeg] = field(default_factory=list)
    exit_fills: list[FillLeg] = field(default_factory=list)
    mark_samples: list[MarkSample] = field(default_factory=list)

    # Timing
    opened_at: datetime | None = None
    closed_at: datetime | None = None

    # Governance context
    maturity_level: str = ""
    health_score_at_entry: float = 1.0
    governance_sizing_multiplier: float = 1.0

    # Tags and metadata
    tags: list[str] = field(default_factory=list)
    mistakes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Computed properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    def entry_qty(self) -> Decimal:
        """Total quantity entered."""
        return sum((f.qty for f in self.entry_fills), Decimal("0"))

    @property
    def exit_qty(self) -> Decimal:
        """Total quantity exited."""
        return sum((f.qty for f in self.exit_fills), Decimal("0"))

    @property
    def remaining_qty(self) -> Decimal:
        """Quantity still open."""
        return self.entry_qty - self.exit_qty

    @property
    def avg_entry_price(self) -> Decimal:
        """Volume-weighted average entry price."""
        if not self.entry_fills:
            return Decimal("0")
        total_cost = sum(f.price * f.qty for f in self.entry_fills)
        total_qty = self.entry_qty
        if total_qty == 0:
            return Decimal("0")
        return total_cost / total_qty

    @property
    def avg_exit_price(self) -> Decimal:
        """Volume-weighted average exit price."""
        if not self.exit_fills:
            return Decimal("0")
        total_cost = sum(f.price * f.qty for f in self.exit_fills)
        total_qty = self.exit_qty
        if total_qty == 0:
            return Decimal("0")
        return total_cost / total_qty

    @property
    def total_fees(self) -> Decimal:
        """Total fees across all fills."""
        entry_fees = sum((f.fee for f in self.entry_fills), Decimal("0"))
        exit_fees = sum((f.fee for f in self.exit_fills), Decimal("0"))
        return entry_fees + exit_fees

    @property
    def gross_pnl(self) -> Decimal:
        """Gross P&L before fees (for closed/partially-closed trades)."""
        if not self.exit_fills:
            return Decimal("0")
        if self.direction == "long":
            return (self.avg_exit_price - self.avg_entry_price) * self.exit_qty
        else:
            return (self.avg_entry_price - self.avg_exit_price) * self.exit_qty

    @property
    def net_pnl(self) -> Decimal:
        """Net P&L after all fees."""
        return self.gross_pnl - self.total_fees

    @property
    def net_pnl_pct(self) -> float:
        """Net P&L as a percentage of entry notional."""
        entry_notional = self.avg_entry_price * self.entry_qty
        if entry_notional == 0:
            return 0.0
        return float(self.net_pnl / entry_notional)

    @property
    def outcome(self) -> TradeOutcome:
        """Win / loss / break-even classification."""
        pnl = self.net_pnl
        if pnl > Decimal("0"):
            return TradeOutcome.WIN
        if pnl < Decimal("0"):
            return TradeOutcome.LOSS
        return TradeOutcome.BREAKEVEN

    @property
    def hold_duration_seconds(self) -> float:
        """Seconds between first entry and last exit (or now)."""
        if self.opened_at is None:
            return 0.0
        end = self.closed_at or datetime.now(timezone.utc)
        return (end - self.opened_at).total_seconds()

    # ------------------------------------------------------------------ #
    # R-Multiple                                                           #
    # ------------------------------------------------------------------ #

    @property
    def r_multiple(self) -> float:
        """Net P&L expressed as a multiple of initial risk (1R).

        Returns 0.0 if initial risk is not set or zero.
        """
        if self.initial_risk_amount is None or self.initial_risk_amount == 0:
            return 0.0
        return float(self.net_pnl / abs(self.initial_risk_amount))

    def compute_initial_risk(self) -> Decimal:
        """Derive initial risk from stop price and entry, if available.

        Sets ``initial_risk_amount`` as a side effect.
        """
        if self.initial_risk_price is None or not self.entry_fills:
            return Decimal("0")
        entry = self.avg_entry_price
        stop = self.initial_risk_price
        if self.direction == "long":
            risk_per_unit = entry - stop
        else:
            risk_per_unit = stop - entry
        risk_amount = abs(risk_per_unit * self.entry_qty)
        self.initial_risk_amount = risk_amount
        return risk_amount

    # ------------------------------------------------------------------ #
    # MAE / MFE (Maximum Adverse / Favorable Excursion)                    #
    # ------------------------------------------------------------------ #

    @property
    def mae(self) -> Decimal:
        """Maximum Adverse Excursion — worst unrealized P&L during the trade.

        Negative value = how far price went against the position.
        """
        if not self.mark_samples:
            return Decimal("0")
        return min(s.unrealized_pnl for s in self.mark_samples)

    @property
    def mfe(self) -> Decimal:
        """Maximum Favorable Excursion — best unrealized P&L during the trade.

        Positive value = how far price went in favour of the position.
        """
        if not self.mark_samples:
            return Decimal("0")
        return max(s.unrealized_pnl for s in self.mark_samples)

    @property
    def mae_price(self) -> Decimal:
        """Price at the point of maximum adverse excursion."""
        if not self.mark_samples:
            return Decimal("0")
        worst = min(self.mark_samples, key=lambda s: s.unrealized_pnl)
        return worst.mark_price

    @property
    def mfe_price(self) -> Decimal:
        """Price at the point of maximum favorable excursion."""
        if not self.mark_samples:
            return Decimal("0")
        best = max(self.mark_samples, key=lambda s: s.unrealized_pnl)
        return best.mark_price

    @property
    def management_efficiency(self) -> float:
        """How much of the available profit was captured (actual / MFE).

        Range [0.0, 1.0+].  Values < 1.0 mean profits were left on
        the table.  Values > 1.0 are impossible for closed trades.
        """
        mfe_val = float(self.mfe)
        if mfe_val <= 0:
            return 0.0
        actual = float(self.net_pnl)
        if actual <= 0:
            return 0.0
        return actual / mfe_val

    @property
    def mae_r(self) -> float:
        """MAE expressed as an R-multiple."""
        if self.initial_risk_amount is None or self.initial_risk_amount == 0:
            return 0.0
        return float(self.mae / abs(self.initial_risk_amount))

    @property
    def mfe_r(self) -> float:
        """MFE expressed as an R-multiple."""
        if self.initial_risk_amount is None or self.initial_risk_amount == 0:
            return 0.0
        return float(self.mfe / abs(self.initial_risk_amount))

    # ------------------------------------------------------------------ #
    # Planned vs Actual                                                    #
    # ------------------------------------------------------------------ #

    @property
    def planned_rr_ratio(self) -> float:
        """Planned risk-reward ratio from signal constraints."""
        if (
            self.initial_risk_price is None
            or self.planned_target_price is None
            or not self.entry_fills
        ):
            return 0.0
        entry = float(self.avg_entry_price)
        stop = float(self.initial_risk_price)
        target = float(self.planned_target_price)
        risk = abs(entry - stop)
        if risk == 0:
            return 0.0
        reward = abs(target - entry)
        return reward / risk

    @property
    def actual_rr_ratio(self) -> float:
        """Actual risk-reward ratio realised."""
        if self.initial_risk_amount is None or self.initial_risk_amount == 0:
            return 0.0
        return abs(self.r_multiple)

    # ------------------------------------------------------------------ #
    # Mutation helpers                                                     #
    # ------------------------------------------------------------------ #

    def add_entry_fill(self, fill: FillLeg) -> None:
        """Add a fill to the entry side and transition to OPEN."""
        self.entry_fills.append(fill)
        if self.phase == TradePhase.PENDING:
            self.phase = TradePhase.OPEN
            self.opened_at = fill.timestamp

    def add_exit_fill(self, fill: FillLeg) -> None:
        """Add a fill to the exit side.  Close if fully exited."""
        self.exit_fills.append(fill)
        if self.remaining_qty <= Decimal("0"):
            self.phase = TradePhase.CLOSED
            self.closed_at = fill.timestamp

    def add_mark_sample(self, sample: MarkSample) -> None:
        """Record a mark-to-market observation."""
        self.mark_samples.append(sample)

    def cancel(self) -> None:
        """Mark the trade as cancelled (no fills received)."""
        if self.phase == TradePhase.PENDING:
            self.phase = TradePhase.CANCELLED

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Export to a flat dictionary for logging / storage."""
        return {
            "trade_id": self.trade_id,
            "trace_id": self.trace_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "direction": self.direction,
            "phase": self.phase.value,
            "outcome": self.outcome.value if self.phase == TradePhase.CLOSED else None,
            "signal_confidence": self.signal_confidence,
            "signal_rationale": self.signal_rationale,
            "avg_entry_price": str(self.avg_entry_price),
            "avg_exit_price": str(self.avg_exit_price),
            "entry_qty": str(self.entry_qty),
            "exit_qty": str(self.exit_qty),
            "gross_pnl": str(self.gross_pnl),
            "net_pnl": str(self.net_pnl),
            "net_pnl_pct": self.net_pnl_pct,
            "total_fees": str(self.total_fees),
            "r_multiple": self.r_multiple,
            "mae": str(self.mae),
            "mfe": str(self.mfe),
            "mae_r": self.mae_r,
            "mfe_r": self.mfe_r,
            "management_efficiency": self.management_efficiency,
            "planned_rr": self.planned_rr_ratio,
            "actual_rr": self.actual_rr_ratio,
            "hold_duration_s": self.hold_duration_seconds,
            "entry_fills": len(self.entry_fills),
            "exit_fills": len(self.exit_fills),
            "mark_samples": len(self.mark_samples),
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "maturity_level": self.maturity_level,
            "health_score_at_entry": self.health_score_at_entry,
            "governance_sizing_multiplier": self.governance_sizing_multiplier,
            "tags": self.tags,
            "mistakes": self.mistakes,
        }
