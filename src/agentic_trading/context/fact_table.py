"""Fact Table — real-time structured state for agent context.

A read-heavy, thread-safe table of current market state. Agents read
snapshots before reasoning. Event handlers update the table as events
flow through the system.

Reads return frozen copies (``model_copy()``) so agents never see
partial updates.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.ids import utc_now as _now

# ---------------------------------------------------------------------------
# Structured state models
# ---------------------------------------------------------------------------


class PriceLevels(BaseModel):
    """Latest price data for a single symbol."""

    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    funding_rate: float = 0.0
    updated_at: datetime = Field(default_factory=_now)


class RiskSnapshot(BaseModel):
    """Current risk parameters and thresholds."""

    max_portfolio_leverage: float = 3.0
    max_single_position_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    current_drawdown_pct: float = 0.0
    kill_switch_active: bool = False
    circuit_breakers_tripped: list[str] = Field(default_factory=list)
    degraded_mode: str = "normal"


class PortfolioSnapshot(BaseModel):
    """Current portfolio metrics."""

    total_equity: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    daily_pnl: float = 0.0
    open_position_count: int = 0
    positions: dict[str, dict[str, Any]] = Field(default_factory=dict)


class FactTableSnapshot(BaseModel):
    """Immutable frozen copy of the entire fact table for audit."""

    timestamp: datetime = Field(default_factory=_now)
    prices: dict[str, PriceLevels] = Field(default_factory=dict)
    risk: RiskSnapshot = Field(default_factory=RiskSnapshot)
    portfolio: PortfolioSnapshot = Field(default_factory=PortfolioSnapshot)
    regimes: dict[str, dict[str, Any]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fact Table
# ---------------------------------------------------------------------------


class FactTable:
    """Read-heavy, thread-safe structured state table.

    Uses ``threading.RLock`` for write safety.  Reads return snapshot
    copies (via ``model_copy()``) so agents never see partial updates.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._prices: dict[str, PriceLevels] = {}
        self._risk: RiskSnapshot = RiskSnapshot()
        self._portfolio: PortfolioSnapshot = PortfolioSnapshot()
        self._regimes: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Reads (return copies for isolation)
    # ------------------------------------------------------------------

    def get_price(self, symbol: str) -> PriceLevels | None:
        """Get the latest price levels for a symbol."""
        entry = self._prices.get(symbol)
        if entry is None:
            return None
        return entry.model_copy()

    def get_all_prices(self) -> dict[str, PriceLevels]:
        """Get all price levels (copies)."""
        return {s: p.model_copy() for s, p in self._prices.items()}

    def get_risk(self) -> RiskSnapshot:
        """Get current risk parameters (copy)."""
        return self._risk.model_copy()

    def get_portfolio(self) -> PortfolioSnapshot:
        """Get current portfolio metrics (copy)."""
        return self._portfolio.model_copy()

    def get_regime(self, symbol: str) -> dict[str, Any]:
        """Get regime state for a symbol (copy)."""
        regime = self._regimes.get(symbol)
        if regime is None:
            return {}
        return dict(regime)

    def snapshot(self) -> FactTableSnapshot:
        """Take a full immutable snapshot for audit or pipeline results."""
        return FactTableSnapshot(
            prices=self.get_all_prices(),
            risk=self.get_risk(),
            portfolio=self.get_portfolio(),
            regimes={s: dict(r) for s, r in self._regimes.items()},
        )

    # ------------------------------------------------------------------
    # Writes (RLock-protected)
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, **kwargs: Any) -> None:
        """Update price levels for a symbol. Creates entry if absent."""
        with self._lock:
            existing = self._prices.get(symbol)
            if existing is None:
                self._prices[symbol] = PriceLevels(symbol=symbol, **kwargs)
            else:
                self._prices[symbol] = existing.model_copy(update=kwargs)

    def update_risk(self, **kwargs: Any) -> None:
        """Partially update risk parameters."""
        with self._lock:
            self._risk = self._risk.model_copy(update=kwargs)

    def update_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Replace the portfolio snapshot."""
        with self._lock:
            self._portfolio = snapshot

    def update_portfolio_fields(self, **kwargs: Any) -> None:
        """Partially update portfolio metrics."""
        with self._lock:
            self._portfolio = self._portfolio.model_copy(update=kwargs)

    def update_regime(self, symbol: str, regime_data: dict[str, Any]) -> None:
        """Update regime state for a symbol."""
        with self._lock:
            self._regimes[symbol] = dict(regime_data)

    def clear(self) -> None:
        """Reset all state."""
        with self._lock:
            self._prices.clear()
            self._risk = RiskSnapshot()
            self._portfolio = PortfolioSnapshot()
            self._regimes.clear()
