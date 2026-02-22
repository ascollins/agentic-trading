"""Protocol interfaces for the trading platform.

All module boundaries are defined here as Protocol classes.
Implementations can be swapped (live/paper/backtest) without changing callers.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable

from .clock import IClock
from .enums import AgentType, Side
from .events import (
    AgentCapabilities,
    AgentHealthReport,
    BaseEvent,
    FeatureVector,
    OrderAck,
    OrderIntent,
    RiskCheckResult,
    Signal,
    RegimeState,
)
from .models import Balance, Candle, Fill, Instrument, Order, Position


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

@runtime_checkable
class IEventBus(Protocol):
    """Publish/subscribe event bus."""

    async def publish(self, topic: str, event: BaseEvent) -> None: ...

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: Callable[[BaseEvent], Coroutine[Any, Any, None]],
    ) -> None: ...

    async def start(self) -> None: ...
    async def stop(self) -> None: ...


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@runtime_checkable
class IAgent(Protocol):
    """Autonomous agent with lifecycle management.

    Agents are the primary operators in the institutional architecture.
    Each agent has a unique identity, a type, lifecycle (start/stop),
    health reporting, and capability declarations.
    """

    @property
    def agent_id(self) -> str: ...

    @property
    def agent_type(self) -> AgentType: ...

    @property
    def is_running(self) -> bool: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def health_check(self) -> AgentHealthReport: ...

    def capabilities(self) -> AgentCapabilities: ...


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@runtime_checkable
class IStrategy(Protocol):
    """Trading strategy interface.

    Strategy code must NOT know if it's backtest/paper/live.
    It receives data via TradingContext and returns Signals.
    """

    @property
    def strategy_id(self) -> str: ...

    def on_candle(
        self,
        ctx: "TradingContext",
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        """Process a new candle + features. Return a signal or None."""
        ...

    def on_regime_change(self, regime: RegimeState) -> None:
        """Notify strategy of a regime change."""
        ...

    def get_parameters(self) -> dict[str, Any]:
        """Return current strategy parameters (for logging/audit)."""
        ...


# ---------------------------------------------------------------------------
# Exchange Adapter
# ---------------------------------------------------------------------------

@runtime_checkable
class IExchangeAdapter(Protocol):
    """Unified exchange interface. Implemented by CCXT, paper, and backtest adapters.

    Core methods (required by all adapters):
        submit_order, cancel_order, cancel_all_orders, get_open_orders,
        get_positions, get_balances, get_instrument, get_funding_rate

    V5-enhanced methods (optional, raise NotImplementedError if unsupported):
        amend_order, batch_submit_orders, set_leverage, set_position_mode,
        set_trading_stop, get_closed_pnl
    """

    async def submit_order(self, intent: OrderIntent) -> OrderAck: ...

    async def cancel_order(self, order_id: str, symbol: str) -> OrderAck: ...

    async def cancel_all_orders(self, symbol: str | None = None) -> list[OrderAck]: ...

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]: ...

    async def get_positions(self, symbol: str | None = None) -> list[Position]: ...

    async def get_balances(self) -> list[Balance]: ...

    async def get_instrument(self, symbol: str) -> Instrument: ...

    async def get_funding_rate(self, symbol: str) -> Decimal: ...

    # ---- V5-enhanced methods (Bybit V5 / modern exchange capabilities) ----

    async def amend_order(
        self,
        order_id: str,
        symbol: str,
        *,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> OrderAck: ...

    async def batch_submit_orders(
        self, intents: list[OrderIntent]
    ) -> list[OrderAck]: ...

    async def set_leverage(
        self, symbol: str, leverage: int
    ) -> dict[str, Any]: ...

    async def set_position_mode(
        self, symbol: str, mode: str
    ) -> dict[str, Any]: ...

    async def set_trading_stop(
        self,
        symbol: str,
        *,
        take_profit: Decimal | None = None,
        stop_loss: Decimal | None = None,
        trailing_stop: Decimal | None = None,
        active_price: Decimal | None = None,
    ) -> dict[str, Any]: ...

    async def get_closed_pnl(
        self, symbol: str, *, limit: int = 50
    ) -> list[dict[str, Any]]: ...

    # ---- FX-specific methods (optional, raise NotImplementedError) ----

    async def get_rollover_rates(self, symbol: str) -> dict[str, Decimal]: ...

    async def get_spread(self, symbol: str) -> dict[str, Decimal]: ...


# ---------------------------------------------------------------------------
# Risk Checker
# ---------------------------------------------------------------------------

@runtime_checkable
class IRiskChecker(Protocol):
    """Pre-trade and post-trade risk checks."""

    def pre_trade_check(
        self, intent: OrderIntent, portfolio_state: "PortfolioState"
    ) -> RiskCheckResult: ...

    def post_trade_check(
        self, fill: Fill, portfolio_state: "PortfolioState"
    ) -> RiskCheckResult: ...


# ---------------------------------------------------------------------------
# Portfolio State (read-only snapshot)
# ---------------------------------------------------------------------------

class PortfolioState:
    """Immutable snapshot of current portfolio for risk checks and strategy context."""

    def __init__(
        self,
        positions: dict[str, Position] | None = None,
        balances: dict[str, Balance] | None = None,
        open_orders: list[Order] | None = None,
    ) -> None:
        self.positions = positions or {}
        self.balances = balances or {}
        self.open_orders = open_orders or []

    @property
    def gross_exposure(self) -> Decimal:
        return sum(
            abs(p.notional) for p in self.positions.values()
        )

    @property
    def net_exposure(self) -> Decimal:
        total = Decimal("0")
        for p in self.positions.values():
            if p.side.value == "long":
                total += p.notional
            else:
                total -= p.notional
        return total

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def get_balance(self, currency: str) -> Balance | None:
        return self.balances.get(currency)


# ---------------------------------------------------------------------------
# Trading Context (passed to strategies)
# ---------------------------------------------------------------------------

class TradingContext:
    """Mode-agnostic context passed to strategies.

    Strategies interact ONLY through this interface.
    They never import mode-specific modules.
    """

    def __init__(
        self,
        clock: IClock,
        event_bus: IEventBus,
        instruments: dict[str, Instrument],
        regime: RegimeState | None = None,
        portfolio_state: PortfolioState | None = None,
        risk_limits: dict[str, Any] | None = None,
        position_bounds: Any | None = None,
    ) -> None:
        self.clock = clock
        self.event_bus = event_bus
        self.instruments = instruments
        self.regime = regime or RegimeState(symbol="*")
        self.portfolio_state = portfolio_state or PortfolioState()
        self.risk_limits = risk_limits or {}
        self.position_bounds = position_bounds

    def get_instrument(self, symbol: str) -> Instrument | None:
        return self.instruments.get(symbol)
