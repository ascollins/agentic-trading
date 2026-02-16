"""Backtest engine: event replay and orchestration.

Replays historical candles through the strategy pipeline with
simulated execution, producing deterministic results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np

from agentic_trading.core.clock import SimClock
from agentic_trading.core.enums import Exchange, Mode, OrderType, Side, Timeframe
from agentic_trading.core.events import (
    CandleEvent,
    FeatureVector,
    OrderIntent,
    Signal,
)
from agentic_trading.core.interfaces import PortfolioState, TradingContext
from agentic_trading.core.models import Balance, Candle, Instrument, Position
from agentic_trading.event_bus.memory_bus import MemoryEventBus

from .fee_model import FeeModel, FundingModel
from .fill_simulator import FillSimulator
from .results import BacktestResult, compute_metrics
from .slippage import create_slippage_model

logger = logging.getLogger(__name__)


@dataclass
class _SimPosition:
    """Tracks a single simulated position in the backtest."""

    symbol: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    stop_price: float = 0.0  # 0 = no stop
    strategy_id: str = ""
    entry_time: datetime | None = None


class BacktestEngine:
    """Event-driven backtesting engine.

    Replays candles through: features → strategy → risk → execution.
    Uses SimClock for deterministic time, seeded RNG for reproducibility.
    """

    def __init__(
        self,
        strategies: list[Any],
        feature_engine: Any,
        risk_checker: Any | None = None,
        instruments: dict[str, Instrument] | None = None,
        initial_capital: float = 100_000.0,
        slippage_model: str = "volatility_based",
        slippage_bps: float = 5.0,
        fee_maker: float = 0.0002,
        fee_taker: float = 0.0004,
        funding_enabled: bool = True,
        partial_fills: bool = True,
        latency_ms: int = 50,
        seed: int = 42,
    ) -> None:
        self._strategies = strategies
        self._feature_engine = feature_engine
        self._risk_checker = risk_checker
        self._instruments = instruments or {}
        self._initial_capital = initial_capital
        self._seed = seed

        # Simulation components
        self._clock = SimClock()
        self._event_bus = MemoryEventBus()
        self._slippage = create_slippage_model(slippage_model, seed=seed, bps=slippage_bps)
        self._fee_model = FeeModel(maker_fee=fee_maker, taker_fee=fee_taker)
        self._funding_model = FundingModel() if funding_enabled else None
        self._fill_sim = FillSimulator(
            slippage_model=self._slippage,
            fee_model=self._fee_model,
            partial_fills=partial_fills,
            latency_ms=latency_ms,
            seed=seed,
        )

        # State
        self._equity = initial_capital
        self._cash = initial_capital
        self._sim_positions: dict[str, _SimPosition] = {}  # symbol → position
        self._last_price: dict[str, float] = {}  # symbol → last known close
        self._equity_curve: list[float] = [initial_capital]
        self._trade_returns: list[float] = []
        self._total_fees = 0.0
        self._total_funding = 0.0
        self._event_log: list[dict[str, Any]] = []

    async def run(
        self,
        candles_by_symbol: dict[str, list[Candle]],
    ) -> BacktestResult:
        """Run the backtest on historical candle data.

        Args:
            candles_by_symbol: {symbol: [candles sorted by time]}

        Returns:
            BacktestResult with all metrics.
        """
        await self._event_bus.start()

        # Merge all candles into chronological order
        all_events: list[tuple[datetime, str, Candle]] = []
        for symbol, candles in candles_by_symbol.items():
            for candle in candles:
                all_events.append((candle.timestamp, symbol, candle))

        all_events.sort(key=lambda x: x[0])

        if not all_events:
            return BacktestResult()

        logger.info(
            "Starting backtest: %d candles, %d symbols, capital=%.2f",
            len(all_events),
            len(candles_by_symbol),
            self._initial_capital,
        )

        funding_period = 0

        for ts, symbol, candle in all_events:
            self._clock.set_time(ts)
            self._last_price[symbol] = candle.close

            # 0. Check stop-losses before anything else
            pos = self._sim_positions.get(symbol)
            if pos is not None and pos.stop_price > 0:
                stopped = False
                if pos.side == "long" and candle.low <= pos.stop_price:
                    stopped = True
                elif pos.side == "short" and candle.high >= pos.stop_price:
                    stopped = True
                if stopped:
                    self._close_position_at_stop(pos, candle)
                    # Notify strategies that position was stopped out
                    for strategy in self._strategies:
                        if hasattr(strategy, "_record_exit"):
                            strategy._record_exit(symbol)

            # 1. Compute features (with aliasing for strategy compat)
            features = self._compute_features(symbol, candle)
            f = features.features
            if "adx_14" in f and "adx" not in f:
                f["adx"] = f["adx_14"]
            if "atr_14" in f and "atr" not in f:
                f["atr"] = f["atr_14"]

            # 2. Build context
            ctx = TradingContext(
                clock=self._clock,
                event_bus=self._event_bus,
                instruments=self._instruments,
                portfolio_state=self._get_portfolio_state(),
            )

            # 3. Run strategies
            for strategy in self._strategies:
                signal = strategy.on_candle(ctx, candle, features)
                if signal is not None:
                    self._process_signal(signal, candle)

            # 4. Apply funding (every 8h equivalent)
            if self._funding_model:
                new_period = self._funding_model.hours_to_periods(
                    (ts - all_events[0][0]).total_seconds() / 3600
                )
                if new_period > funding_period:
                    self._apply_funding(symbol, candle.close, new_period)
                    funding_period = new_period

            # 5. Update equity
            self._update_equity(candle)

            # 6. Log event
            self._event_log.append({
                "timestamp": ts.isoformat(),
                "symbol": symbol,
                "close": candle.close,
                "equity": self._equity,
            })

        await self._event_bus.stop()

        # Compute results
        result = compute_metrics(
            equity_curve=self._equity_curve,
            trade_returns=self._trade_returns,
            fees=self._total_fees,
            funding=self._total_funding,
        )
        result.strategy_id = (
            self._strategies[0].strategy_id if self._strategies else ""
        )
        result.deterministic_hash = self._compute_hash()

        logger.info("Backtest complete: %s", result.summary())
        return result

    def _compute_features(self, symbol: str, candle: Candle) -> FeatureVector:
        """Compute features for a candle using accumulated buffer."""
        if self._feature_engine:
            self._feature_engine.add_candle(candle)
            buf = self._feature_engine.get_buffer(symbol, candle.timeframe)
            return self._feature_engine.compute_features(symbol, candle.timeframe, buf)
        # Fallback: empty features
        return FeatureVector(symbol=symbol, timeframe=candle.timeframe, features={})

    def _process_signal(self, signal: Signal, candle: Candle) -> None:
        """Process a strategy signal with position-aware logic.

        Rules:
        - FLAT signal → close existing position
        - Same direction as existing position → skip (already positioned)
        - Opposite direction → close existing, then open new
        - No existing position → open new
        """
        symbol = signal.symbol
        direction = signal.direction.value  # "long", "short", or "flat"
        existing = self._sim_positions.get(symbol)

        if direction == "flat":
            if existing:
                self._close_position(existing, candle)
            return

        # Already positioned in the same direction → skip
        if existing and existing.side == direction:
            return

        # Opposite direction → close first
        if existing and existing.side != direction:
            self._close_position(existing, candle)

        # Open new position
        self._open_position(signal, candle, direction)

    def _open_position(self, signal: Signal, candle: Candle, direction: str) -> None:
        """Open a new position via simulated fill."""
        qty = self._compute_qty(signal, candle)
        if qty <= 0:
            return

        side = Side.BUY if direction == "long" else Side.SELL

        intent = OrderIntent(
            dedupe_key=f"{signal.strategy_id}:{signal.symbol}:{signal.event_id}",
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            exchange=Exchange.BINANCE,
            side=side,
            order_type=OrderType.MARKET,
            qty=Decimal(str(qty)),
        )

        fills = self._fill_sim.simulate_fill(intent, candle, self._clock.now())

        # Compute stop price from risk constraints
        rc = signal.risk_constraints
        stop_distance = rc.get("stop_distance_atr", rc.get("stop_distance", 0))

        for fill in fills:
            fill_price = float(fill.price)
            fill_qty = float(fill.qty)
            fee = float(fill.fee)

            self._total_fees += fee
            self._cash -= fee

            # Cash accounting: buying costs cash, selling (short) adds cash
            if side == Side.BUY:
                self._cash -= fill_price * fill_qty
            else:
                self._cash += fill_price * fill_qty

            # Compute stop price relative to fill price
            if stop_distance > 0:
                if direction == "long":
                    stop_price = fill_price - stop_distance
                else:
                    stop_price = fill_price + stop_distance
            else:
                stop_price = 0.0  # No stop

            self._sim_positions[signal.symbol] = _SimPosition(
                symbol=signal.symbol,
                side=direction,
                qty=fill_qty,
                entry_price=fill_price,
                stop_price=stop_price,
                strategy_id=signal.strategy_id,
                entry_time=self._clock.now(),
            )

    def _close_position(self, pos: _SimPosition, candle: Candle) -> None:
        """Close an existing position and record trade return."""
        close_side = Side.SELL if pos.side == "long" else Side.BUY

        intent = OrderIntent(
            dedupe_key=f"close:{pos.symbol}:{uuid.uuid4()}",
            strategy_id="close",
            symbol=pos.symbol,
            exchange=Exchange.BINANCE,
            side=close_side,
            order_type=OrderType.MARKET,
            qty=Decimal(str(pos.qty)),
        )

        fills = self._fill_sim.simulate_fill(intent, candle, self._clock.now())

        for fill in fills:
            fill_price = float(fill.price)
            fill_qty = float(fill.qty)
            fee = float(fill.fee)

            self._total_fees += fee
            self._cash -= fee

            # Cash accounting
            if close_side == Side.SELL:
                self._cash += fill_price * fill_qty
            else:
                self._cash -= fill_price * fill_qty

            # Record trade PnL
            if pos.entry_price > 0:
                if pos.side == "long":
                    trade_ret = (fill_price - pos.entry_price) / pos.entry_price
                else:
                    trade_ret = (pos.entry_price - fill_price) / pos.entry_price
                self._trade_returns.append(trade_ret)

        # Remove position
        self._sim_positions.pop(pos.symbol, None)

    def _close_position_at_stop(self, pos: _SimPosition, candle: Candle) -> None:
        """Close a position at its stop price (not candle close).

        Uses the stop price for fill calculation to simulate realistic
        stop-loss execution.  Fees are still applied via the fee model.
        """
        stop_price = pos.stop_price
        fee = float(self._fee_model.compute_fee(
            price=stop_price,
            qty=pos.qty,
            is_maker=False,  # Stop-loss fills as taker
        ))

        self._total_fees += fee
        self._cash -= fee

        # Cash accounting at the stop price
        if pos.side == "long":
            self._cash += stop_price * pos.qty  # Sell to close
        else:
            self._cash -= stop_price * pos.qty  # Buy to close

        # Record trade PnL
        if pos.entry_price > 0:
            if pos.side == "long":
                trade_ret = (stop_price - pos.entry_price) / pos.entry_price
            else:
                trade_ret = (pos.entry_price - stop_price) / pos.entry_price
            self._trade_returns.append(trade_ret)

        logger.debug(
            "Stop-loss hit: %s %s @ %.2f (entry %.2f, stop %.2f)",
            pos.side, pos.symbol, stop_price, pos.entry_price, stop_price,
        )

        # Remove position
        self._sim_positions.pop(pos.symbol, None)

    def _compute_qty(self, signal: Signal, candle: Candle) -> float:
        """Compute order quantity with cash and leverage constraints.

        Position sizing uses fixed-fractional risk:
        - Risk 1% of equity per trade
        - Stop distance from signal or fallback to ATR * 2
        - Max notional capped at 10% of equity (max ~10x leverage across portfolio)
        """
        rc = signal.risk_constraints
        atr = rc.get("atr", candle.high - candle.low)
        price = candle.close

        if price <= 0 or self._equity <= 0:
            return 0

        # Risk 0.5% of equity per trade — conservative for 1m candle strategies
        risk_amount = self._equity * 0.005

        # Use the stop distance from signal if provided, else fallback
        stop_distance = rc.get("stop_distance_atr", rc.get("stop_distance", 0))
        if stop_distance <= 0:
            stop_distance = atr * 2.0 if atr > 0 else price * 0.02

        qty = risk_amount / stop_distance

        # Scale by confidence
        qty *= signal.confidence

        # Max position size: 5% of equity in notional value
        max_notional = self._equity * 0.05
        max_qty_by_notional = max_notional / price
        qty = min(qty, max_qty_by_notional)

        # Cash constraint: can't spend more than available cash
        max_qty_by_cash = max(0, self._cash * 0.95) / price  # Keep 5% cash buffer
        qty = min(qty, max_qty_by_cash)

        return qty

    def _apply_funding(self, symbol: str, mark_price: float, period: int) -> None:
        """Apply funding payment if position exists."""
        if not self._funding_model:
            return
        pos = self._sim_positions.get(symbol)
        if pos:
            signed_qty = pos.qty if pos.side == "long" else -pos.qty
            payment = self._funding_model.compute_funding_payment(
                symbol, signed_qty, mark_price, period
            )
            self._cash += float(payment)
            self._total_funding += float(payment)

    def _update_equity(self, candle: Candle) -> None:
        """Update equity curve with current mark-to-market.

        Uses the current candle's close for its symbol, and last known
        prices for other symbols, so multi-symbol positions are valued
        correctly.
        """
        # Update last known price for this candle's symbol
        self._last_price[candle.symbol] = candle.close

        unrealized = 0.0
        for symbol, pos in self._sim_positions.items():
            mark_price = self._last_price.get(symbol, pos.entry_price)
            if pos.side == "long":
                unrealized += pos.qty * (mark_price - pos.entry_price)
            else:
                unrealized += pos.qty * (pos.entry_price - mark_price)

        self._equity = self._cash + unrealized
        self._equity_curve.append(self._equity)

    def _get_portfolio_state(self) -> PortfolioState:
        """Build portfolio state from current positions."""
        positions = {}
        for symbol, pos in self._sim_positions.items():
            from agentic_trading.core.enums import MarginMode, PositionSide

            positions[symbol] = Position(
                symbol=symbol,
                exchange=Exchange.BINANCE,
                side=PositionSide.LONG if pos.side == "long" else PositionSide.SHORT,
                qty=Decimal(str(abs(pos.qty))),
                entry_price=Decimal(str(pos.entry_price)),
            )

        balances = {
            "USDT": Balance(
                currency="USDT",
                exchange=Exchange.BINANCE,
                total=Decimal(str(self._cash)),
                free=Decimal(str(self._cash)),
            )
        }

        return PortfolioState(positions=positions, balances=balances)

    def _compute_hash(self) -> str:
        """Compute deterministic hash of the backtest results."""
        data = json.dumps(
            {
                "equity_curve": [round(e, 6) for e in self._equity_curve],
                "trade_returns": [round(r, 8) for r in self._trade_returns],
                "total_fees": round(self._total_fees, 6),
                "seed": self._seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @property
    def event_log(self) -> list[dict[str, Any]]:
        """Return event log for replay/debugging."""
        return list(self._event_log)
