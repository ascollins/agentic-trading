"""Backtest engine: event replay and orchestration.

Replays historical candles through the strategy pipeline with
simulated execution, producing deterministic results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
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
        self._positions: dict[str, dict] = {}  # symbol → {qty, entry_price, side}
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
        """Process a strategy signal: check risk, simulate fill."""
        if signal.direction.value == "flat":
            return

        # Create order intent
        intent = OrderIntent(
            dedupe_key=f"{signal.strategy_id}:{signal.symbol}:{signal.event_id}",
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            exchange=Exchange.BINANCE,  # Backtest default
            side=Side.BUY if signal.direction.value == "long" else Side.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal(str(self._compute_qty(signal, candle))),
        )

        if float(intent.qty) <= 0:
            return

        # Simulate fill
        fills = self._fill_sim.simulate_fill(
            intent, candle, self._clock.now()
        )

        for fill in fills:
            self._apply_fill(fill, signal)

    def _compute_qty(self, signal: Signal, candle: Candle) -> float:
        """Compute order quantity from signal."""
        rc = signal.risk_constraints
        atr = rc.get("atr", candle.high - candle.low)
        price = candle.close

        if price <= 0:
            return 0

        # Risk 2% of equity per trade, sized by ATR
        risk_amount = self._equity * 0.02
        stop_distance = atr * 2.0 if atr > 0 else price * 0.02
        qty = risk_amount / stop_distance

        # Scale by confidence
        qty *= signal.confidence

        return qty

    def _apply_fill(self, fill: Any, signal: Signal) -> None:
        """Apply a fill to the portfolio state."""
        symbol = fill.symbol
        price = float(fill.price)
        qty = float(fill.qty)
        fee = float(fill.fee)
        is_buy = fill.side == Side.BUY

        self._total_fees += fee
        self._cash -= fee

        if symbol not in self._positions:
            self._positions[symbol] = {"qty": 0.0, "entry_price": 0.0, "cost_basis": 0.0}

        pos = self._positions[symbol]

        if is_buy:
            # Add to position
            new_qty = pos["qty"] + qty
            if new_qty != 0:
                pos["entry_price"] = (
                    pos["entry_price"] * pos["qty"] + price * qty
                ) / new_qty
            pos["qty"] = new_qty
            self._cash -= price * qty
        else:
            # Reduce or reverse position
            if pos["qty"] > 0:
                # Closing long
                pnl = (price - pos["entry_price"]) * min(qty, pos["qty"])
                self._trade_returns.append(pnl / (pos["entry_price"] * min(qty, pos["qty"])) if pos["entry_price"] > 0 else 0)
                self._cash += price * qty
            pos["qty"] -= qty
            if pos["qty"] < 0:
                pos["entry_price"] = price
            elif pos["qty"] == 0:
                pos["entry_price"] = 0.0

    def _apply_funding(self, symbol: str, mark_price: float, period: int) -> None:
        """Apply funding payment if position exists."""
        if not self._funding_model:
            return
        pos = self._positions.get(symbol)
        if pos and pos["qty"] != 0:
            payment = self._funding_model.compute_funding_payment(
                symbol, pos["qty"], mark_price, period
            )
            self._cash += float(payment)
            self._total_funding += float(payment)

    def _update_equity(self, candle: Candle) -> None:
        """Update equity curve with current mark-to-market."""
        unrealized = 0.0
        for symbol, pos in self._positions.items():
            if pos["qty"] != 0:
                # Use latest candle close as mark price
                unrealized += pos["qty"] * (candle.close - pos["entry_price"])

        self._equity = self._cash + unrealized
        self._equity_curve.append(self._equity)

    def _get_portfolio_state(self) -> PortfolioState:
        """Build portfolio state from current positions."""
        positions = {}
        for symbol, pos in self._positions.items():
            if pos["qty"] != 0:
                from agentic_trading.core.enums import MarginMode, PositionSide

                positions[symbol] = Position(
                    symbol=symbol,
                    exchange=Exchange.BINANCE,
                    side=PositionSide.LONG if pos["qty"] > 0 else PositionSide.SHORT,
                    qty=Decimal(str(abs(pos["qty"]))),
                    entry_price=Decimal(str(pos["entry_price"])),
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
