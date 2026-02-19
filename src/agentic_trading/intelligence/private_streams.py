"""Private WebSocket stream manager for real-time account updates.

Watches order, execution, position, and wallet updates via CCXT Pro's
private WebSocket streams and publishes them to the platform event bus.

Architecture
------------
* One asyncio task per stream type per exchange.
* CCXT Pro handles WebSocket reconnection and authentication internally.
* Incoming updates are normalised into platform events (OrderUpdate,
  FillEvent, PositionUpdate, BalanceUpdate) and published to the event bus.
* Complementary to ``FeedManager`` which handles public market data streams.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import ccxt.pro as ccxtpro

from agentic_trading.core.config import ExchangeConfig
from agentic_trading.core.enums import (
    Exchange,
    OrderStatus,
    PositionSide,
    Side,
)
from agentic_trading.core.events import (
    BalanceUpdate,
    FillEvent,
    OrderUpdate,
    PositionUpdate,
)
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


# Status mapping (mirrors ccxt_adapter._CCXT_STATUS_MAP)
_STATUS_MAP: dict[str, OrderStatus] = {
    "open": OrderStatus.SUBMITTED,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}

# Map our Exchange enum to CCXT Pro exchange class constructors.
_EXCHANGE_CLASSES: dict[Exchange, type] = {
    Exchange.BINANCE: ccxtpro.binance,
    Exchange.BYBIT: ccxtpro.bybit,
}


def _build_ccxt_pro_exchange(config: ExchangeConfig) -> Any:
    """Instantiate a CCXT Pro exchange for private streams."""
    cls = _EXCHANGE_CLASSES.get(config.name)
    if cls is None:
        raise ValueError(f"Unsupported exchange for private streams: {config.name.value}")

    options: dict[str, Any] = {}

    if config.name == Exchange.BINANCE:
        options["defaultType"] = "future"
    if config.name == Exchange.BYBIT:
        options["defaultType"] = "swap"

    exchange = cls({
        "apiKey": config.api_key,
        "secret": config.secret,
        "enableRateLimit": True,
        "rateLimit": config.rate_limit,
        "timeout": config.timeout,
        "options": options,
    })

    if config.demo:
        exchange.enable_demo_trading(True)
    elif config.testnet:
        exchange.set_sandbox_mode(True)

    return exchange


class PrivateStreamManager:
    """Manages private WebSocket feeds for real-time account state.

    Streams
    -------
    * **Orders** — real-time order status updates (new, partially filled,
      filled, cancelled, amended).
    * **Trades** — execution/fill notifications.
    * **Positions** — position size, PnL, and liquidation price changes.
    * **Balance** — wallet balance updates (margin, available, equity).

    Parameters
    ----------
    event_bus:
        Platform event bus for publishing state events.
    exchange_configs:
        List of exchange configurations (must include API credentials).
    symbols:
        List of unified symbols to watch (e.g. ``["BTC/USDT:USDT"]``).
    order_topic:
        Event bus topic for order updates.
    fill_topic:
        Event bus topic for fill events.
    position_topic:
        Event bus topic for position updates.
    balance_topic:
        Event bus topic for balance updates.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        exchange_configs: list[ExchangeConfig],
        symbols: list[str],
        order_topic: str = "execution.order",
        fill_topic: str = "execution.fill",
        position_topic: str = "state.position",
        balance_topic: str = "state.balance",
    ) -> None:
        self._event_bus = event_bus
        self._exchange_configs = exchange_configs
        self._symbols = symbols
        self._order_topic = order_topic
        self._fill_topic = fill_topic
        self._position_topic = position_topic
        self._balance_topic = balance_topic

        # Runtime state
        self._exchanges: dict[Exchange, Any] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to all configured exchanges and launch private stream tasks."""
        if self._running:
            logger.warning("PrivateStreamManager.start() called while already running.")
            return

        self._running = True
        logger.info(
            "Starting PrivateStreamManager: exchanges=%s, symbols=%s",
            [c.name.value for c in self._exchange_configs],
            self._symbols,
        )

        for config in self._exchange_configs:
            if not config.api_key or not config.secret:
                logger.warning(
                    "Skipping private streams for %s: no API credentials.",
                    config.name.value,
                )
                continue

            try:
                exchange = _build_ccxt_pro_exchange(config)
                self._exchanges[config.name] = exchange
                mode_label = (
                    "demo" if config.demo
                    else ("testnet" if config.testnet else "live")
                )
                logger.info(
                    "Initialized CCXT Pro exchange for private streams: %s (mode=%s)",
                    config.name.value,
                    mode_label,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize exchange %s for private streams",
                    config.name.value,
                )
                continue

            # Launch stream tasks
            self._tasks.append(asyncio.create_task(
                self._watch_orders_loop(config.name, exchange),
                name=f"private:orders:{config.name.value}",
            ))
            self._tasks.append(asyncio.create_task(
                self._watch_trades_loop(config.name, exchange),
                name=f"private:trades:{config.name.value}",
            ))
            self._tasks.append(asyncio.create_task(
                self._watch_positions_loop(config.name, exchange),
                name=f"private:positions:{config.name.value}",
            ))
            self._tasks.append(asyncio.create_task(
                self._watch_balance_loop(config.name, exchange),
                name=f"private:balance:{config.name.value}",
            ))

        logger.info(
            "PrivateStreamManager launched %d stream tasks.", len(self._tasks)
        )

    async def stop(self) -> None:
        """Cancel all stream tasks and close exchange connections."""
        self._running = False

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for name, exchange in self._exchanges.items():
            try:
                await exchange.close()
                logger.info("Closed private stream connection: %s", name.value)
            except Exception:
                logger.exception(
                    "Error closing private stream exchange %s", name.value
                )
        self._exchanges.clear()
        logger.info("PrivateStreamManager stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def active_task_count(self) -> int:
        return sum(1 for t in self._tasks if not t.done())

    # ------------------------------------------------------------------
    # Stream loops
    # ------------------------------------------------------------------

    async def _watch_orders_loop(
        self, exchange_name: Exchange, exchange: Any
    ) -> None:
        """Watch private order updates (Bybit V5: ``order`` topic)."""
        logger.info("Private order stream started: %s", exchange_name.value)
        consecutive_errors = 0

        while self._running:
            try:
                orders = await exchange.watch_orders()
                consecutive_errors = 0

                for raw in orders:
                    event = self._parse_order_update(exchange_name, raw)
                    if event:
                        await self._event_bus.publish(self._order_topic, event)

            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_errors += 1
                backoff = min(1.0 * (2 ** (consecutive_errors - 1)), 60.0)
                logger.exception(
                    "Error in private order stream %s (attempt %d, backoff %.1fs)",
                    exchange_name.value, consecutive_errors, backoff,
                )
                if consecutive_errors >= 20:
                    logger.critical(
                        "Private order stream %s exceeded max errors; stopping.",
                        exchange_name.value,
                    )
                    break
                await asyncio.sleep(backoff)

    async def _watch_trades_loop(
        self, exchange_name: Exchange, exchange: Any
    ) -> None:
        """Watch private trade/execution updates (Bybit V5: ``execution`` topic)."""
        logger.info("Private trade stream started: %s", exchange_name.value)
        consecutive_errors = 0

        while self._running:
            try:
                trades = await exchange.watch_my_trades()
                consecutive_errors = 0

                for raw in trades:
                    event = self._parse_fill_event(exchange_name, raw)
                    if event:
                        await self._event_bus.publish(self._fill_topic, event)

            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_errors += 1
                backoff = min(1.0 * (2 ** (consecutive_errors - 1)), 60.0)
                logger.exception(
                    "Error in private trade stream %s (attempt %d, backoff %.1fs)",
                    exchange_name.value, consecutive_errors, backoff,
                )
                if consecutive_errors >= 20:
                    break
                await asyncio.sleep(backoff)

    async def _watch_positions_loop(
        self, exchange_name: Exchange, exchange: Any
    ) -> None:
        """Watch position updates (Bybit V5: ``position`` topic)."""
        logger.info("Private position stream started: %s", exchange_name.value)
        consecutive_errors = 0

        while self._running:
            try:
                # watch_positions may not accept symbols on all exchanges
                try:
                    positions = await exchange.watch_positions(self._symbols)
                except TypeError:
                    positions = await exchange.watch_positions()
                consecutive_errors = 0

                for raw in positions:
                    event = self._parse_position_update(exchange_name, raw)
                    if event:
                        await self._event_bus.publish(
                            self._position_topic, event
                        )

            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_errors += 1
                backoff = min(1.0 * (2 ** (consecutive_errors - 1)), 60.0)
                logger.exception(
                    "Error in private position stream %s (attempt %d, backoff %.1fs)",
                    exchange_name.value, consecutive_errors, backoff,
                )
                if consecutive_errors >= 20:
                    break
                await asyncio.sleep(backoff)

    async def _watch_balance_loop(
        self, exchange_name: Exchange, exchange: Any
    ) -> None:
        """Watch wallet/balance updates (Bybit V5: ``wallet`` topic)."""
        logger.info("Private balance stream started: %s", exchange_name.value)
        consecutive_errors = 0

        while self._running:
            try:
                balance = await exchange.watch_balance()
                consecutive_errors = 0

                # CCXT Pro normalises balance into {currency: {free, used, total}}
                for currency in balance.get("free", {}):
                    free = balance["free"].get(currency, 0) or 0
                    used = balance["used"].get(currency, 0) or 0
                    total = balance["total"].get(currency, 0) or 0

                    if float(total) == 0:
                        continue

                    event = BalanceUpdate(
                        source_module="data.private_streams",
                        currency=currency,
                        exchange=exchange_name,
                        total=Decimal(str(total)),
                        free=Decimal(str(free)),
                        used=Decimal(str(used)),
                    )
                    await self._event_bus.publish(self._balance_topic, event)

            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_errors += 1
                backoff = min(1.0 * (2 ** (consecutive_errors - 1)), 60.0)
                logger.exception(
                    "Error in private balance stream %s (attempt %d, backoff %.1fs)",
                    exchange_name.value, consecutive_errors, backoff,
                )
                if consecutive_errors >= 20:
                    break
                await asyncio.sleep(backoff)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_order_update(
        exchange_name: Exchange, raw: dict[str, Any]
    ) -> OrderUpdate | None:
        """Parse a CCXT Pro raw order dict into an ``OrderUpdate`` event."""
        try:
            status_str = str(raw.get("status", "open"))
            status = _STATUS_MAP.get(status_str, OrderStatus.SUBMITTED)

            return OrderUpdate(
                source_module="data.private_streams",
                order_id=str(raw.get("id", "")),
                client_order_id=str(raw.get("clientOrderId", "")),
                symbol=str(raw.get("symbol", "")),
                exchange=exchange_name,
                status=status,
                filled_qty=Decimal(str(raw.get("filled", 0) or 0)),
                remaining_qty=Decimal(str(raw.get("remaining", 0) or 0)),
                avg_fill_price=(
                    Decimal(str(raw["average"]))
                    if raw.get("average")
                    else None
                ),
            )
        except Exception:
            logger.exception("Failed to parse order update: %s", raw)
            return None

    @staticmethod
    def _parse_fill_event(
        exchange_name: Exchange, raw: dict[str, Any]
    ) -> FillEvent | None:
        """Parse a CCXT Pro raw trade dict into a ``FillEvent``."""
        try:
            side_str = str(raw.get("side", "buy")).lower()
            fee_info = raw.get("fee", {}) or {}

            return FillEvent(
                source_module="data.private_streams",
                fill_id=str(raw.get("id", "")),
                order_id=str(raw.get("order", "")),
                client_order_id=str(raw.get("clientOrderId", "")),
                symbol=str(raw.get("symbol", "")),
                exchange=exchange_name,
                side=Side.BUY if side_str == "buy" else Side.SELL,
                price=Decimal(str(raw.get("price", 0))),
                qty=Decimal(str(raw.get("amount", 0))),
                fee=Decimal(str(fee_info.get("cost", 0) or 0)),
                fee_currency=str(fee_info.get("currency", "")),
                is_maker=bool(raw.get("maker", raw.get("takerOrMaker") == "maker")),
            )
        except Exception:
            logger.exception("Failed to parse fill event: %s", raw)
            return None

    @staticmethod
    def _parse_position_update(
        exchange_name: Exchange, raw: dict[str, Any]
    ) -> PositionUpdate | None:
        """Parse a CCXT Pro raw position dict into a ``PositionUpdate`` event."""
        try:
            contracts = raw.get("contracts", 0) or 0
            contract_size = raw.get("contractSize", 1) or 1
            qty = Decimal(str(contracts)) * Decimal(str(contract_size))

            entry = Decimal(str(raw.get("entryPrice", 0) or 0))
            mark = Decimal(str(raw.get("markPrice", 0) or 0))
            unrealized = Decimal(str(raw.get("unrealizedPnl", 0) or 0))
            realized = Decimal(str(raw.get("realizedPnl", 0) or 0))
            leverage = int(raw.get("leverage", 1) or 1)

            return PositionUpdate(
                source_module="data.private_streams",
                symbol=str(raw.get("symbol", "")),
                exchange=exchange_name,
                qty=qty,
                entry_price=entry,
                mark_price=mark,
                unrealized_pnl=unrealized,
                realized_pnl=realized,
                leverage=leverage,
            )
        except Exception:
            logger.exception("Failed to parse position update: %s", raw)
            return None
