"""Unified CCXT adapter for Binance and Bybit.

Uses the ``ccxt`` library in async mode to implement the
``IExchangeAdapter`` protocol.  Supports spot and perpetual futures
across both supported exchanges.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt_async

from agentic_trading.core.enums import (
    Exchange,
    InstrumentType,
    MarginMode,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
    TimeInForce,
)
from agentic_trading.core.errors import (
    AuthenticationError,
    ExchangeError,
    InsufficientBalanceError,
    RateLimitError,
)
from agentic_trading.core.events import OrderAck, OrderIntent
from agentic_trading.core.models import Balance, Instrument, Order, Position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status mapping: CCXT status string -> OrderStatus
# ---------------------------------------------------------------------------

_CCXT_STATUS_MAP: dict[str, OrderStatus] = {
    "open": OrderStatus.SUBMITTED,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


def _map_status(ccxt_status: str) -> OrderStatus:
    """Convert a CCXT order status string to our ``OrderStatus``."""
    return _CCXT_STATUS_MAP.get(ccxt_status, OrderStatus.SUBMITTED)


# ---------------------------------------------------------------------------
# Order type mapping: our OrderType -> CCXT type string
# ---------------------------------------------------------------------------

_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "market",
    OrderType.LIMIT: "limit",
    OrderType.STOP_MARKET: "stop_market",
    OrderType.STOP_LIMIT: "stop_limit",
    OrderType.TAKE_PROFIT_MARKET: "take_profit_market",
    OrderType.TAKE_PROFIT_LIMIT: "take_profit_limit",
}


class CCXTAdapter:
    """Exchange adapter backed by ``ccxt.async_support``.

    Parameters
    ----------
    exchange_name:
        The exchange identifier -- ``"binance"`` or ``"bybit"``.
    api_key:
        API key for the exchange.
    api_secret:
        API secret for the exchange.
    passphrase:
        Optional passphrase (required by some exchanges).
    sandbox:
        If ``True``, use the exchange's sandbox/testnet environment.
    default_type:
        Market type for CCXT (``"spot"``, ``"swap"``, ``"future"``).
        Defaults to ``"swap"`` for perpetual futures.
    options:
        Extra CCXT options dict merged into the exchange constructor.
    """

    def __init__(
        self,
        exchange_name: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        sandbox: bool = False,
        default_type: str = "swap",
        options: dict[str, Any] | None = None,
    ) -> None:
        self._exchange_name = exchange_name.lower()
        self._exchange_enum = Exchange(self._exchange_name)

        # Resolve the CCXT exchange class
        exchange_class = getattr(ccxt_async, self._exchange_name, None)
        if exchange_class is None:
            raise ExchangeError(
                f"Unsupported exchange: {self._exchange_name}"
            )

        config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": default_type,
                **(options or {}),
            },
        }
        if passphrase:
            config["password"] = passphrase

        self._ccxt: ccxt_async.Exchange = exchange_class(config)

        if sandbox:
            self._ccxt.set_sandbox_mode(True)
            logger.info(
                "CCXT adapter initialised in SANDBOX mode for %s",
                self._exchange_name,
            )
        else:
            logger.info(
                "CCXT adapter initialised for %s (type=%s)",
                self._exchange_name,
                default_type,
            )

        self._markets_loaded: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_markets(self) -> None:
        """Lazy-load markets on first use."""
        if not self._markets_loaded:
            await self._ccxt.load_markets()
            self._markets_loaded = True

    def _wrap_error(self, exc: Exception) -> ExchangeError:
        """Convert CCXT exceptions to our error hierarchy."""
        if isinstance(exc, ccxt_async.AuthenticationError):
            return AuthenticationError(str(exc))
        if isinstance(exc, ccxt_async.RateLimitExceeded):
            return RateLimitError(str(exc))
        if isinstance(exc, ccxt_async.InsufficientFunds):
            return InsufficientBalanceError(str(exc))
        if isinstance(exc, ccxt_async.BaseError):
            return ExchangeError(str(exc))
        return ExchangeError(str(exc))

    # ------------------------------------------------------------------
    # IExchangeAdapter: submit_order
    # ------------------------------------------------------------------

    async def submit_order(self, intent: OrderIntent) -> OrderAck:
        """Submit an order to the exchange.

        Maps the ``OrderIntent`` to a CCXT ``create_order`` call.  The
        ``dedupe_key`` is forwarded as the ``clientOrderId`` parameter so
        the exchange enforces idempotency.
        """
        await self._ensure_markets()
        try:
            ccxt_type = _ORDER_TYPE_MAP.get(intent.order_type, "limit")
            side_str = intent.side.value  # "buy" or "sell"
            price = float(intent.price) if intent.price is not None else None
            params: dict[str, Any] = {
                "newClientOrderId": intent.dedupe_key,
            }

            # Binance uses "newClientOrderId", Bybit uses "orderLinkId"
            if self._exchange_name == "bybit":
                params["orderLinkId"] = intent.dedupe_key

            if intent.stop_price is not None:
                params["stopPrice"] = float(intent.stop_price)
            if intent.reduce_only:
                params["reduceOnly"] = True
            if intent.post_only:
                params["postOnly"] = True
            if intent.leverage is not None:
                params["leverage"] = intent.leverage
            if intent.time_in_force != TimeInForce.GTC:
                params["timeInForce"] = intent.time_in_force.value

            result: dict[str, Any] = await self._ccxt.create_order(
                symbol=intent.symbol,
                type=ccxt_type,
                side=side_str,
                amount=float(intent.qty),
                price=price,
                params=params,
            )

            order_id = str(result.get("id", ""))
            client_order_id = str(
                result.get("clientOrderId", intent.dedupe_key)
            )
            status = _map_status(str(result.get("status", "open")))

            logger.info(
                "CCXT order created: id=%s client_id=%s symbol=%s "
                "side=%s type=%s qty=%s price=%s status=%s",
                order_id,
                client_order_id,
                intent.symbol,
                side_str,
                ccxt_type,
                intent.qty,
                price,
                status.value,
            )
            return OrderAck(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=intent.symbol,
                exchange=self._exchange_enum,
                status=status,
                trace_id=intent.trace_id,
            )

        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: cancel_order
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str, symbol: str) -> OrderAck:
        """Cancel an open order on the exchange."""
        await self._ensure_markets()
        try:
            result: dict[str, Any] = await self._ccxt.cancel_order(
                id=order_id, symbol=symbol
            )
            return OrderAck(
                order_id=str(result.get("id", order_id)),
                client_order_id=str(result.get("clientOrderId", "")),
                symbol=symbol,
                exchange=self._exchange_enum,
                status=OrderStatus.CANCELLED,
            )
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: cancel_all_orders
    # ------------------------------------------------------------------

    async def cancel_all_orders(
        self, symbol: str | None = None
    ) -> list[OrderAck]:
        """Cancel all open orders, optionally filtered by symbol."""
        await self._ensure_markets()
        try:
            if hasattr(self._ccxt, "cancel_all_orders"):
                results: list[dict[str, Any]] = (
                    await self._ccxt.cancel_all_orders(symbol=symbol)
                    if symbol
                    else await self._ccxt.cancel_all_orders()
                )
            else:
                # Fallback: fetch open orders and cancel one by one
                open_orders = await self._ccxt.fetch_open_orders(
                    symbol=symbol
                )
                results = []
                for o in open_orders:
                    cancelled = await self._ccxt.cancel_order(
                        id=o["id"], symbol=o.get("symbol", symbol)
                    )
                    results.append(cancelled)

            acks: list[OrderAck] = []
            for r in results:
                acks.append(
                    OrderAck(
                        order_id=str(r.get("id", "")),
                        client_order_id=str(r.get("clientOrderId", "")),
                        symbol=r.get("symbol", symbol or ""),
                        exchange=self._exchange_enum,
                        status=OrderStatus.CANCELLED,
                    )
                )
            logger.info("Cancelled %d orders (symbol=%s)", len(acks), symbol)
            return acks

        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_open_orders
    # ------------------------------------------------------------------

    async def get_open_orders(
        self, symbol: str | None = None
    ) -> list[Order]:
        """Fetch currently open orders from the exchange."""
        await self._ensure_markets()
        try:
            raw_orders: list[dict[str, Any]] = (
                await self._ccxt.fetch_open_orders(symbol=symbol)
            )
            return [self._parse_order(o) for o in raw_orders]
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_positions
    # ------------------------------------------------------------------

    async def get_positions(
        self, symbol: str | None = None
    ) -> list[Position]:
        """Fetch open positions (perpetual futures) from the exchange."""
        await self._ensure_markets()
        try:
            symbols = [symbol] if symbol else None
            raw_positions: list[dict[str, Any]] = (
                await self._ccxt.fetch_positions(symbols=symbols)
            )
            positions: list[Position] = []
            for p in raw_positions:
                parsed = self._parse_position(p)
                if parsed is not None:
                    positions.append(parsed)
            return positions
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_balances
    # ------------------------------------------------------------------

    async def get_balances(self) -> list[Balance]:
        """Fetch account balances from the exchange."""
        await self._ensure_markets()
        try:
            raw: dict[str, Any] = await self._ccxt.fetch_balance()
            balances: list[Balance] = []
            info = raw.get("info", {})
            # CCXT normalises balances under currency keys
            for currency in raw.get("free", {}):
                free_val = Decimal(str(raw["free"].get(currency, 0) or 0))
                used_val = Decimal(str(raw["used"].get(currency, 0) or 0))
                total_val = Decimal(str(raw["total"].get(currency, 0) or 0))
                if total_val == Decimal("0"):
                    continue
                balances.append(
                    Balance(
                        currency=currency,
                        exchange=self._exchange_enum,
                        total=total_val,
                        free=free_val,
                        used=used_val,
                        updated_at=datetime.now(timezone.utc),
                    )
                )
            return balances
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_instrument
    # ------------------------------------------------------------------

    async def get_instrument(self, symbol: str) -> Instrument:
        """Load market metadata for a symbol and return an ``Instrument``."""
        await self._ensure_markets()
        try:
            market = self._ccxt.market(symbol)
            if market is None:
                raise ExchangeError(f"Unknown symbol: {symbol}")

            # Determine instrument type
            if market.get("swap"):
                inst_type = InstrumentType.PERP
            elif market.get("future"):
                inst_type = InstrumentType.FUTURE
            else:
                inst_type = InstrumentType.SPOT

            precision = market.get("precision", {})
            limits = market.get("limits", {})
            amount_limits = limits.get("amount", {})
            cost_limits = limits.get("cost", {})

            # Extract tick and step sizes from precision
            price_prec = precision.get("price", 2)
            qty_prec = precision.get("amount", 6)

            # Some CCXT exchanges return precision as number of decimals,
            # others as the actual step.  Normalise to Decimal step.
            if isinstance(price_prec, float) and price_prec < 1:
                tick_size = Decimal(str(price_prec))
                price_prec_int = abs(tick_size.as_tuple().exponent)
            else:
                price_prec_int = int(price_prec)
                tick_size = Decimal(10) ** -price_prec_int

            if isinstance(qty_prec, float) and qty_prec < 1:
                step_size = Decimal(str(qty_prec))
                qty_prec_int = abs(step_size.as_tuple().exponent)
            else:
                qty_prec_int = int(qty_prec)
                step_size = Decimal(10) ** -qty_prec_int

            fees = market.get("fees", market.get("fee", {})) or {}
            maker_fee = Decimal(str(fees.get("maker", market.get("maker", 0.0002)) or 0.0002))
            taker_fee = Decimal(str(fees.get("taker", market.get("taker", 0.0004)) or 0.0004))

            return Instrument(
                symbol=market.get("symbol", symbol),
                exchange=self._exchange_enum,
                instrument_type=inst_type,
                base=market.get("base", ""),
                quote=market.get("quote", ""),
                settle=market.get("settle"),
                price_precision=price_prec_int,
                qty_precision=qty_prec_int,
                tick_size=tick_size,
                step_size=step_size,
                min_qty=Decimal(str(amount_limits.get("min", 0) or 0)),
                max_qty=Decimal(str(amount_limits.get("max", 999999999) or 999999999)),
                min_notional=Decimal(str(cost_limits.get("min", 0) or 0)),
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                is_active=market.get("active", True),
            )
        except ExchangeError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_funding_rate
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Decimal:
        """Fetch the current funding rate for a perpetual futures symbol."""
        await self._ensure_markets()
        try:
            result: dict[str, Any] = await self._ccxt.fetch_funding_rate(
                symbol=symbol
            )
            rate = result.get("fundingRate", 0)
            return Decimal(str(rate or 0))
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying CCXT exchange connection."""
        await self._ccxt.close()
        logger.info("CCXT adapter closed for %s", self._exchange_name)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_order(self, raw: dict[str, Any]) -> Order:
        """Parse a CCXT raw order dict into our canonical ``Order``."""
        side_str = str(raw.get("side", "buy")).lower()
        order_type_str = str(raw.get("type", "limit")).lower()

        # Map order type
        type_reverse: dict[str, OrderType] = {
            v: k for k, v in _ORDER_TYPE_MAP.items()
        }
        order_type = type_reverse.get(order_type_str, OrderType.LIMIT)

        created = raw.get("timestamp")
        created_dt = (
            datetime.fromtimestamp(created / 1000, tz=timezone.utc)
            if created
            else None
        )
        updated = raw.get("lastTradeTimestamp") or created
        updated_dt = (
            datetime.fromtimestamp(updated / 1000, tz=timezone.utc)
            if updated
            else None
        )

        return Order(
            order_id=str(raw.get("id", "")),
            client_order_id=str(raw.get("clientOrderId", "")),
            symbol=str(raw.get("symbol", "")),
            exchange=self._exchange_enum,
            side=Side.BUY if side_str == "buy" else Side.SELL,
            order_type=order_type,
            price=(
                Decimal(str(raw["price"])) if raw.get("price") else None
            ),
            stop_price=(
                Decimal(str(raw["stopPrice"]))
                if raw.get("stopPrice")
                else None
            ),
            qty=Decimal(str(raw.get("amount", 0))),
            filled_qty=Decimal(str(raw.get("filled", 0))),
            remaining_qty=Decimal(str(raw.get("remaining", 0))),
            avg_fill_price=(
                Decimal(str(raw["average"]))
                if raw.get("average")
                else None
            ),
            status=_map_status(str(raw.get("status", "open"))),
            reduce_only=bool(raw.get("reduceOnly", False)),
            post_only=bool(raw.get("postOnly", False)),
            created_at=created_dt,
            updated_at=updated_dt,
        )

    def _parse_position(self, raw: dict[str, Any]) -> Position | None:
        """Parse a CCXT raw position dict into our canonical ``Position``.

        Returns ``None`` for positions with zero size (closed).
        """
        contracts = raw.get("contracts", 0) or 0
        contract_size = raw.get("contractSize", 1) or 1
        qty = Decimal(str(contracts)) * Decimal(str(contract_size))

        # Skip closed positions
        if qty == Decimal("0"):
            return None

        side_str = str(raw.get("side", "long")).lower()
        if side_str == "short":
            pos_side = PositionSide.SHORT
        elif side_str == "long":
            pos_side = PositionSide.LONG
        else:
            pos_side = PositionSide.BOTH

        margin_mode_str = str(raw.get("marginMode", "cross")).lower()
        margin_mode = (
            MarginMode.ISOLATED
            if margin_mode_str == "isolated"
            else MarginMode.CROSS
        )

        entry = Decimal(str(raw.get("entryPrice", 0) or 0))
        mark = Decimal(str(raw.get("markPrice", 0) or 0))
        liq = raw.get("liquidationPrice")
        liq_price = Decimal(str(liq)) if liq else None
        unrealized = Decimal(str(raw.get("unrealizedPnl", 0) or 0))
        notional = Decimal(str(raw.get("notional", 0) or 0))
        leverage = int(raw.get("leverage", 1) or 1)

        ts = raw.get("timestamp")
        updated_at = (
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts else None
        )

        return Position(
            symbol=str(raw.get("symbol", "")),
            exchange=self._exchange_enum,
            side=pos_side,
            qty=qty,
            entry_price=entry,
            mark_price=mark,
            liquidation_price=liq_price,
            unrealized_pnl=unrealized,
            leverage=leverage,
            margin_mode=margin_mode,
            notional=notional,
            updated_at=updated_at,
        )
