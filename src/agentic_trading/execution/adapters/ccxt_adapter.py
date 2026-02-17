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
    demo:
        If ``True``, enable Bybit demo trading (production URL, virtual
        funds).  Mutually exclusive with *sandbox*.
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
        demo: bool = False,
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

        if demo:
            # Bybit demo trading: production URL, virtual funds
            self._ccxt.enable_demo_trading(True)
            logger.info(
                "CCXT adapter initialised in DEMO mode for %s (type=%s)",
                self._exchange_name,
                default_type,
            )
        elif sandbox:
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
        self._default_type = default_type

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_markets(self) -> None:
        """Lazy-load markets on first use."""
        if not self._markets_loaded:
            await self._ccxt.load_markets()
            self._markets_loaded = True

    def _to_swap_symbol(self, symbol: str) -> str:
        """Convert spot-style symbol to perpetual format if needed.

        ``BTC/USDT`` → ``BTC/USDT:USDT`` for Bybit linear perpetuals.
        If the symbol already has a settle suffix it is returned unchanged.
        """
        if self._default_type != "swap":
            return symbol
        if ":" in symbol:
            return symbol  # already has settle suffix
        # For Bybit/Binance linear USDT perpetuals, append :USDT
        quote = symbol.split("/")[1] if "/" in symbol else "USDT"
        return f"{symbol}:{quote}"

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
            # Convert spot-format symbol to perpetual if needed
            ccxt_symbol = self._to_swap_symbol(intent.symbol)

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
                symbol=ccxt_symbol,
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
            raw_status = result.get("status")
            avg_fill_price = result.get("average") or result.get("price")

            # Bybit market orders often return status=None in the create
            # response.  Fetch the order to get the real status and fill
            # price so downstream consumers (journal, narration) see a
            # proper FILLED ack.
            if raw_status is None and order_id and ccxt_type == "market":
                try:
                    # Bybit requires fetchClosedOrder for filled market
                    # orders (fetchOrder only works for recent open orders).
                    if (
                        self._exchange_name == "bybit"
                        and hasattr(self._ccxt, "fetch_closed_order")
                    ):
                        fetched = await self._ccxt.fetch_closed_order(
                            order_id, ccxt_symbol
                        )
                    else:
                        fetched = await self._ccxt.fetch_order(
                            order_id, ccxt_symbol
                        )
                    raw_status = fetched.get("status")
                    avg_fill_price = (
                        fetched.get("average")
                        or fetched.get("price")
                        or avg_fill_price
                    )
                    logger.info(
                        "Fetched order %s for status resolution: status=%s avg=%s",
                        order_id, raw_status, avg_fill_price,
                    )
                except Exception as fetch_exc:
                    # Market orders on Bybit almost always fill instantly;
                    # if we can't fetch it, assume filled.
                    if ccxt_type == "market":
                        raw_status = "closed"
                        # Try to get last price from ticker for the fill
                        if avg_fill_price is None:
                            try:
                                ticker = await self._ccxt.fetch_ticker(ccxt_symbol)
                                avg_fill_price = ticker.get("last")
                            except Exception:
                                pass
                        logger.info(
                            "Could not fetch market order %s (%s); "
                            "assuming filled at %s",
                            order_id, fetch_exc, avg_fill_price,
                        )
                    else:
                        logger.warning(
                            "Could not fetch order %s for status: %s",
                            order_id, fetch_exc,
                        )

            status = _map_status(str(raw_status or "open"))

            # Store avg fill price so engine can read it for FillEvent
            self._last_fill_price: Decimal | None = (
                Decimal(str(avg_fill_price)) if avg_fill_price else None
            )

            logger.info(
                "CCXT order created: id=%s client_id=%s symbol=%s "
                "side=%s type=%s qty=%s price=%s avg_fill=%s status=%s",
                order_id,
                client_order_id,
                intent.symbol,
                side_str,
                ccxt_type,
                intent.qty,
                price,
                avg_fill_price,
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
            ccxt_symbol = self._to_swap_symbol(symbol)
            result: dict[str, Any] = await self._ccxt.cancel_order(
                id=order_id, symbol=ccxt_symbol
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
            symbols = [self._to_swap_symbol(symbol)] if symbol else None
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
    # IExchangeAdapter: amend_order (V5-enhanced)
    # ------------------------------------------------------------------

    async def amend_order(
        self,
        order_id: str,
        symbol: str,
        *,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> OrderAck:
        """Amend an open order in-place (modify price/qty without cancel+replace).

        Maps to Bybit V5 ``/v5/order/amend`` via CCXT's ``edit_order``.
        On Binance, falls back to cancel-and-replace.
        """
        await self._ensure_markets()
        try:
            params: dict[str, Any] = {}
            if stop_price is not None:
                params["stopPrice"] = float(stop_price)

            amount = float(qty) if qty is not None else None
            new_price = float(price) if price is not None else None

            if hasattr(self._ccxt, "edit_order"):
                result: dict[str, Any] = await self._ccxt.edit_order(
                    id=order_id,
                    symbol=symbol,
                    type=None,  # preserve existing type
                    side=None,  # preserve existing side
                    amount=amount,
                    price=new_price,
                    params=params,
                )
            else:
                # Fallback: cancel + re-create (less atomic)
                await self._ccxt.cancel_order(id=order_id, symbol=symbol)
                # Re-fetch the original order info to get type/side
                raise ExchangeError(
                    f"Exchange {self._exchange_name} does not support edit_order; "
                    "use cancel_order + submit_order instead."
                )

            logger.info(
                "CCXT order amended: id=%s symbol=%s qty=%s price=%s stop=%s",
                order_id, symbol, qty, price, stop_price,
            )
            return OrderAck(
                order_id=str(result.get("id", order_id)),
                client_order_id=str(result.get("clientOrderId", "")),
                symbol=symbol,
                exchange=self._exchange_enum,
                status=_map_status(str(result.get("status", "open"))),
            )
        except ExchangeError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: batch_submit_orders (V5-enhanced)
    # ------------------------------------------------------------------

    async def batch_submit_orders(
        self, intents: list[OrderIntent]
    ) -> list[OrderAck]:
        """Submit multiple orders in a single request where supported.

        Maps to Bybit V5 ``/v5/order/create-batch`` (up to 20 orders).
        Falls back to sequential ``submit_order`` calls for exchanges
        without native batch support.
        """
        await self._ensure_markets()
        try:
            # Check if exchange supports native batch orders
            if hasattr(self._ccxt, "create_orders"):
                # Build CCXT order list
                order_dicts: list[dict[str, Any]] = []
                for intent in intents:
                    ccxt_type = _ORDER_TYPE_MAP.get(intent.order_type, "limit")
                    side_str = intent.side.value
                    price = float(intent.price) if intent.price is not None else None
                    params: dict[str, Any] = {}

                    if self._exchange_name == "bybit":
                        params["orderLinkId"] = intent.dedupe_key
                    else:
                        params["newClientOrderId"] = intent.dedupe_key

                    if intent.stop_price is not None:
                        params["stopPrice"] = float(intent.stop_price)
                    if intent.reduce_only:
                        params["reduceOnly"] = True
                    if intent.post_only:
                        params["postOnly"] = True
                    if intent.time_in_force != TimeInForce.GTC:
                        params["timeInForce"] = intent.time_in_force.value

                    order_dicts.append({
                        "symbol": intent.symbol,
                        "type": ccxt_type,
                        "side": side_str,
                        "amount": float(intent.qty),
                        "price": price,
                        "params": params,
                    })

                results: list[dict[str, Any]] = await self._ccxt.create_orders(
                    order_dicts
                )

                acks: list[OrderAck] = []
                for i, result in enumerate(results):
                    trace_id = intents[i].trace_id if i < len(intents) else ""
                    acks.append(
                        OrderAck(
                            order_id=str(result.get("id", "")),
                            client_order_id=str(
                                result.get("clientOrderId", intents[i].dedupe_key)
                            ),
                            symbol=result.get("symbol", intents[i].symbol),
                            exchange=self._exchange_enum,
                            status=_map_status(str(result.get("status", "open"))),
                            trace_id=trace_id,
                        )
                    )

                logger.info(
                    "CCXT batch submitted %d orders (native batch)", len(acks)
                )
                return acks

            else:
                # Fallback: sequential submission
                acks = []
                for intent in intents:
                    ack = await self.submit_order(intent)
                    acks.append(ack)
                logger.info(
                    "CCXT batch submitted %d orders (sequential fallback)",
                    len(acks),
                )
                return acks

        except ExchangeError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: set_leverage (V5-enhanced)
    # ------------------------------------------------------------------

    async def set_leverage(
        self, symbol: str, leverage: int
    ) -> dict[str, Any]:
        """Set per-symbol leverage for linear/inverse perpetuals.

        Maps to Bybit V5 ``/v5/position/set-leverage``.
        """
        await self._ensure_markets()
        try:
            result = await self._ccxt.set_leverage(leverage, symbol)
            logger.info(
                "CCXT leverage set: symbol=%s leverage=%dx", symbol, leverage
            )
            return result if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: set_position_mode (V5-enhanced)
    # ------------------------------------------------------------------

    async def set_position_mode(
        self, symbol: str, mode: str
    ) -> dict[str, Any]:
        """Switch between one-way and hedge position mode.

        Maps to Bybit V5 ``/v5/position/switch-mode``.
        ``mode`` should be ``"one_way"`` or ``"hedge"``.
        """
        await self._ensure_markets()
        try:
            # CCXT uses hedged=True/False
            hedged = mode.lower() == "hedge"
            result = await self._ccxt.set_position_mode(hedged, symbol)
            logger.info(
                "CCXT position mode set: symbol=%s mode=%s (hedged=%s)",
                symbol, mode, hedged,
            )
            return result if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: set_trading_stop (V5-enhanced)
    # ------------------------------------------------------------------

    async def set_trading_stop(
        self,
        symbol: str,
        *,
        take_profit: Decimal | None = None,
        stop_loss: Decimal | None = None,
        trailing_stop: Decimal | None = None,
    ) -> dict[str, Any]:
        """Set server-side TP/SL/trailing stop on an existing position.

        Maps to Bybit V5 ``/v5/position/trading-stop``.
        For exchanges without a native endpoint, raises ExchangeError.
        """
        await self._ensure_markets()
        try:
            if take_profit is None and stop_loss is None and trailing_stop is None:
                raise ExchangeError(
                    "set_trading_stop requires at least one of "
                    "take_profit, stop_loss, or trailing_stop"
                )

            if self._exchange_name == "bybit":
                # Bybit V5: use the private endpoint via CCXT's implicit API.
                # Required params: category, symbol, positionIdx, tpslMode.
                # positionIdx=0 for one-way mode (our default).
                # tpslMode="Full" applies TP/SL to the entire position.

                # Round TP/SL/trailing to the instrument's tick size so Bybit
                # doesn't reject for invalid precision.
                try:
                    _swap_sym = self._to_swap_symbol(symbol)
                    _mkt = self._ccxt.market(_swap_sym)
                    _prec = _mkt.get("precision", {}) if _mkt else {}
                    _price_prec = _prec.get("price", 4)
                    if isinstance(_price_prec, float) and _price_prec < 1:
                        _dp = abs(Decimal(str(_price_prec)).as_tuple().exponent)
                    else:
                        _dp = int(_price_prec)
                except Exception:
                    _dp = 4  # safe fallback

                def _round_price(v: Decimal) -> Decimal:
                    return v.quantize(Decimal(10) ** -_dp)

                request_params: dict[str, Any] = {
                    "category": "linear",
                    "symbol": self._to_bybit_symbol(symbol),
                    "positionIdx": 0,
                    "tpslMode": "Full",
                    "tpTriggerBy": "LastPrice",
                    "slTriggerBy": "LastPrice",
                }
                if take_profit is not None:
                    request_params["takeProfit"] = str(_round_price(take_profit))
                if stop_loss is not None:
                    request_params["stopLoss"] = str(_round_price(stop_loss))
                if trailing_stop is not None:
                    request_params["trailingStop"] = str(_round_price(trailing_stop))

                result = await self._ccxt.privatePostV5PositionTradingStop(
                    request_params
                )
                logger.info(
                    "CCXT trading stop set: symbol=%s tp=%s sl=%s trail=%s",
                    symbol, take_profit, stop_loss, trailing_stop,
                )
                return result if isinstance(result, dict) else {"result": result}

            # Generic fallback for other exchanges
            raise ExchangeError(
                f"Exchange {self._exchange_name} does not support "
                "set_trading_stop; use conditional stop orders instead."
            )

        except ExchangeError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_closed_pnl (V5-enhanced)
    # ------------------------------------------------------------------

    async def get_closed_pnl(
        self, symbol: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch closed PnL records for a symbol.

        Maps to Bybit V5 ``/v5/position/closed-pnl``.
        Falls back to CCXT's ``fetch_my_trades`` and aggregates by position.
        """
        await self._ensure_markets()
        try:
            # Bybit-specific: native closed PnL endpoint
            if (
                self._exchange_name == "bybit"
                and hasattr(self._ccxt, "privateGetV5PositionClosedPnl")
            ):
                result = await self._ccxt.privateGetV5PositionClosedPnl({
                    "category": "linear",
                    "symbol": self._to_bybit_symbol(symbol),
                    "limit": str(limit),
                })
                raw_list = result.get("result", {}).get("list", [])
                records: list[dict[str, Any]] = []
                for r in raw_list:
                    records.append({
                        "symbol": symbol,
                        "side": r.get("side", ""),
                        "qty": r.get("qty", "0"),
                        "entry_price": r.get("avgEntryPrice", "0"),
                        "exit_price": r.get("avgExitPrice", "0"),
                        "closed_pnl": r.get("closedPnl", "0"),
                        "fill_count": r.get("fillCount", 0),
                        "leverage": r.get("leverage", "1"),
                        "created_at": r.get("createdTime", ""),
                        "updated_at": r.get("updatedTime", ""),
                        "order_id": r.get("orderId", ""),
                    })
                logger.info(
                    "CCXT closed PnL fetched: symbol=%s count=%d",
                    symbol, len(records),
                )
                return records

            # Generic fallback: use fetch_my_trades
            trades = await self._ccxt.fetch_my_trades(
                symbol=symbol, limit=limit
            )
            records = []
            for t in trades:
                records.append({
                    "symbol": t.get("symbol", symbol),
                    "side": t.get("side", ""),
                    "qty": str(t.get("amount", 0)),
                    "price": str(t.get("price", 0)),
                    "fee": str(t.get("fee", {}).get("cost", 0)),
                    "fee_currency": t.get("fee", {}).get("currency", ""),
                    "timestamp": t.get("timestamp", 0),
                    "order_id": t.get("order", ""),
                    "trade_id": t.get("id", ""),
                })
            logger.info(
                "CCXT trades fetched (closed PnL fallback): symbol=%s count=%d",
                symbol, len(records),
            )
            return records

        except ExchangeError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # ------------------------------------------------------------------
    # Internal: symbol conversion
    # ------------------------------------------------------------------

    def _to_bybit_symbol(self, unified_symbol: str) -> str:
        """Convert CCXT unified symbol (e.g. 'BTC/USDT:USDT') to Bybit format ('BTCUSDT')."""
        # Remove settle currency suffix and slash
        return unified_symbol.replace("/", "").split(":")[0]

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
