"""Abstract FX broker adapter skeleton.

Implements :class:`IExchangeAdapter` for FX brokers (OANDA, IB, LMAX).
Subclasses override the ``_create_order``, ``_cancel_order``,
``_fetch_positions``, ``_fetch_balances``, and ``_fetch_instrument``
hooks for each broker's API.

Idempotency: every ``submit_order`` call propagates
``intent.dedupe_key`` as the idempotency key to the broker.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import (
    AssetClass,
    Exchange,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from agentic_trading.core.events import OrderAck, OrderIntent
from agentic_trading.core.ids import new_id, utc_now
from agentic_trading.core.models import Balance, Instrument, Order, Position

from .base import AdapterCapabilities

logger = logging.getLogger(__name__)


class FXBrokerAdapter:
    """Abstract FX broker adapter.

    Provides the :class:`IExchangeAdapter` interface.  Concrete subclasses
    (e.g. ``OandaAdapter``) implement the private ``_*`` hooks.

    Parameters
    ----------
    exchange:
        The :class:`Exchange` enum member for this broker (e.g. ``OANDA``).
    account_ccy:
        Account denomination currency (default ``"USD"``).
    instruments:
        Pre-loaded instrument metadata keyed by symbol.
    """

    def __init__(
        self,
        exchange: Exchange,
        account_ccy: str = "USD",
        instruments: dict[str, Instrument] | None = None,
    ) -> None:
        self._exchange = exchange
        self._account_ccy = account_ccy
        self._instruments: dict[str, Instrument] = instruments or {}
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            asset_classes=[AssetClass.FX],
            supports_leverage=True,
            supports_short=True,
            supports_funding=False,
            supports_rollover=True,
            supports_batch_orders=False,
            supports_amend=True,
            position_mode="netting",
            supported_order_types=[
                OrderType.MARKET,
                OrderType.LIMIT,
                OrderType.STOP_MARKET,
            ],
            supported_tif=[
                TimeInForce.GTC,
                TimeInForce.IOC,
                TimeInForce.FOK,
            ],
        )

    # ------------------------------------------------------------------
    # IExchangeAdapter — core methods
    # ------------------------------------------------------------------

    async def submit_order(self, intent: OrderIntent) -> OrderAck:
        """Submit an FX order to the broker.

        Subclasses implement ``_create_order`` with broker-specific logic.
        """
        instrument = self._instruments.get(intent.symbol)
        if instrument is None:
            return OrderAck(
                order_id="",
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                status=OrderStatus.REJECTED,
                message=f"Instrument not loaded: {intent.symbol}",
                trace_id=intent.trace_id,
                causation_id=intent.event_id,
            )
        try:
            return await self._create_order(intent, instrument)
        except Exception as exc:
            logger.error(
                "FX order submission failed: symbol=%s error=%s",
                intent.symbol,
                exc,
            )
            return OrderAck(
                order_id="",
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                status=OrderStatus.REJECTED,
                message=str(exc),
                trace_id=intent.trace_id,
                causation_id=intent.event_id,
            )

    async def cancel_order(self, order_id: str, symbol: str) -> OrderAck:
        return await self._cancel_order(order_id, symbol)

    async def cancel_all_orders(
        self, symbol: str | None = None
    ) -> list[OrderAck]:
        raise NotImplementedError("cancel_all_orders")

    async def get_open_orders(
        self, symbol: str | None = None
    ) -> list[Order]:
        raise NotImplementedError("get_open_orders")

    async def get_positions(
        self, symbol: str | None = None
    ) -> list[Position]:
        raise NotImplementedError("get_positions")

    async def get_balances(self) -> list[Balance]:
        raise NotImplementedError("get_balances")

    async def get_instrument(self, symbol: str) -> Instrument:
        inst = self._instruments.get(symbol)
        if inst is not None:
            return inst
        return await self._fetch_instrument(symbol)

    async def get_funding_rate(self, symbol: str) -> Decimal:
        """FX has no funding rate. Always returns zero."""
        return Decimal("0")

    # ------------------------------------------------------------------
    # IExchangeAdapter — V5-enhanced (optional)
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
        raise NotImplementedError("amend_order")

    async def batch_submit_orders(
        self, intents: list[OrderIntent]
    ) -> list[OrderAck]:
        raise NotImplementedError("batch_submit_orders")

    async def set_leverage(
        self, symbol: str, leverage: int
    ) -> dict[str, Any]:
        raise NotImplementedError("set_leverage")

    async def set_position_mode(
        self, symbol: str, mode: str
    ) -> dict[str, Any]:
        raise NotImplementedError("set_position_mode")

    async def set_trading_stop(
        self,
        symbol: str,
        *,
        take_profit: Decimal | None = None,
        stop_loss: Decimal | None = None,
        trailing_stop: Decimal | None = None,
        active_price: Decimal | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("set_trading_stop")

    async def get_closed_pnl(
        self, symbol: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("get_closed_pnl")

    # ------------------------------------------------------------------
    # IExchangeAdapter — FX-specific
    # ------------------------------------------------------------------

    async def get_rollover_rates(
        self, symbol: str
    ) -> dict[str, Decimal]:
        """Fetch overnight rollover/swap rates for *symbol*.

        Returns ``{"long": rate, "short": rate}``.
        """
        raise NotImplementedError("get_rollover_rates")

    async def get_spread(self, symbol: str) -> dict[str, Decimal]:
        """Return current bid, ask, and spread in pips.

        Returns ``{"bid": ..., "ask": ..., "spread_pips": ...}``.
        """
        raise NotImplementedError("get_spread")

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    async def _create_order(
        self, intent: OrderIntent, instrument: Instrument
    ) -> OrderAck:
        """Broker-specific order creation. Override in subclass."""
        raise NotImplementedError

    async def _cancel_order(
        self, order_id: str, symbol: str
    ) -> OrderAck:
        """Broker-specific order cancellation. Override in subclass."""
        raise NotImplementedError

    async def _fetch_instrument(self, symbol: str) -> Instrument:
        """Broker-specific instrument fetch. Override in subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def load_instrument(self, instrument: Instrument) -> None:
        """Register an instrument for local lookup."""
        self._instruments[instrument.symbol] = instrument
