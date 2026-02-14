"""Whale activity monitor scaffold.

Provides the interface for tracking large on-chain transactions,
exchange inflow/outflow events, and producing whale-activity features
that strategies can consume.

**Status: Scaffold** -- the on-chain data ingestion and analysis core
is marked as TODO.  The full public API and event-bus integration are
implemented so that downstream consumers can code against a stable
interface today.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from agentic_trading.core.enums import Timeframe
from agentic_trading.core.events import BaseEvent, FeatureVector, WhaleEvent
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)

# Maximum whale events kept per symbol.
_DEFAULT_HISTORY_SIZE = 200

# Threshold in USD above which a transaction is considered "whale".
_DEFAULT_WHALE_THRESHOLD_USD = 1_000_000.0


class WhaleMonitor:
    """Monitors and records large-value (whale) transactions and
    exchange flow events per symbol.

    **Scaffold implementation** -- the actual on-chain data source
    integration (e.g. Whale Alert API, Arkham, on-chain RPC) is a
    placeholder.  The public interface is complete so strategies and
    risk modules can depend on it today.

    Usage (event-bus driven)::

        monitor = WhaleMonitor(event_bus=bus)
        await monitor.start()

    Usage (direct / backtesting)::

        monitor = WhaleMonitor()
        event = monitor.check_whale_activity("BTC/USDT")
        if event is not None:
            print(f"Whale detected: {event.direction} ${event.amount_usd:,.0f}")
    """

    def __init__(
        self,
        event_bus: IEventBus | None = None,
        history_size: int = _DEFAULT_HISTORY_SIZE,
        whale_threshold_usd: float = _DEFAULT_WHALE_THRESHOLD_USD,
    ) -> None:
        self._event_bus = event_bus
        self._history_size = history_size
        self._whale_threshold_usd = whale_threshold_usd

        # Per-symbol store of recent whale events.
        self._whale_store: dict[str, deque[WhaleEvent]] = defaultdict(
            lambda: deque(maxlen=self._history_size)
        )

    # ------------------------------------------------------------------
    # Event bus lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to whale events on the event bus and (in a full
        implementation) start the on-chain polling / websocket listener.
        """
        if self._event_bus is None:
            logger.warning(
                "WhaleMonitor started without event bus - direct mode only"
            )
            return

        await self._event_bus.subscribe(
            topic="feature.whale",
            group="whale_monitor",
            handler=self._handle_whale_event,
        )
        logger.info("WhaleMonitor subscribed to feature.whale")

        # TODO: Start background polling task for on-chain data sources.
        # e.g. self._poll_task = asyncio.create_task(self._poll_chain())

    async def stop(self) -> None:
        """Stop the monitor and cancel any background tasks."""
        # TODO: Cancel background polling task if running.
        logger.info("WhaleMonitor stopped")

    async def _handle_whale_event(self, event: BaseEvent) -> None:
        """Internal handler for externally-published ``WhaleEvent``."""
        if not isinstance(event, WhaleEvent):
            return

        if event.amount_usd < self._whale_threshold_usd:
            return

        self._whale_store[event.symbol].append(event)

        # Publish derived features.
        if self._event_bus is not None:
            features = self._compute_whale_features(event.symbol)
            fv = FeatureVector(
                symbol=event.symbol,
                timeframe=Timeframe.M1,
                features=features,
                source_module="features.whale_monitor",
            )
            await self._event_bus.publish("feature.vector", fv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_whale_activity(self, symbol: str) -> WhaleEvent | None:
        """Check for the most recent whale activity on *symbol*.

        TODO: In a full implementation this would query an on-chain
        data source in real time.  Currently returns the latest stored
        event, or ``None``.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.

        Returns:
            The most recent :class:`WhaleEvent` if one exists, else
            ``None``.
        """
        # TODO: Replace with real on-chain query.
        # Candidates for data sources:
        #   - Whale Alert API (https://whale-alert.io)
        #   - Arkham Intelligence API
        #   - Direct RPC calls to blockchain nodes
        #   - Nansen / Glassnode API
        events = self._whale_store.get(symbol)
        if not events:
            return None
        return events[-1]

    def record_whale_event(
        self,
        symbol: str,
        direction: str,
        amount_usd: float,
        wallet: str = "",
        exchange_name: str = "",
    ) -> WhaleEvent:
        """Manually record a whale event.

        Useful for injecting events during backtesting or from external
        feeds that are not routed through the event bus.

        Args:
            symbol: Trading pair.
            direction: One of ``"inflow"``, ``"outflow"``, ``"transfer"``.
            amount_usd: Transaction value in USD.
            wallet: Wallet address (optional).
            exchange_name: Exchange involved (optional).

        Returns:
            The created :class:`WhaleEvent`.
        """
        event = WhaleEvent(
            symbol=symbol,
            direction=direction,
            amount_usd=amount_usd,
            wallet=wallet,
            exchange_name=exchange_name,
        )
        self._whale_store[symbol].append(event)
        return event

    def get_recent_events(
        self,
        symbol: str,
        limit: int = 10,
        direction: str | None = None,
    ) -> list[WhaleEvent]:
        """Return recent whale events for *symbol*.

        Args:
            symbol: Trading pair.
            limit: Maximum events to return.
            direction: Filter by direction (``"inflow"`` /
                ``"outflow"`` / ``"transfer"``).  ``None`` = all.

        Returns:
            List of :class:`WhaleEvent`, most recent last.
        """
        events = self._whale_store.get(symbol)
        if not events:
            return []

        result = list(events)
        if direction is not None:
            result = [e for e in result if e.direction == direction]
        return result[-limit:]

    def get_whale_features(self, symbol: str) -> dict[str, float]:
        """Return the current whale-activity feature dict for *symbol*.

        Convenience wrapper around :meth:`_compute_whale_features`.
        """
        return self._compute_whale_features(symbol)

    def clear(self, symbol: str | None = None) -> None:
        """Reset event history.  If *symbol* is ``None``, clears all."""
        if symbol is None:
            self._whale_store.clear()
        else:
            self._whale_store.pop(symbol, None)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_whale_features(self, symbol: str) -> dict[str, float]:
        """Compute aggregate whale features for *symbol* from the
        stored event history.

        Features produced:

        * ``whale_count_1h`` -- number of whale events in last hour.
        * ``whale_inflow_usd_1h`` -- total USD of exchange inflows.
        * ``whale_outflow_usd_1h`` -- total USD of exchange outflows.
        * ``whale_net_flow_usd_1h`` -- inflow minus outflow.
        * ``whale_largest_usd_1h`` -- single largest transaction.
        """
        now = datetime.now(timezone.utc)
        lookback = 3600.0  # 1 hour

        events = self._whale_store.get(symbol)
        if not events:
            return {
                "whale_count_1h": 0.0,
                "whale_inflow_usd_1h": 0.0,
                "whale_outflow_usd_1h": 0.0,
                "whale_net_flow_usd_1h": 0.0,
                "whale_largest_usd_1h": 0.0,
            }

        count = 0
        inflow = 0.0
        outflow = 0.0
        largest = 0.0

        for ev in events:
            age = (now - ev.timestamp).total_seconds()
            if age < 0 or age > lookback:
                continue

            count += 1
            if ev.amount_usd > largest:
                largest = ev.amount_usd

            if ev.direction == "inflow":
                inflow += ev.amount_usd
            elif ev.direction == "outflow":
                outflow += ev.amount_usd

        return {
            "whale_count_1h": float(count),
            "whale_inflow_usd_1h": inflow,
            "whale_outflow_usd_1h": outflow,
            "whale_net_flow_usd_1h": inflow - outflow,
            "whale_largest_usd_1h": largest,
        }

    async def _poll_chain(self) -> None:
        """Background task that polls on-chain data sources.

        TODO: Implement real on-chain polling.  This method should:
        1. Connect to blockchain RPCs or third-party APIs.
        2. Filter for transactions above ``self._whale_threshold_usd``.
        3. Classify as inflow / outflow / transfer.
        4. Publish ``WhaleEvent`` to the event bus.
        """
        # TODO: Implement real on-chain polling loop.
        # import asyncio
        # while True:
        #     try:
        #         # Query data source
        #         # Process and filter
        #         # Publish events
        #         pass
        #     except Exception:
        #         logger.exception("Error in whale poll loop")
        #     await asyncio.sleep(30)  # Poll every 30 seconds
        pass
