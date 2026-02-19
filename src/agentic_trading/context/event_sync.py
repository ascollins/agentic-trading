"""Fact Table Event Sync — keeps the fact table updated from the event bus.

Subscribes to event bus topics (market.candle, state.*, risk.*, system.*)
and automatically updates the ``FactTable`` as events flow. Only active
in paper/live mode — backtest updates facts directly during replay.
"""

from __future__ import annotations

import logging
from typing import Any

from .fact_table import FactTable

logger = logging.getLogger(__name__)


class FactTableEventSync:
    """Subscribes to event bus topics and updates the FactTable.

    No agent needs to manually update the fact table — this runs as
    a background subscriber keeping it in sync with the live event
    stream.

    Parameters
    ----------
    fact_table:
        The FactTable to update.
    event_bus:
        The legacy IEventBus to subscribe to.
    """

    def __init__(self, fact_table: FactTable, event_bus: Any) -> None:
        self._ft = fact_table
        self._bus = event_bus

    async def start(self) -> None:
        """Subscribe to relevant event bus topics."""
        await self._bus.subscribe(
            "market.candle", "fact_table_sync", self._on_candle
        )
        await self._bus.subscribe(
            "state.position", "fact_table_sync", self._on_position
        )
        await self._bus.subscribe(
            "state.balance", "fact_table_sync", self._on_balance
        )
        await self._bus.subscribe(
            "risk.circuit_breaker", "fact_table_sync", self._on_circuit_breaker
        )
        await self._bus.subscribe(
            "system.kill_switch", "fact_table_sync", self._on_kill_switch
        )
        logger.info("FactTableEventSync started")

    async def _on_candle(self, event: Any) -> None:
        """Update price from candle close."""
        try:
            self._ft.update_price(
                event.symbol,
                last=float(event.close),
                updated_at=event.timestamp,
            )
        except Exception:
            logger.debug("Failed to update price from candle event")

    async def _on_position(self, event: Any) -> None:
        """Update price from position mark price."""
        try:
            if hasattr(event, "mark_price") and event.mark_price:
                self._ft.update_price(
                    event.symbol,
                    last=float(event.mark_price),
                )
        except Exception:
            logger.debug("Failed to update price from position event")

    async def _on_balance(self, event: Any) -> None:
        """Update portfolio equity from balance events."""
        try:
            if hasattr(event, "total"):
                self._ft.update_portfolio_fields(
                    total_equity=float(event.total),
                )
        except Exception:
            logger.debug("Failed to update portfolio from balance event")

    async def _on_circuit_breaker(self, event: Any) -> None:
        """Track tripped circuit breakers."""
        try:
            if hasattr(event, "tripped") and event.tripped:
                risk = self._ft.get_risk()
                tripped = list(risk.circuit_breakers_tripped)
                breaker = str(
                    event.breaker_type.value
                    if hasattr(event.breaker_type, "value")
                    else event.breaker_type
                )
                if breaker not in tripped:
                    tripped.append(breaker)
                    self._ft.update_risk(circuit_breakers_tripped=tripped)
        except Exception:
            logger.debug("Failed to update risk from circuit breaker event")

    async def _on_kill_switch(self, event: Any) -> None:
        """Track kill switch activation."""
        try:
            if hasattr(event, "activated"):
                self._ft.update_risk(kill_switch_active=event.activated)
        except Exception:
            logger.debug("Failed to update risk from kill switch event")
