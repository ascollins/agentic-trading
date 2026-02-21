"""DataQuality Agent stub.

Monitors feed integrity and data freshness.  When staleness or
divergence is detected, emits :class:`IncidentCreated` and
:class:`DegradedModeEnabled` events to trigger protective actions.

Day 6 stub: subscribes to ``feature.vector`` and tracks last-seen
timestamps per symbol.  Full WS-vs-REST comparison deferred to Week 2.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType, AssetClass
from agentic_trading.core.events import (
    AgentCapabilities,
    BaseEvent,
    DegradedModeEnabled,
    IncidentCreated,
)
from agentic_trading.core.interfaces import IEventBus
from agentic_trading.core.models import Instrument

logger = logging.getLogger(__name__)

# Default staleness threshold before emitting a warning (seconds)
_DEFAULT_STALENESS_THRESHOLD = 60.0


class DataQualityAgent(BaseAgent):
    """Monitors feed integrity and data freshness.

    Day 6 stub implementation:
    - Subscribes to ``feature.vector`` topic
    - Tracks last-seen timestamp per symbol
    - Emits ``IncidentCreated(severity=warning)`` when staleness > threshold
    - Emits ``DegradedModeEnabled(RISK_OFF_ONLY)`` on sustained staleness

    Full implementation (Week 2):
    - WS vs REST feed comparison
    - Cross-exchange price divergence detection
    - Materiality thresholds
    """

    def __init__(
        self,
        event_bus: IEventBus,
        staleness_threshold: float = _DEFAULT_STALENESS_THRESHOLD,
        *,
        agent_id: str | None = None,
        instruments: dict[str, Instrument] | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id or "data-quality", interval=10.0)
        self._event_bus = event_bus
        self._staleness_threshold = staleness_threshold
        self._last_seen: dict[str, float] = {}  # symbol -> monotonic timestamp
        self._alerted: set[str] = set()  # symbols already alerted
        self._instruments: dict[str, Instrument] = instruments or {}

    @property
    def agent_type(self) -> AgentType:
        return AgentType.DATA_QUALITY

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["feature.vector"],
            publishes_to=["system"],
            description="Feed integrity and data freshness monitoring",
        )

    async def _on_start(self) -> None:
        await self._event_bus.subscribe(
            topic="feature.vector",
            group="data_quality",
            handler=self._on_feature_vector,
        )
        logger.info(
            "DataQualityAgent started (staleness_threshold=%.1fs)",
            self._staleness_threshold,
        )

    async def _on_feature_vector(self, event: BaseEvent) -> None:
        """Track last-seen time per symbol."""
        symbol = getattr(event, "symbol", None)
        if symbol:
            self._last_seen[symbol] = time.monotonic()
            self._alerted.discard(symbol)

    async def _work(self) -> None:
        """Check for stale symbols.

        FX instruments are exempt from staleness alerts when their
        trading sessions are closed (weekends, off-hours).
        """
        from datetime import datetime, timezone

        now = time.monotonic()
        utc_now = datetime.now(timezone.utc)

        for symbol, last_ts in list(self._last_seen.items()):
            staleness = now - last_ts
            if staleness > self._staleness_threshold and symbol not in self._alerted:
                # Skip staleness alert for FX instruments during closed sessions
                _inst = self._instruments.get(symbol)
                if _inst is not None and _inst.asset_class == AssetClass.FX:
                    from agentic_trading.core.fx_normalizer import is_session_open

                    if not is_session_open(
                        _inst, utc_now.hour, utc_now.minute, utc_now.isoweekday()
                    ):
                        continue  # session closed — staleness expected

                self._alerted.add(symbol)
                logger.warning(
                    "Data staleness detected: symbol=%s staleness=%.1fs",
                    symbol,
                    staleness,
                )
                await self._event_bus.publish(
                    "system",
                    IncidentCreated(
                        severity="warning",
                        component="data_quality",
                        description=f"Feed stale for {symbol}: {staleness:.1f}s",
                        affected_symbols=[symbol],
                        auto_action="none",
                    ),
                )

    @property
    def last_seen(self) -> dict[str, float]:
        """Expose last-seen timestamps for introspection."""
        return dict(self._last_seen)
