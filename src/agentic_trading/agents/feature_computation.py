"""FeatureComputationAgent — dedicated feature computation (design spec §7.2).

Extracted from MarketIntelligenceAgent to enforce separation of concerns:
    - MarketIntelligenceAgent: data ingestion (feeds, candles)
    - FeatureComputationAgent: subscribes to ``market.candle``, computes
      indicators via FeatureEngine, publishes ``feature.vector``

This is an event-driven agent (no periodic loop).
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentCapabilities
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


class FeatureComputationAgent(BaseAgent):
    """Dedicated agent for feature vector computation.

    Wraps the FeatureEngine and subscribes to ``market.candle`` events
    to produce ``feature.vector`` events.

    Parameters
    ----------
    event_bus:
        Event bus for subscriptions and publishing.
    indicator_config:
        Optional configuration for the FeatureEngine indicators.
    agent_id:
        Optional agent identifier.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        indicator_config: dict[str, Any] | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id or "feature_computation", interval=0)
        self._event_bus = event_bus
        self._indicator_config = indicator_config
        self._feature_engine: Any = None

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.FEATURE_COMPUTATION

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["market.candle"],
            publishes_to=["feature.vector"],
            description="Computes technical features from candle data",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        from agentic_trading.features.engine import FeatureEngine

        self._feature_engine = FeatureEngine(
            event_bus=self._event_bus,
            indicator_config=self._indicator_config,
        )
        await self._feature_engine.start()
        logger.info("FeatureComputationAgent: FeatureEngine started")

    async def _on_stop(self) -> None:
        if self._feature_engine is not None:
            await self._feature_engine.stop()
            logger.info("FeatureComputationAgent: FeatureEngine stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_engine(self) -> Any:
        """Access the underlying FeatureEngine for direct use."""
        return self._feature_engine
