"""Market Intelligence Agent.

Wraps FeedManager, CandleBuilder, and FeatureEngine into a single
agent responsible for ingesting market data, computing features,
and publishing FeatureVector events.

This is an event-driven agent (no periodic loop) -- it subscribes
to market.candle events via the FeatureEngine and publishes
feature.vector events.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentCapabilities
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


class MarketIntelligenceAgent(BaseAgent):
    """Ingests market data and produces feature vectors.

    Orchestrates:
    - FeedManager (live WebSocket feeds)
    - CandleBuilder (timeframe aggregation)
    - FeatureEngine (indicator computation)

    Usage::

        agent = MarketIntelligenceAgent(
            event_bus=event_bus,
            exchange_configs=settings.exchanges,
            symbols=["BTC/USDT", "ETH/USDT"],
        )
        await agent.start()
    """

    def __init__(
        self,
        event_bus: IEventBus,
        exchange_configs: list[Any] | None = None,
        symbols: list[str] | None = None,
        indicator_config: dict[str, Any] | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, interval=0)
        self._event_bus = event_bus
        self._exchange_configs = exchange_configs or []
        self._symbols = symbols or []
        self._indicator_config = indicator_config

        # Components -- created during start
        self._feature_engine: Any = None
        self._feed_manager: Any = None
        self._candle_builder: Any = None

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.MARKET_INTELLIGENCE

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["market.candle"],
            publishes_to=["feature.vector"],
            description="Ingests market data and computes technical features",
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
        logger.info(
            "MarketIntelligenceAgent: FeatureEngine started",
        )

        # Start live feeds if exchange configs are provided
        if self._exchange_configs:
            try:
                from agentic_trading.data.candle_builder import CandleBuilder
                from agentic_trading.data.feed_manager import FeedManager

                self._candle_builder = CandleBuilder(
                    event_bus=self._event_bus,
                )
                self._feed_manager = FeedManager(
                    event_bus=self._event_bus,
                    candle_builder=self._candle_builder,
                    exchange_configs=self._exchange_configs,
                    symbols=self._symbols,
                )
                await self._feed_manager.start()
                logger.info(
                    "MarketIntelligenceAgent: FeedManager started "
                    "(%d tasks for %s)",
                    self._feed_manager.active_task_count,
                    self._symbols,
                )
            except Exception:
                logger.warning(
                    "MarketIntelligenceAgent: FeedManager start failed",
                    exc_info=True,
                )
                self._feed_manager = None
        else:
            logger.info(
                "MarketIntelligenceAgent: no exchange configs, "
                "running without live feeds",
            )

    async def _on_stop(self) -> None:
        if self._feed_manager is not None:
            await self._feed_manager.stop()
            logger.info("MarketIntelligenceAgent: FeedManager stopped")
        if self._feature_engine is not None:
            await self._feature_engine.stop()
            logger.info("MarketIntelligenceAgent: FeatureEngine stopped")

    # ------------------------------------------------------------------
    # Public API (delegates to FeatureEngine)
    # ------------------------------------------------------------------

    @property
    def feature_engine(self) -> Any:
        """Access the underlying FeatureEngine for direct use."""
        return self._feature_engine

    @property
    def feed_manager(self) -> Any:
        """Access the underlying FeedManager (may be None)."""
        return self._feed_manager
