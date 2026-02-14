"""News sentiment feature scaffold.

Provides the interface for ingesting news headlines, computing sentiment
scores, and producing time-decayed impact features that strategies can
consume.

**Status: Scaffold** -- the NLP / sentiment classification core is
marked as TODO.  The full public API and event-bus integration are
implemented so that downstream consumers can code against a stable
interface today.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from agentic_trading.core.enums import Timeframe
from agentic_trading.core.events import BaseEvent, FeatureVector, NewsEvent
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)

# Maximum number of news events kept in memory per symbol.
_DEFAULT_HISTORY_SIZE = 200

# Default half-life for exponential time-decay (seconds).
_DEFAULT_DECAY_HALF_LIFE = 1800  # 30 minutes


class SentimentEngine:
    """Ingests news events, scores sentiment, and computes time-decayed
    impact features per symbol.

    **Scaffold implementation** -- the actual NLP classification in
    :meth:`_classify_sentiment` is a placeholder that returns neutral
    sentiment.  Replace it with a real model (e.g. FinBERT, OpenAI, or
    a custom classifier) when ready.

    Usage (event-bus driven)::

        engine = SentimentEngine(event_bus=bus)
        await engine.start()

    Usage (direct)::

        engine = SentimentEngine()
        news = engine.parse_news(
            headline="SEC approves spot Bitcoin ETF",
            body="The Securities and Exchange Commission ...",
        )
        score = engine.get_impact_score("BTC/USDT", lookback_seconds=3600)
    """

    def __init__(
        self,
        event_bus: IEventBus | None = None,
        history_size: int = _DEFAULT_HISTORY_SIZE,
        decay_half_life: float = _DEFAULT_DECAY_HALF_LIFE,
    ) -> None:
        self._event_bus = event_bus
        self._history_size = history_size
        self._decay_half_life = decay_half_life

        # Per-symbol store of recent news events.
        # Key: symbol -> deque of NewsEvent
        self._news_store: dict[str, deque[NewsEvent]] = defaultdict(
            lambda: deque(maxlen=self._history_size)
        )

    # ------------------------------------------------------------------
    # Event bus lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to news events on the event bus."""
        if self._event_bus is None:
            logger.warning(
                "SentimentEngine started without event bus - direct mode only"
            )
            return

        await self._event_bus.subscribe(
            topic="feature.news",
            group="sentiment_engine",
            handler=self._handle_news_event,
        )
        logger.info("SentimentEngine subscribed to feature.news")

    async def stop(self) -> None:
        """Clean up (currently a no-op)."""
        logger.info("SentimentEngine stopped")

    async def _handle_news_event(self, event: BaseEvent) -> None:
        """Internal handler for incoming ``NewsEvent`` instances."""
        if not isinstance(event, NewsEvent):
            return

        # Store the event for each mentioned symbol.
        for symbol in event.symbols:
            self._news_store[symbol].append(event)

        # Publish a feature vector with impact scores for each symbol.
        if self._event_bus is not None:
            for symbol in event.symbols:
                score = self.get_impact_score(symbol, lookback_seconds=3600)
                fv = FeatureVector(
                    symbol=symbol,
                    timeframe=Timeframe.M1,
                    features={
                        "news_impact_1h": score,
                        "news_count_1h": float(
                            self._count_recent(symbol, lookback_seconds=3600)
                        ),
                        "news_latest_sentiment": event.sentiment,
                        "news_latest_urgency": event.urgency,
                    },
                    source_module="features.sentiment",
                )
                await self._event_bus.publish("feature.vector", fv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_news(
        self,
        headline: str,
        body: str = "",
        source: str = "",
        symbols: list[str] | None = None,
    ) -> NewsEvent:
        """Parse a raw news item into a :class:`NewsEvent`.

        Performs symbol extraction and sentiment classification.

        Args:
            headline: News headline text.
            body: Full article body (optional).
            source: Name of the news source / feed.
            symbols: Explicit list of symbols mentioned.  If ``None``,
                the engine will attempt to extract them from the text.

        Returns:
            A populated :class:`NewsEvent`.
        """
        detected_symbols = symbols if symbols is not None else self._extract_symbols(headline, body)

        sentiment, urgency = self._classify_sentiment(headline, body)

        entities = self._extract_entities(headline, body)

        news = NewsEvent(
            headline=headline,
            source=source,
            symbols=detected_symbols,
            sentiment=sentiment,
            urgency=urgency,
            entities=entities,
            decay_seconds=int(self._decay_half_life * 2),
        )

        # Store in per-symbol history.
        for sym in detected_symbols:
            self._news_store[sym].append(news)

        return news

    def get_impact_score(
        self,
        symbol: str,
        lookback_seconds: int = 3600,
    ) -> float:
        """Compute a time-decayed aggregate sentiment impact score for
        *symbol* over the last *lookback_seconds*.

        The score is a weighted sum of individual news sentiments where
        weights decay exponentially based on age::

            weight(age) = exp(-age * ln(2) / half_life)

        Args:
            symbol: Trading pair to query.
            lookback_seconds: How far back to look (seconds).

        Returns:
            Aggregate impact score.  Positive = bullish sentiment,
            negative = bearish.  Returns ``0.0`` if no recent news.
        """
        now = datetime.now(timezone.utc)
        events = self._news_store.get(symbol)
        if not events:
            return 0.0

        decay_constant = math.log(2) / self._decay_half_life
        total_weight = 0.0
        weighted_sentiment = 0.0

        for ev in events:
            age = (now - ev.timestamp).total_seconds()
            if age < 0:
                age = 0.0
            if age > lookback_seconds:
                continue

            weight = math.exp(-decay_constant * age)
            # Boost weight by urgency (urgency 1.0 doubles the weight).
            weight *= 1.0 + ev.urgency
            weighted_sentiment += ev.sentiment * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sentiment / total_weight

    def get_recent_news(
        self, symbol: str, limit: int = 10
    ) -> list[NewsEvent]:
        """Return the most recent news events for *symbol*."""
        events = self._news_store.get(symbol)
        if not events:
            return []
        return list(events)[-limit:]

    def clear(self, symbol: str | None = None) -> None:
        """Reset news history.  If *symbol* is ``None``, clears all."""
        if symbol is None:
            self._news_store.clear()
        else:
            self._news_store.pop(symbol, None)

    # ------------------------------------------------------------------
    # Internal methods (stubs / scaffolds)
    # ------------------------------------------------------------------

    def _classify_sentiment(
        self, headline: str, body: str
    ) -> tuple[float, float]:
        """Classify headline + body into (sentiment, urgency).

        TODO: Replace with a real NLP model.  Candidates:
            - FinBERT (HuggingFace transformers)
            - OpenAI / Anthropic API call
            - Custom distilled model for crypto-specific jargon

        Returns:
            Tuple of:
            - sentiment: float in [-1.0, 1.0] (bearish to bullish).
            - urgency: float in [0.0, 1.0].
        """
        # TODO: Implement real sentiment classification.
        # Placeholder returns neutral sentiment with low urgency.
        logger.debug(
            "Sentiment classification is a scaffold - returning neutral "
            "for headline: %s",
            headline[:80],
        )
        return 0.0, 0.0

    def _extract_symbols(
        self, headline: str, body: str
    ) -> list[str]:
        """Extract trading symbols mentioned in the text.

        TODO: Replace with a proper NER or keyword-matching pipeline
        that maps token names / tickers to canonical symbols.

        Returns:
            List of symbol strings, e.g. ``["BTC/USDT", "ETH/USDT"]``.
        """
        # TODO: Implement real symbol extraction.
        # Placeholder: simple keyword scan for common tickers.
        known_tickers: dict[str, str] = {
            "BTC": "BTC/USDT",
            "Bitcoin": "BTC/USDT",
            "ETH": "ETH/USDT",
            "Ethereum": "ETH/USDT",
            "SOL": "SOL/USDT",
            "Solana": "SOL/USDT",
            "XRP": "XRP/USDT",
            "DOGE": "DOGE/USDT",
            "ADA": "ADA/USDT",
            "AVAX": "AVAX/USDT",
            "MATIC": "MATIC/USDT",
            "DOT": "DOT/USDT",
            "LINK": "LINK/USDT",
        }
        text = f"{headline} {body}"
        found: list[str] = []
        seen: set[str] = set()
        for token, symbol in known_tickers.items():
            if token in text and symbol not in seen:
                found.append(symbol)
                seen.add(symbol)
        return found

    def _extract_entities(
        self, headline: str, body: str
    ) -> list[str]:
        """Extract named entities (people, organizations, etc.).

        TODO: Replace with a proper NER model.

        Returns:
            List of entity strings.
        """
        # TODO: Implement real NER extraction.
        return []

    def _count_recent(
        self, symbol: str, lookback_seconds: int
    ) -> int:
        """Count news events for *symbol* within *lookback_seconds*."""
        now = datetime.now(timezone.utc)
        events = self._news_store.get(symbol)
        if not events:
            return 0
        count = 0
        for ev in events:
            age = (now - ev.timestamp).total_seconds()
            if 0 <= age <= lookback_seconds:
                count += 1
        return count
