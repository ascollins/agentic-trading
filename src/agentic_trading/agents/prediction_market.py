"""Prediction market intelligence agent.

Polls Polymarket via the Gamma REST API for prediction market data,
classifies markets by relevance to crypto trading, computes
per-symbol consensus scores, and publishes events for downstream
strategies, portfolio manager, and governance to consume.

Uses ``httpx`` to query ``https://gamma-api.polymarket.com/markets``
— no external CLI required.

Integration points:
- Publishes ``PredictionMarketEvent`` per market on ``intelligence.prediction``
- Publishes ``PredictionConsensus`` per symbol (aggregated directional signal)
- FeatureEngine can inject ``pm_consensus_*`` fields into FeatureVector
- PortfolioManager reads consensus for confidence adjustment
- GovernanceGate reads event_risk_level for sizing policy
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import httpx

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.config import PredictionMarketConfig
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import (
    AgentCapabilities,
    PredictionConsensus,
    PredictionMarketEvent,
)
from agentic_trading.core.ids import new_id, utc_now

logger = logging.getLogger(__name__)

# Gamma API base URL (public, no auth needed for read-only)
_GAMMA_API_BASE = "https://gamma-api.polymarket.com"
_GAMMA_MARKETS_ENDPOINT = f"{_GAMMA_API_BASE}/markets"

# Maps prediction market keywords to traded symbols
_SYMBOL_KEYWORD_MAP: dict[str, list[str]] = {
    "BTC/USDT": ["bitcoin", "btc", "crypto"],
    "ETH/USDT": ["ethereum", "eth", "crypto"],
    "SOL/USDT": ["solana", "sol", "crypto"],
    "XRP/USDT": ["xrp", "ripple", "sec", "crypto"],
    "BNB/USDT": ["bnb", "binance", "crypto"],
    "DOGE/USDT": ["doge", "dogecoin", "crypto"],
    "TRX/USDT": ["trx", "tron", "crypto"],
    "SUI/USDT": ["sui", "crypto"],
    "ADA/USDT": ["ada", "cardano", "crypto"],
    "AAVE/USDT": ["aave", "defi", "crypto"],
    "PEPE/USDT": ["pepe", "meme", "crypto"],
}

# Macro keywords that affect all crypto (risk-on / risk-off)
_MACRO_KEYWORDS: list[str] = [
    "fed", "rate", "inflation", "cpi", "fomc", "recession",
    "interest rate", "monetary policy", "quantitative",
]

_REGULATORY_KEYWORDS: list[str] = [
    "sec", "regulation", "crypto ban", "stablecoin", "cbdc",
    "enforcement", "lawsuit", "etf approval", "spot etf",
]

_GEOPOLITICAL_KEYWORDS: list[str] = [
    "war", "sanctions", "conflict", "tariff", "trade war",
    "election", "president",
]


def _classify_market(question: str) -> str:
    """Classify a prediction market question into a category."""
    q_lower = question.lower()
    for kw in _MACRO_KEYWORDS:
        if kw in q_lower:
            return "macro"
    for kw in _REGULATORY_KEYWORDS:
        if kw in q_lower:
            return "regulatory"
    for kw in _GEOPOLITICAL_KEYWORDS:
        if kw in q_lower:
            return "geopolitical"
    return "crypto_price"


def _infer_direction(question: str, probability: float) -> str:
    """Infer directional implication for crypto from a market question.

    Returns "bullish", "bearish", or "neutral".
    """
    q_lower = question.lower()

    # Price-up markets: high probability = bullish
    bullish_patterns = [
        r"(above|exceed|reach|hit|surpass)\s+\$",
        r"(price|btc|bitcoin|eth|ethereum).*\$([\d,]+)",
        r"(rate cut|rate decrease|easing)",
        r"(etf approv|spot etf)",
        r"(bull|rally|surge|breakout)",
    ]
    bearish_patterns = [
        r"(below|fall|drop|crash|decline)\s+\$",
        r"(rate hike|rate increase|tightening)",
        r"(ban|restrict|enforcement|lawsuit|sue)",
        r"(war|conflict|sanction)",
        r"(bear|crash|collapse)",
        r"(recession)",
    ]

    for pattern in bullish_patterns:
        if re.search(pattern, q_lower):
            if probability > 0.6:
                return "bullish"
            elif probability < 0.4:
                return "bearish"
            return "neutral"

    for pattern in bearish_patterns:
        if re.search(pattern, q_lower):
            if probability > 0.6:
                return "bearish"
            elif probability < 0.4:
                return "bullish"
            return "neutral"

    return "neutral"


def _affected_symbols(question: str, category: str) -> list[str]:
    """Determine which traded symbols a market affects."""
    q_lower = question.lower()

    # Macro/geopolitical events affect all crypto
    if category in ("macro", "geopolitical"):
        return list(_SYMBOL_KEYWORD_MAP.keys())

    # Check specific symbol keywords
    affected = []
    for symbol, keywords in _SYMBOL_KEYWORD_MAP.items():
        for kw in keywords:
            if kw == "crypto":
                continue  # Too generic, skip
            if kw in q_lower:
                affected.append(symbol)
                break

    # Regulatory often affects all crypto
    if category == "regulatory" and not affected:
        return list(_SYMBOL_KEYWORD_MAP.keys())

    # "crypto" keyword without specific symbols → all symbols
    if not affected and "crypto" in q_lower:
        return list(_SYMBOL_KEYWORD_MAP.keys())

    return affected


def _direction_to_score(direction: str, probability: float) -> float:
    """Convert direction and probability to a -1.0 to +1.0 consensus score."""
    # Distance from 50% (uncertainty)
    edge = abs(probability - 0.5) * 2.0  # 0.0 to 1.0

    if direction == "bullish":
        return edge
    elif direction == "bearish":
        return -edge
    return 0.0


def _extract_probability(market: dict[str, Any]) -> float:
    """Extract the YES outcome probability from a Gamma API market dict.

    Gamma markets have ``outcomePrices`` as a JSON-encoded string list,
    e.g. ``'["0.72", "0.28"]'`` for [Yes, No].  Falls back to the
    first token price if available.
    """
    # 1. outcomePrices (JSON-encoded list string)
    outcome_prices = market.get("outcomePrices")
    if outcome_prices:
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, ValueError):
                outcome_prices = None
        if isinstance(outcome_prices, list) and outcome_prices:
            try:
                return float(outcome_prices[0])
            except (ValueError, TypeError):
                pass

    # 2. Nested "price" field (flat format)
    price = market.get("price")
    if price is not None:
        try:
            return float(price)
        except (ValueError, TypeError):
            pass

    # 3. tokens[0].price  (CLOB API format)
    tokens = market.get("tokens")
    if isinstance(tokens, list) and tokens:
        try:
            return float(tokens[0].get("price", 0))
        except (ValueError, TypeError):
            pass

    return 0.0


class PredictionMarketAgent(BaseAgent):
    """Polls prediction markets and publishes intelligence events.

    Queries the Polymarket Gamma API to fetch active markets, filters by
    configured keywords and minimum volume, classifies each market by
    category and directional implication, computes per-symbol consensus
    scores, and publishes structured events.

    Parameters
    ----------
    event_bus:
        Event bus instance for publishing prediction events.
    config:
        PredictionMarketConfig with polling interval, thresholds, etc.
    symbols:
        List of traded symbols to compute consensus for.
    """

    def __init__(
        self,
        event_bus: Any,
        config: PredictionMarketConfig,
        symbols: list[str] | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id or "prediction-market",
            interval=config.poll_interval_seconds,
        )
        self._event_bus = event_bus
        self._config = config
        self._symbols = symbols or list(_SYMBOL_KEYWORD_MAP.keys())

        # TWAP state: market_id → list of (timestamp, probability)
        self._probability_history: dict[str, list[tuple[datetime, float]]] = (
            defaultdict(list)
        )
        # Latest consensus per symbol
        self._latest_consensus: dict[str, PredictionConsensus] = {}

        # Shared httpx async client (created lazily)
        self._http_client: httpx.AsyncClient | None = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.PREDICTION_MARKET

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=[],
            publishes_to=["intelligence.prediction"],
            description=(
                "Polls Polymarket Gamma API for prediction market probabilities, "
                "classifies by category, and publishes per-symbol consensus "
                "for confidence adjustment and event risk gating."
            ),
        )

    def get_consensus(self, symbol: str) -> PredictionConsensus | None:
        """Return the latest consensus for a symbol (used by PortfolioManager)."""
        return self._latest_consensus.get(symbol)

    async def start(self) -> None:
        """Start the agent and create the HTTP client."""
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        await super().start()

    async def stop(self) -> None:
        """Stop the agent and close the HTTP client."""
        await super().stop()
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _work(self) -> None:
        """Single poll cycle: fetch markets, classify, publish."""
        try:
            raw_markets = await self._fetch_markets()
            if not raw_markets:
                logger.debug("PredictionMarketAgent: no markets returned")
                return

            events = self._process_markets(raw_markets)
            consensus_map = self._compute_consensus(events)

            # Publish individual market events
            for event in events:
                await self._event_bus.publish(
                    "intelligence.prediction", event
                )

            # Publish per-symbol consensus
            for symbol, consensus in consensus_map.items():
                self._latest_consensus[symbol] = consensus
                await self._event_bus.publish(
                    "intelligence.prediction", consensus
                )

            logger.info(
                "PredictionMarketAgent: processed %d markets → %d events, "
                "%d symbol consensus updates",
                len(raw_markets),
                len(events),
                len(consensus_map),
            )

        except Exception:
            logger.exception("PredictionMarketAgent: poll cycle failed")
            raise

    async def _fetch_markets(self) -> list[dict[str, Any]]:
        """Fetch active markets from the Polymarket Gamma API.

        Queries ``GET /markets`` with keyword-based client-side filtering.
        The Gamma API supports ``active``, ``closed``, ``limit``, and
        ``offset`` query parameters but not free-text search, so we pull
        a broad set of active high-volume markets and filter locally.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
            )

        all_markets: list[dict[str, Any]] = []

        try:
            # Fetch top active markets by volume (descending).
            # Pull up to 200 markets in 2 pages to cast a wide net.
            for offset in (0, 100):
                params: dict[str, Any] = {
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "limit": 100,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false",
                }
                resp = await self._http_client.get(
                    _GAMMA_MARKETS_ENDPOINT, params=params
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    all_markets.extend(data)
                else:
                    logger.warning(
                        "PredictionMarketAgent: unexpected response type: %s",
                        type(data).__name__,
                    )
                    break

                # Stop if fewer than page size returned
                if isinstance(data, list) and len(data) < 100:
                    break

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "PredictionMarketAgent: Gamma API HTTP %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return []
        except httpx.ConnectError:
            logger.warning(
                "PredictionMarketAgent: cannot connect to Gamma API "
                "(DNS or network issue)"
            )
            return []
        except Exception:
            logger.exception("PredictionMarketAgent: error fetching markets")
            return []

        # Client-side keyword filtering: keep markets whose question
        # matches at least one configured keyword.
        keywords_lower = [kw.lower() for kw in self._config.keywords]
        filtered: list[dict[str, Any]] = []
        for m in all_markets:
            question = (m.get("question") or m.get("title") or "").lower()
            description = (m.get("description") or "").lower()
            searchable = question + " " + description

            if any(kw in searchable for kw in keywords_lower):
                filtered.append(m)

        # Deduplicate by condition_id or question
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for m in filtered:
            key = m.get("conditionId") or m.get("question", "")
            if key and key not in seen:
                seen.add(key)
                unique.append(m)

        return unique

    def _process_markets(
        self, raw_markets: list[dict[str, Any]]
    ) -> list[PredictionMarketEvent]:
        """Convert raw market data into typed events."""
        events: list[PredictionMarketEvent] = []
        now = utc_now()
        min_volume = self._config.min_volume_usd

        for market in raw_markets:
            question = market.get("question", market.get("title", ""))
            if not question:
                continue

            # Extract probability
            probability = _extract_probability(market)
            if probability <= 0 or probability >= 1.0:
                continue

            # Volume: prefer volume24hr, fall back to volumeNum or volume
            volume = float(
                market.get("volume24hr")
                or market.get("volumeNum")
                or market.get("volume")
                or 0
            )
            if volume < min_volume:
                continue

            # Market ID: conditionId is the canonical Gamma identifier
            market_id = str(
                market.get("conditionId")
                or market.get("id")
                or question[:50]
            )

            # Update TWAP
            self._update_twap(market_id, probability, now)
            twap_prob = self._get_twap(market_id)

            category = _classify_market(question)
            direction = _infer_direction(question, twap_prob)
            affected = _affected_symbols(question, category)
            consensus = _direction_to_score(direction, twap_prob)
            relevance = min(1.0, volume / 1_000_000) * 0.7 + (
                0.3 if category != "crypto_price" else 0.1
            )

            # Resolution date
            resolution = (
                market.get("endDateIso")
                or market.get("endDate")
                or market.get("resolution_date")
                or ""
            )

            event = PredictionMarketEvent(
                category=category,
                market_question=question,
                probability=round(twap_prob, 4),
                volume_usd=volume,
                direction_implication=direction,
                affected_symbols=affected,
                consensus_score=round(consensus, 4),
                confidence_in_relevance=round(relevance, 3),
                resolution_date=resolution,
                market_id=market_id,
                pm_source="polymarket",
            )
            events.append(event)

        return events

    def _update_twap(
        self, market_id: str, probability: float, now: datetime
    ) -> None:
        """Update the time-weighted average probability for a market."""
        history = self._probability_history[market_id]
        history.append((now, probability))

        # Prune old entries beyond TWAP window
        cutoff_seconds = self._config.twap_window_hours * 3600
        history[:] = [
            (t, p) for t, p in history
            if (now - t).total_seconds() < cutoff_seconds
        ]

    def _get_twap(self, market_id: str) -> float:
        """Get the time-weighted average probability for a market."""
        history = self._probability_history.get(market_id, [])
        if not history:
            return 0.0
        if len(history) == 1:
            return history[0][1]

        # Time-weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        for i in range(1, len(history)):
            dt = (history[i][0] - history[i - 1][0]).total_seconds()
            if dt > 0:
                weighted_sum += history[i][1] * dt
                total_weight += dt

        if total_weight > 0:
            return weighted_sum / total_weight
        return history[-1][1]

    def _compute_consensus(
        self, events: list[PredictionMarketEvent]
    ) -> dict[str, PredictionConsensus]:
        """Aggregate individual market events into per-symbol consensus."""
        # Group events by affected symbol
        by_symbol: dict[str, list[PredictionMarketEvent]] = defaultdict(list)
        for event in events:
            for symbol in event.affected_symbols:
                if symbol in self._symbols:
                    by_symbol[symbol].append(event)

        consensus_map: dict[str, PredictionConsensus] = {}

        for symbol, sym_events in by_symbol.items():
            if not sym_events:
                continue

            # Volume-weighted consensus score
            total_volume = sum(e.volume_usd for e in sym_events)
            if total_volume <= 0:
                continue

            weighted_score = sum(
                e.consensus_score * e.volume_usd for e in sym_events
            ) / total_volume

            avg_volume = total_volume / len(sym_events)
            categories = list({e.category for e in sym_events})

            # Event risk: check if any market has uncertain outcome
            # (probability between 0.35-0.65 = genuinely uncertain binary event)
            lo, hi = self._config.event_risk_uncertainty_range
            event_risk = 0.0
            hours_to_event = -1.0
            for e in sym_events:
                if lo <= e.probability <= hi:
                    # Uncertain event — higher risk
                    risk_score = 1.0 - 2.0 * abs(e.probability - 0.5)
                    event_risk = max(event_risk, risk_score)

            consensus = PredictionConsensus(
                symbol=symbol,
                consensus_score=round(weighted_score, 4),
                market_count=len(sym_events),
                avg_volume_usd=round(avg_volume, 2),
                categories=categories,
                event_risk_level=round(event_risk, 3),
                hours_to_nearest_event=hours_to_event,
                details={
                    "markets": [
                        {
                            "question": e.market_question[:80],
                            "probability": e.probability,
                            "direction": e.direction_implication,
                            "volume": e.volume_usd,
                        }
                        for e in sym_events[:5]
                    ]
                },
            )
            consensus_map[symbol] = consensus

        return consensus_map
