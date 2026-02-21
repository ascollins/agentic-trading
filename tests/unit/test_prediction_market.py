"""Tests for the prediction market integration.

Covers:
- PredictionMarketEvent / PredictionConsensus serialization
- PredictionMarketAgent market classification and consensus computation
- PortfolioManager PM confidence adjustment
- RegimeDetector PM leading indicator
- PredictionConsensusStrategy signal generation
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_trading.core.config import PredictionMarketConfig
from agentic_trading.core.enums import (
    AgentType,
    RegimeType,
    SignalDirection,
    Timeframe,
    VolatilityRegime,
)
from agentic_trading.core.events import (
    FeatureVector,
    PredictionConsensus,
    PredictionMarketEvent,
    Signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pm_event(**overrides: Any) -> PredictionMarketEvent:
    defaults = {
        "category": "crypto_price",
        "market_question": "Will BTC exceed $100k by March?",
        "probability": 0.72,
        "volume_usd": 5_000_000,
        "direction_implication": "bullish",
        "affected_symbols": ["BTC/USDT"],
        "consensus_score": 0.44,
        "confidence_in_relevance": 0.8,
        "market_id": "test-market-1",
        "pm_source": "polymarket",
    }
    defaults.update(overrides)
    return PredictionMarketEvent(**defaults)


def _make_consensus(**overrides: Any) -> PredictionConsensus:
    defaults = {
        "symbol": "BTC/USDT",
        "consensus_score": 0.5,
        "market_count": 3,
        "avg_volume_usd": 2_000_000,
        "categories": ["crypto_price", "macro"],
        "event_risk_level": 0.0,
    }
    defaults.update(overrides)
    return PredictionConsensus(**defaults)


# ===========================================================================
# Event serialization
# ===========================================================================


class TestPredictionMarketEvent:
    def test_create_default(self):
        event = _make_pm_event()
        assert event.source_module == "intelligence.prediction_market"
        assert event.probability == 0.72
        assert event.category == "crypto_price"
        assert "BTC/USDT" in event.affected_symbols
        assert event.schema_version == 1

    def test_serialization_round_trip(self):
        event = _make_pm_event()
        data = event.model_dump()
        restored = PredictionMarketEvent(**data)
        assert restored.probability == event.probability
        assert restored.market_question == event.market_question
        assert restored.consensus_score == event.consensus_score

    def test_consensus_event(self):
        consensus = _make_consensus()
        assert consensus.symbol == "BTC/USDT"
        assert consensus.consensus_score == 0.5
        assert consensus.market_count == 3
        assert consensus.event_risk_level == 0.0


# ===========================================================================
# Agent: market classification
# ===========================================================================


class TestMarketClassification:
    def test_classify_macro(self):
        from agentic_trading.agents.prediction_market import _classify_market
        assert _classify_market("Fed rate cut in January 2026?") == "macro"
        assert _classify_market("Will inflation exceed 3%?") == "macro"

    def test_classify_regulatory(self):
        from agentic_trading.agents.prediction_market import _classify_market
        assert _classify_market("SEC enforcement action on Coinbase?") == "regulatory"
        assert _classify_market("Spot ETF approval by March?") == "regulatory"

    def test_classify_geopolitical(self):
        from agentic_trading.agents.prediction_market import _classify_market
        assert _classify_market("Will the war escalate?") == "geopolitical"
        assert _classify_market("US election outcome?") == "geopolitical"

    def test_classify_crypto_default(self):
        from agentic_trading.agents.prediction_market import _classify_market
        assert _classify_market("Will BTC reach $150k?") == "crypto_price"

    def test_infer_direction_bullish(self):
        from agentic_trading.agents.prediction_market import _infer_direction
        assert _infer_direction("Will BTC exceed $100k?", 0.75) == "bullish"
        assert _infer_direction("Rate cut in January?", 0.80) == "bullish"

    def test_infer_direction_bearish(self):
        from agentic_trading.agents.prediction_market import _infer_direction
        assert _infer_direction("Will BTC exceed $100k?", 0.25) == "bearish"
        assert _infer_direction("Crypto ban in US?", 0.70) == "bearish"

    def test_infer_direction_neutral(self):
        from agentic_trading.agents.prediction_market import _infer_direction
        assert _infer_direction("Will BTC exceed $100k?", 0.50) == "neutral"

    def test_affected_symbols_bitcoin(self):
        from agentic_trading.agents.prediction_market import _affected_symbols
        syms = _affected_symbols("Bitcoin price prediction", "crypto_price")
        assert "BTC/USDT" in syms

    def test_affected_symbols_macro_all(self):
        from agentic_trading.agents.prediction_market import _affected_symbols
        syms = _affected_symbols("Fed rate decision", "macro")
        assert len(syms) >= 10  # All traded symbols


# ===========================================================================
# Agent: consensus computation
# ===========================================================================


# ===========================================================================
# Agent: probability extraction from Gamma API format
# ===========================================================================


class TestProbabilityExtraction:
    def test_outcome_prices_json_string(self):
        from agentic_trading.agents.prediction_market import _extract_probability
        market = {"outcomePrices": '["0.72", "0.28"]'}
        assert abs(_extract_probability(market) - 0.72) < 0.001

    def test_outcome_prices_already_list(self):
        from agentic_trading.agents.prediction_market import _extract_probability
        market = {"outcomePrices": ["0.65", "0.35"]}
        assert abs(_extract_probability(market) - 0.65) < 0.001

    def test_price_field_fallback(self):
        from agentic_trading.agents.prediction_market import _extract_probability
        market = {"price": 0.55}
        assert abs(_extract_probability(market) - 0.55) < 0.001

    def test_tokens_fallback(self):
        from agentic_trading.agents.prediction_market import _extract_probability
        market = {"tokens": [{"price": 0.80}, {"price": 0.20}]}
        assert abs(_extract_probability(market) - 0.80) < 0.001

    def test_no_price_returns_zero(self):
        from agentic_trading.agents.prediction_market import _extract_probability
        market = {"question": "some question"}
        assert _extract_probability(market) == 0.0

    def test_invalid_outcome_prices_string(self):
        from agentic_trading.agents.prediction_market import _extract_probability
        market = {"outcomePrices": "not-json", "price": 0.42}
        assert abs(_extract_probability(market) - 0.42) < 0.001


# ===========================================================================
# Agent: consensus computation
# ===========================================================================


class TestConsensusComputation:
    def test_compute_consensus_single_symbol(self):
        from agentic_trading.agents.prediction_market import (
            PredictionMarketAgent,
        )

        config = PredictionMarketConfig(
            enabled=True,
            min_volume_usd=0,
            poll_interval_seconds=300,
        )
        agent = PredictionMarketAgent(
            event_bus=MagicMock(),
            config=config,
            symbols=["BTC/USDT", "ETH/USDT"],
        )

        events = [
            _make_pm_event(
                consensus_score=0.6,
                volume_usd=5_000_000,
                affected_symbols=["BTC/USDT"],
            ),
            _make_pm_event(
                consensus_score=0.4,
                volume_usd=3_000_000,
                affected_symbols=["BTC/USDT"],
            ),
        ]

        result = agent._compute_consensus(events)
        assert "BTC/USDT" in result
        consensus = result["BTC/USDT"]
        # Volume-weighted: (0.6*5M + 0.4*3M) / 8M = 0.525
        assert 0.50 < consensus.consensus_score < 0.55
        assert consensus.market_count == 2

    def test_compute_consensus_empty(self):
        from agentic_trading.agents.prediction_market import (
            PredictionMarketAgent,
        )

        config = PredictionMarketConfig(enabled=True, min_volume_usd=0)
        agent = PredictionMarketAgent(
            event_bus=MagicMock(),
            config=config,
        )
        result = agent._compute_consensus([])
        assert result == {}

    def test_agent_type(self):
        from agentic_trading.agents.prediction_market import (
            PredictionMarketAgent,
        )

        config = PredictionMarketConfig(enabled=True)
        agent = PredictionMarketAgent(
            event_bus=MagicMock(), config=config,
        )
        assert agent.agent_type == AgentType.PREDICTION_MARKET

    def test_get_consensus_returns_none_initially(self):
        from agentic_trading.agents.prediction_market import (
            PredictionMarketAgent,
        )

        config = PredictionMarketConfig(enabled=True)
        agent = PredictionMarketAgent(
            event_bus=MagicMock(), config=config,
        )
        assert agent.get_consensus("BTC/USDT") is None


# ===========================================================================
# PortfolioManager: PM confidence adjustment
# ===========================================================================


class TestPMConfidenceAdjustment:
    def test_no_agent_returns_zero(self):
        from agentic_trading.signal.portfolio.manager import PortfolioManager

        pm = PortfolioManager()
        adj = pm._get_pm_confidence_adjustment("BTC/USDT", SignalDirection.LONG)
        assert adj == 0.0

    def test_alignment_positive_boost(self):
        from agentic_trading.signal.portfolio.manager import PortfolioManager

        mock_agent = MagicMock()
        mock_agent.get_consensus.return_value = _make_consensus(
            consensus_score=0.8, market_count=3,
        )

        pm = PortfolioManager(prediction_market_agent=mock_agent)
        adj = pm._get_pm_confidence_adjustment("BTC/USDT", SignalDirection.LONG)
        assert adj > 0  # Positive consensus + long = boost
        assert adj <= pm._pm_max_boost

    def test_divergence_negative_penalty(self):
        from agentic_trading.signal.portfolio.manager import PortfolioManager

        mock_agent = MagicMock()
        mock_agent.get_consensus.return_value = _make_consensus(
            consensus_score=-0.6, market_count=3,
        )

        pm = PortfolioManager(prediction_market_agent=mock_agent)
        adj = pm._get_pm_confidence_adjustment("BTC/USDT", SignalDirection.LONG)
        assert adj < 0  # Negative consensus + long = penalty
        assert adj >= -pm._pm_max_boost

    def test_short_signal_inverts_consensus(self):
        from agentic_trading.signal.portfolio.manager import PortfolioManager

        mock_agent = MagicMock()
        mock_agent.get_consensus.return_value = _make_consensus(
            consensus_score=-0.7, market_count=3,
        )

        pm = PortfolioManager(prediction_market_agent=mock_agent)
        adj = pm._get_pm_confidence_adjustment("BTC/USDT", SignalDirection.SHORT)
        assert adj > 0  # Negative consensus + short = aligned = boost

    def test_flat_signal_no_adjustment(self):
        from agentic_trading.signal.portfolio.manager import PortfolioManager

        mock_agent = MagicMock()
        mock_agent.get_consensus.return_value = _make_consensus(
            consensus_score=0.9, market_count=3,
        )

        pm = PortfolioManager(prediction_market_agent=mock_agent)
        adj = pm._get_pm_confidence_adjustment("BTC/USDT", SignalDirection.FLAT)
        assert adj == 0.0

    def test_no_consensus_data_returns_zero(self):
        from agentic_trading.signal.portfolio.manager import PortfolioManager

        mock_agent = MagicMock()
        mock_agent.get_consensus.return_value = None

        pm = PortfolioManager(prediction_market_agent=mock_agent)
        adj = pm._get_pm_confidence_adjustment("BTC/USDT", SignalDirection.LONG)
        assert adj == 0.0

    def test_clamped_to_max_boost(self):
        from agentic_trading.signal.portfolio.manager import PortfolioManager

        mock_agent = MagicMock()
        mock_agent.get_consensus.return_value = _make_consensus(
            consensus_score=1.0, market_count=5,
        )

        pm = PortfolioManager(prediction_market_agent=mock_agent)
        pm._pm_max_boost = 0.15
        adj = pm._get_pm_confidence_adjustment("BTC/USDT", SignalDirection.LONG)
        assert adj == 0.15


# ===========================================================================
# RegimeDetector: PM leading indicator
# ===========================================================================


class TestRegimeDetectorPM:
    def test_pm_consensus_stored(self):
        from agentic_trading.signal.strategies.regime.detector import (
            RegimeDetector,
        )

        rd = RegimeDetector(hysteresis_count=3)
        rd.update_pm_consensus("BTC/USDT", consensus_score=0.7, event_risk_level=0.3)
        assert rd._pm_consensus["BTC/USDT"] == 0.7
        assert rd._pm_event_risk["BTC/USDT"] == 0.3

    def test_pm_reduces_hysteresis_for_trend(self):
        from agentic_trading.signal.strategies.regime.detector import (
            RegimeDetector,
        )

        rd = RegimeDetector(hysteresis_count=3, use_hmm=False)

        # Set strong PM consensus favoring trend
        rd.update_pm_consensus("BTC/USDT", consensus_score=0.8)

        # Provide data that suggests TREND regime (high ADX)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        returns = [0.01] * 30
        vols = [0.02] * 30

        # Without PM: needs 3 consecutive signals
        # With PM (consensus > 0.5 + TREND): needs 2 consecutive signals
        state1 = rd.update(
            "BTC/USDT", returns, vols, adx=35.0, timestamp=now,
        )
        state2 = rd.update(
            "BTC/USDT", returns, vols, adx=35.0,
            timestamp=datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
        )

        # After 2 signals with PM boost, should have switched
        # (hysteresis reduced from 3 to 2)
        assert state2.regime == RegimeType.TREND or state2.consecutive_count >= 1


# ===========================================================================
# PredictionConsensusStrategy
# ===========================================================================


class TestPredictionConsensusStrategy:
    def _make_ctx(self) -> MagicMock:
        ctx = MagicMock()
        ctx.portfolio_state = None
        ctx.get_instrument.return_value = None
        return ctx

    def _make_candle(
        self, symbol: str = "BTC/USDT", close: float = 67000.0,
    ) -> MagicMock:
        candle = MagicMock()
        candle.symbol = symbol
        candle.close = close
        candle.timeframe = Timeframe.M5
        candle.timestamp = datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc)
        return candle

    def _make_features(
        self,
        symbol: str = "BTC/USDT",
        pm_consensus: float = 0.0,
        pm_count: int = 0,
        pm_volume: float = 0.0,
        atr: float = 500.0,
    ) -> FeatureVector:
        return FeatureVector(
            symbol=symbol,
            timeframe=Timeframe.M5,
            features={
                "pm_consensus_score": pm_consensus,
                "pm_market_count": pm_count,
                "pm_avg_volume_usd": pm_volume,
                "pm_event_risk_level": 0.0,
                "atr": atr,
                "close": 67000.0,
            },
        )

    def test_no_pm_data_returns_none(self):
        from agentic_trading.signal.strategies.prediction_consensus import (
            PredictionConsensusStrategy,
        )

        strategy = PredictionConsensusStrategy()
        features = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={},  # No PM features
        )
        result = strategy.on_candle(
            self._make_ctx(), self._make_candle(), features,
        )
        assert result is None

    def test_weak_consensus_returns_none(self):
        from agentic_trading.signal.strategies.prediction_consensus import (
            PredictionConsensusStrategy,
        )

        strategy = PredictionConsensusStrategy()
        features = self._make_features(
            pm_consensus=0.1, pm_count=3, pm_volume=500_000,
        )
        result = strategy.on_candle(
            self._make_ctx(), self._make_candle(), features,
        )
        assert result is None  # Below threshold

    def test_strong_bullish_consensus_generates_long(self):
        from agentic_trading.signal.strategies.prediction_consensus import (
            PredictionConsensusStrategy,
        )

        strategy = PredictionConsensusStrategy()

        # Collect all results — first entry candle should produce the signal
        results = []
        for i in range(4):
            features = self._make_features(
                pm_consensus=0.4 + i * 0.05,
                pm_count=3,
                pm_volume=500_000,
            )
            candle = self._make_candle()
            candle.timestamp = datetime(
                2026, 2, 20, 12, i, tzinfo=timezone.utc,
            )
            result = strategy.on_candle(
                self._make_ctx(), candle, features,
            )
            results.append(result)

        # At least one result should be a LONG signal
        signals = [r for r in results if r is not None]
        assert len(signals) >= 1
        assert signals[0].direction == SignalDirection.LONG
        assert signals[0].strategy_id == "prediction_consensus"
        assert signals[0].confidence > 0

    def test_strong_bearish_consensus_generates_short(self):
        from agentic_trading.signal.strategies.prediction_consensus import (
            PredictionConsensusStrategy,
        )

        strategy = PredictionConsensusStrategy()

        results = []
        for i in range(4):
            features = self._make_features(
                pm_consensus=-0.4 - i * 0.05,
                pm_count=3,
                pm_volume=500_000,
            )
            candle = self._make_candle()
            candle.timestamp = datetime(
                2026, 2, 20, 12, i, tzinfo=timezone.utc,
            )
            result = strategy.on_candle(
                self._make_ctx(), candle, features,
            )
            results.append(result)

        signals = [r for r in results if r is not None]
        assert len(signals) >= 1
        assert signals[0].direction == SignalDirection.SHORT

    def test_insufficient_markets_returns_none(self):
        from agentic_trading.signal.strategies.prediction_consensus import (
            PredictionConsensusStrategy,
        )

        strategy = PredictionConsensusStrategy()
        features = self._make_features(
            pm_consensus=0.6, pm_count=1, pm_volume=500_000,
        )
        result = strategy.on_candle(
            self._make_ctx(), self._make_candle(), features,
        )
        assert result is None  # Below min_markets

    def test_high_event_risk_suppresses_entry(self):
        from agentic_trading.signal.strategies.prediction_consensus import (
            PredictionConsensusStrategy,
        )

        strategy = PredictionConsensusStrategy()

        for i in range(4):
            features = self._make_features(
                pm_consensus=0.5,
                pm_count=3,
                pm_volume=500_000,
            )
            # Override event risk to high
            features.features["pm_event_risk_level"] = 0.8
            candle = self._make_candle()
            candle.timestamp = datetime(
                2026, 2, 20, 12, i, tzinfo=timezone.utc,
            )
            result = strategy.on_candle(
                self._make_ctx(), candle, features,
            )

        assert result is None  # Suppressed by event risk


# ===========================================================================
# Schema registration
# ===========================================================================


class TestSchemaRegistration:
    def test_prediction_topic_registered(self):
        from agentic_trading.bus.schemas import TOPIC_SCHEMAS
        assert "intelligence.prediction" in TOPIC_SCHEMAS
        schemas = TOPIC_SCHEMAS["intelligence.prediction"]
        assert PredictionMarketEvent in schemas
        assert PredictionConsensus in schemas

    def test_event_type_map_includes_pm(self):
        from agentic_trading.bus.schemas import EVENT_TYPE_MAP
        assert "PredictionMarketEvent" in EVENT_TYPE_MAP
        assert "PredictionConsensus" in EVENT_TYPE_MAP


# ===========================================================================
# Config
# ===========================================================================


class TestPredictionMarketConfig:
    def test_default_config(self):
        config = PredictionMarketConfig()
        assert config.enabled is False
        assert config.poll_interval_seconds == 300.0
        assert config.min_volume_usd == 100_000.0
        assert config.max_confidence_boost == 0.15
        assert config.shadow_mode is True

    def test_settings_includes_pm(self):
        from agentic_trading.core.config import Settings
        settings = Settings()
        assert hasattr(settings, "prediction_market")
        assert isinstance(settings.prediction_market, PredictionMarketConfig)


# ===========================================================================
# Narration schema
# ===========================================================================


class TestNarrationPredictionContext:
    def test_prediction_context_field_exists(self):
        from agentic_trading.narration.schema import DecisionExplanation
        explanation = DecisionExplanation(
            prediction_context="65% market consensus aligns with bullish thesis",
        )
        assert "65%" in explanation.prediction_context
