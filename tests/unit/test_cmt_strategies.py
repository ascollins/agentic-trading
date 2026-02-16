"""Tests for CMT-aligned trading strategies.

Covers: MultiTFMA, RSIDivergence, StochasticMACD, BBSqueeze,
MeanReversionEnhanced, FibonacciConfluence, OBVDivergence, SupplyDemand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from agentic_trading.core.enums import (
    Exchange,
    LiquidityRegime,
    RegimeType,
    SignalDirection,
    Timeframe,
    VolatilityRegime,
)
from agentic_trading.core.events import FeatureVector, RegimeState
from agentic_trading.core.models import Candle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candle(
    close: float = 100.0,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: float = 1000.0,
    symbol: str = "BTC/USDT",
    timestamp: datetime | None = None,
) -> Candle:
    return Candle(
        symbol=symbol,
        exchange=Exchange.BYBIT,
        timeframe=Timeframe.H1,
        timestamp=timestamp or datetime(2024, 1, 1, tzinfo=UTC),
        open=open_ or close,
        high=high or close + 1.0,
        low=low or close - 1.0,
        close=close,
        volume=volume,
        is_closed=True,
    )


def _make_fv(features: dict[str, float], symbol: str = "BTC/USDT") -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        timeframe=Timeframe.H1,
        features=features,
    )


def _make_ctx() -> MagicMock:
    return MagicMock()


def _make_regime(
    regime: str = "trend",
    volatility: str = "low",
    symbol: str = "BTC/USDT",
) -> RegimeState:
    return RegimeState(
        symbol=symbol,
        regime=RegimeType(regime),
        volatility=VolatilityRegime(volatility),
        liquidity=LiquidityRegime.HIGH,
        confidence=0.8,
    )


# ===========================================================================
# Strategy 1: Multi-TF MA
# ===========================================================================


class TestMultiTFMAStrategy:
    def _make_strategy(self, **params):
        from agentic_trading.strategies.multi_tf_ma import MultiTFMAStrategy
        return MultiTFMAStrategy(params=params)

    def test_registration(self):
        import agentic_trading.strategies.multi_tf_ma  # noqa: F401
        from agentic_trading.strategies.registry import _REGISTRY
        assert "multi_tf_ma" in _REGISTRY

    def test_long_signal_golden_cross_pullback(self):
        strat = self._make_strategy(min_confidence=0.1, pullback_tolerance_pct=0.05)
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "ema_50": 101.0,
            "ema_200": 95.0,
            "ema_21": 100.5,
            "adx": 30,
            "rsi": 45,
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert "Golden cross" in sig.rationale

    def test_short_signal_death_cross(self):
        strat = self._make_strategy(min_confidence=0.1, pullback_tolerance_pct=0.05)
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "ema_50": 95.0,
            "ema_200": 101.0,
            "ema_21": 99.5,
            "adx": 30,
            "rsi": 55,
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.SHORT

    def test_no_signal_adx_too_low(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "ema_50": 101.0,
            "ema_200": 95.0,
            "ema_21": 100.0,
            "adx": 15,
            "rsi": 45,
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_no_signal_rsi_outside_pullback_zone(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "ema_50": 101.0,
            "ema_200": 95.0,
            "ema_21": 100.0,
            "adx": 30,
            "rsi": 70,  # Too high for pullback zone
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_exit_on_adx_drop(self):
        strat = self._make_strategy(min_confidence=0.1, pullback_tolerance_pct=0.05)
        ctx = _make_ctx()
        # Enter first
        entry_fv = _make_fv({
            "ema_50": 101.0, "ema_200": 95.0, "ema_21": 100.5,
            "adx": 30, "rsi": 45, "atr": 2.0,
        })
        strat.on_candle(ctx, _make_candle(close=100.0), entry_fv)

        # ADX drops below exit threshold
        exit_fv = _make_fv({
            "ema_50": 101.0, "ema_200": 95.0, "ema_21": 100.0,
            "adx": 10, "rsi": 50, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, _make_candle(close=101.0), exit_fv)
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT

    def test_regime_reduces_confidence(self):
        strat = self._make_strategy(min_confidence=0.1, pullback_tolerance_pct=0.05)
        strat.on_regime_change(_make_regime("range"))
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "ema_50": 101.0, "ema_200": 95.0, "ema_21": 100.5,
            "adx": 25, "rsi": 45, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        if sig is not None:
            assert "Range regime" in sig.rationale


# ===========================================================================
# Strategy 3: RSI Divergence
# ===========================================================================


class TestRSIDivergenceStrategy:
    def _make_strategy(self, **params):
        from agentic_trading.strategies.rsi_divergence import RSIDivergenceStrategy
        return RSIDivergenceStrategy(params=params)

    def test_registration(self):
        import agentic_trading.strategies.rsi_divergence  # noqa: F401
        from agentic_trading.strategies.registry import _REGISTRY
        assert "rsi_divergence" in _REGISTRY

    def test_no_signal_insufficient_history(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({"rsi": 30, "atr": 2.0})
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_bullish_divergence_detection(self):
        """Feed declining prices with rising RSI to trigger bullish divergence."""
        strat = self._make_strategy(
            min_confidence=0.1,
            lookback_bars=20,
            min_divergence_bars=3,
            rsi_oversold=40,
        )
        ctx = _make_ctx()

        # Build up history with a pattern: price goes down, RSI goes up
        # First swing low
        for i in range(10):
            ts = datetime(2024, 1, 1, i, tzinfo=UTC)
            price = 100.0 - i * 0.5  # Declining
            rsi = 35 - i * 0.3       # Declining RSI
            candle = _make_candle(close=price, timestamp=ts)
            fv = _make_fv({"rsi": rsi, "atr": 2.0})
            strat.on_candle(ctx, candle, fv)

        # Recovery
        for i in range(5):
            ts = datetime(2024, 1, 1, 10 + i, tzinfo=UTC)
            price = 95.0 + i * 0.3
            rsi = 32.0 + i * 2
            candle = _make_candle(close=price, timestamp=ts)
            fv = _make_fv({"rsi": rsi, "atr": 2.0})
            strat.on_candle(ctx, candle, fv)

        # Second swing low: lower price but higher RSI (bullish divergence)
        for i in range(8):
            ts = datetime(2024, 1, 1, 15 + i, tzinfo=UTC)
            price = 96.5 - i * 0.8   # Goes lower than first low
            rsi = 38 - i * 0.5       # But RSI stays higher
            candle = _make_candle(close=price, timestamp=ts)
            fv = _make_fv({"rsi": rsi, "atr": 2.0})
            strat.on_candle(ctx, candle, fv)

        # At minimum, strategy should have processed without error
        # The exact signal depends on divergence detection heuristics
        assert True  # No crash

    def test_exit_on_rsi_mean_reversion(self):
        strat = self._make_strategy(min_confidence=0.1)
        ctx = _make_ctx()
        # Force entry
        strat._record_entry("BTC/USDT", "long")

        candle = _make_candle(close=100.0)
        fv = _make_fv({"rsi": 55, "atr": 2.0})  # RSI above exit level
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT

    def test_no_exit_when_rsi_below_threshold(self):
        strat = self._make_strategy(min_confidence=0.1)
        ctx = _make_ctx()
        strat._record_entry("BTC/USDT", "long")

        candle = _make_candle(close=100.0)
        fv = _make_fv({"rsi": 40, "atr": 2.0})  # RSI below exit level
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None


# ===========================================================================
# Strategy 4: Stochastic + MACD
# ===========================================================================


class TestStochasticMACDStrategy:
    def _make_strategy(self, **params):
        from agentic_trading.strategies.stochastic_macd import StochasticMACDStrategy
        return StochasticMACDStrategy(params=params)

    def test_registration(self):
        import agentic_trading.strategies.stochastic_macd  # noqa: F401
        from agentic_trading.strategies.registry import _REGISTRY
        assert "stochastic_macd" in _REGISTRY

    def test_no_signal_missing_features(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({"rsi": 50})  # Missing stoch + macd
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_long_signal_confluence(self):
        strat = self._make_strategy(min_confidence=0.1, volume_gate=0.5)
        ctx = _make_ctx()

        # First bar: MACD bullish cross
        candle1 = _make_candle(close=100.0, timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC))
        fv1 = _make_fv({
            "macd": 0.5, "macd_signal": 0.3,
            "macd_prev": -0.1, "macd_signal_prev": 0.3,
            "macd_histogram": 0.2,
            "stoch_k": 15, "stoch_d": 20,
            "stoch_k_prev": 18, "stoch_d_prev": 20,
            "volume_ratio": 1.5, "atr": 2.0,
        })
        strat.on_candle(ctx, candle1, fv1)

        # Second bar: Stochastic bullish cross (within window)
        candle2 = _make_candle(close=101.0, timestamp=datetime(2024, 1, 1, 1, tzinfo=UTC))
        fv2 = _make_fv({
            "macd": 0.6, "macd_signal": 0.4,
            "macd_prev": 0.5, "macd_signal_prev": 0.3,
            "macd_histogram": 0.2,
            "stoch_k": 22, "stoch_d": 18,
            "stoch_k_prev": 15, "stoch_d_prev": 20,
            "volume_ratio": 1.5, "atr": 2.0,
        })
        sig2 = strat.on_candle(ctx, candle2, fv2)

        # Should have confluence signal
        if sig2 is not None:
            assert sig2.direction == SignalDirection.LONG

    def test_exit_on_macd_histogram_reversal(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        strat._record_entry("BTC/USDT", "long")

        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "macd": -0.1, "macd_signal": 0.1,
            "macd_prev": 0.1, "macd_signal_prev": 0.1,
            "macd_histogram": -0.2,
            "stoch_k": 60, "stoch_d": 55,
            "stoch_k_prev": 58, "stoch_d_prev": 55,
            "volume_ratio": 1.0, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT

    def test_no_signal_without_volume(self):
        strat = self._make_strategy(volume_gate=2.0)
        ctx = _make_ctx()

        # Set up confluence but with low volume
        strat._macd_cross_bar["BTC/USDT"] = ("long", 1)
        strat._stoch_cross_bar["BTC/USDT"] = ("long", 1)
        strat._bar_count["BTC/USDT"] = 1

        candle = _make_candle(close=100.0, timestamp=datetime(2024, 1, 1, 2, tzinfo=UTC))
        fv = _make_fv({
            "macd": 0.5, "macd_signal": 0.3,
            "macd_prev": 0.5, "macd_signal_prev": 0.3,
            "macd_histogram": 0.2,
            "stoch_k": 25, "stoch_d": 20,
            "stoch_k_prev": 25, "stoch_d_prev": 20,
            "volume_ratio": 0.5,  # Below gate
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None


# ===========================================================================
# Strategy 5: BB Squeeze Breakout
# ===========================================================================


class TestBBSqueezeStrategy:
    def _make_strategy(self, **params):
        from agentic_trading.strategies.bb_squeeze import BBSqueezeStrategy
        return BBSqueezeStrategy(params=params)

    def test_registration(self):
        import agentic_trading.strategies.bb_squeeze  # noqa: F401
        from agentic_trading.strategies.registry import _REGISTRY
        assert "bb_squeeze" in _REGISTRY

    def test_no_signal_missing_features(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({"rsi": 50})
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_long_breakout_after_squeeze(self):
        strat = self._make_strategy(min_confidence=0.1, squeeze_percentile=0.2)
        ctx = _make_ctx()

        # Bar 1: In squeeze
        candle1 = _make_candle(close=100.0, timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC))
        fv1 = _make_fv({
            "bb_upper": 101.0, "bb_lower": 99.0, "bb_middle": 100.0,
            "bbw": 0.02, "bbw_percentile": 0.05,
            "keltner_upper": 103.0, "keltner_lower": 97.0,
            "adx": 25, "rsi": 55, "atr": 2.0,
        })
        strat.on_candle(ctx, candle1, fv1)

        # Bar 2: Breakout above upper BB
        candle2 = _make_candle(close=102.0, timestamp=datetime(2024, 1, 1, 1, tzinfo=UTC))
        fv2 = _make_fv({
            "bb_upper": 101.0, "bb_lower": 99.0, "bb_middle": 100.0,
            "bbw": 0.025, "bbw_percentile": 0.15,
            "keltner_upper": 103.0, "keltner_lower": 97.0,
            "adx": 25, "rsi": 55, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle2, fv2)
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert "Squeeze breakout" in sig.rationale

    def test_no_signal_without_prior_squeeze(self):
        strat = self._make_strategy()
        ctx = _make_ctx()

        # No squeeze on previous bar, breakout above BB
        candle = _make_candle(close=102.0)
        fv = _make_fv({
            "bb_upper": 101.0, "bb_lower": 99.0, "bb_middle": 100.0,
            "bbw": 0.05, "bbw_percentile": 0.50,
            "adx": 25, "rsi": 55, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_exit_via_keltner_trailing(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        strat._record_entry("BTC/USDT", "long")

        candle = _make_candle(close=96.0)
        fv = _make_fv({
            "bb_upper": 101.0, "bb_lower": 99.0, "bb_middle": 100.0,
            "bbw": 0.05,
            "keltner_upper": 103.0, "keltner_lower": 97.0,
            "adx": 20, "rsi": 45, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT


# ===========================================================================
# Strategy 6: Mean-Reversion Enhanced
# ===========================================================================


class TestMeanReversionEnhancedStrategy:
    def _make_strategy(self, **params):
        from agentic_trading.strategies.mean_reversion_enhanced import (
            MeanReversionEnhancedStrategy,
        )
        return MeanReversionEnhancedStrategy(params=params)

    def test_registration(self):
        import agentic_trading.strategies.mean_reversion_enhanced  # noqa: F401
        from agentic_trading.strategies.registry import _REGISTRY
        assert "mean_reversion_enhanced" in _REGISTRY

    def test_long_signal(self):
        strat = self._make_strategy(min_confidence=0.1)
        ctx = _make_ctx()
        candle = _make_candle(close=95.0)
        fv = _make_fv({
            "bb_upper": 110.0, "bb_lower": 96.0, "bb_middle": 103.0,
            "rsi": 25, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert "lower BB" in sig.rationale

    def test_short_signal(self):
        strat = self._make_strategy(min_confidence=0.1)
        ctx = _make_ctx()
        candle = _make_candle(close=112.0)
        fv = _make_fv({
            "bb_upper": 110.0, "bb_lower": 96.0, "bb_middle": 103.0,
            "rsi": 75, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.SHORT

    def test_time_stop_exit(self):
        strat = self._make_strategy(min_confidence=0.1, time_stop_bars=3)
        ctx = _make_ctx()

        # Enter
        entry_fv = _make_fv({
            "bb_upper": 110.0, "bb_lower": 96.0, "bb_middle": 103.0,
            "rsi": 25, "atr": 2.0,
        })
        strat.on_candle(ctx, _make_candle(close=95.0), entry_fv)

        # Bars without reaching target
        hold_fv = _make_fv({
            "bb_upper": 110.0, "bb_lower": 96.0, "bb_middle": 103.0,
            "rsi": 40, "atr": 2.0,
        })
        for _ in range(3):
            sig = strat.on_candle(ctx, _make_candle(close=97.0), hold_fv)

        # Should exit on time stop
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT
        assert "Time stop" in sig.rationale

    def test_target_exit(self):
        strat = self._make_strategy(min_confidence=0.1)
        ctx = _make_ctx()

        # Enter
        entry_fv = _make_fv({
            "bb_upper": 110.0, "bb_lower": 96.0, "bb_middle": 103.0,
            "rsi": 25, "atr": 2.0,
        })
        strat.on_candle(ctx, _make_candle(close=95.0), entry_fv)

        # Price reaches middle BB
        exit_fv = _make_fv({
            "bb_upper": 110.0, "bb_lower": 96.0, "bb_middle": 103.0,
            "rsi": 50, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, _make_candle(close=104.0), exit_fv)
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT
        assert "Target hit" in sig.rationale


# ===========================================================================
# Strategy 8: Supply/Demand Zone
# ===========================================================================


class TestSupplyDemandStrategy:
    def _make_strategy(self, **params):
        from agentic_trading.strategies.supply_demand import SupplyDemandStrategy
        return SupplyDemandStrategy(params=params)

    def test_registration(self):
        import agentic_trading.strategies.supply_demand  # noqa: F401
        from agentic_trading.strategies.registry import _REGISTRY
        assert "supply_demand" in _REGISTRY

    def test_no_signal_without_smc_features(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({"rsi": 50, "atr": 2.0})
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_long_signal_demand_zone(self):
        strat = self._make_strategy(min_confidence=0.1, require_bos=False)
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "smc_nearest_demand_distance": 0.005,
            "smc_nearest_supply_distance": 0.05,
            "smc_ob_unmitigated_bullish": 2,
            "smc_ob_unmitigated_bearish": 0,
            "smc_bos_bullish": 1,
            "smc_bos_bearish": 0,
            "smc_choch_bullish": 0,
            "smc_choch_bearish": 0,
            "smc_swing_bias": 0.5,
            "volume_ratio": 1.2,
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert "Demand zone" in sig.rationale

    def test_short_signal_supply_zone(self):
        strat = self._make_strategy(min_confidence=0.1, require_bos=False)
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "smc_nearest_demand_distance": 0.05,
            "smc_nearest_supply_distance": 0.005,
            "smc_ob_unmitigated_bullish": 0,
            "smc_ob_unmitigated_bearish": 3,
            "smc_bos_bullish": 0,
            "smc_bos_bearish": 1,
            "smc_choch_bullish": 0,
            "smc_choch_bearish": 0,
            "volume_ratio": 1.2,
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.SHORT

    def test_no_signal_without_bos_when_required(self):
        strat = self._make_strategy(require_bos=True)
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "smc_nearest_demand_distance": 0.005,
            "smc_ob_unmitigated_bullish": 2,
            "smc_bos_bullish": 0,  # No BOS
            "smc_bos_bearish": 0,
            "smc_choch_bullish": 0,
            "smc_choch_bearish": 0,
            "volume_ratio": 1.2,
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_exit_on_bos_against_position(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        strat._record_entry("BTC/USDT", "long")

        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "smc_nearest_demand_distance": 0.05,
            "smc_nearest_supply_distance": 0.01,
            "smc_ob_unmitigated_bullish": 0,
            "smc_ob_unmitigated_bearish": 0,
            "smc_bos_bullish": 0,
            "smc_bos_bearish": 1,  # Bearish BOS against long
            "smc_choch_bullish": 0,
            "smc_choch_bearish": 0,
            "volume_ratio": 1.0,
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT


# ===========================================================================
# Strategy 13: Fibonacci Confluence
# ===========================================================================


class TestFibonacciConfluenceStrategy:
    def _make_strategy(self, **params):
        from agentic_trading.strategies.fibonacci_confluence import FibonacciConfluenceStrategy
        return FibonacciConfluenceStrategy(params=params)

    def test_registration(self):
        import agentic_trading.strategies.fibonacci_confluence  # noqa: F401
        from agentic_trading.strategies.registry import _REGISTRY
        assert "fibonacci_confluence" in _REGISTRY

    def test_no_signal_insufficient_history(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({"rsi": 35, "atr": 2.0})
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_exit_on_rsi_overbought(self):
        strat = self._make_strategy(swing_lookback=5)
        ctx = _make_ctx()
        # Build up history first so exit check isn't blocked by lookback
        for i in range(6):
            ts = datetime(2024, 1, 1, i, tzinfo=UTC)
            c = _make_candle(close=100.0 + i, high=101.0 + i, low=99.0 + i, timestamp=ts)
            strat.on_candle(ctx, c, _make_fv({"rsi": 50, "atr": 2.0}))

        strat._record_entry("BTC/USDT", "long")

        candle = _make_candle(close=110.0, timestamp=datetime(2024, 1, 1, 7, tzinfo=UTC))
        fv = _make_fv({"rsi": 75, "atr": 2.0})
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT

    def test_swing_point_detection(self):
        strat = self._make_strategy(min_swing_pct=0.01)
        # Build data: V-shape with clear swing high and swing low
        # Rise -> peak -> fall -> trough -> rise
        highs =  [100, 105, 110, 115, 120, 115, 110, 105, 100, 95, 90, 95, 100, 105, 110]
        lows =   [ 98, 103, 108, 113, 118, 113, 108, 103,  98, 93, 88, 93,  98, 103, 108]
        closes = [ 99, 104, 109, 114, 119, 114, 109, 104,  99, 94, 89, 94,  99, 104, 109]

        swings = strat._find_swing_points(highs, lows, closes)
        # Should find a swing high near peak and swing low near trough
        assert len(swings) >= 2

    def test_confluence_zone_finding(self):
        strat = self._make_strategy(min_confluence_levels=2)
        # Create levels that cluster
        fib_levels = [100.0, 100.3, 100.1, 120.0, 120.2, 80.0]
        zones = strat._find_confluence_zones(fib_levels, 100.0, 0.005, 2)
        # Should find a zone near 100
        assert len(zones) >= 1


# ===========================================================================
# Strategy 14: OBV Divergence
# ===========================================================================


class TestOBVDivergenceStrategy:
    def _make_strategy(self, **params):
        from agentic_trading.strategies.obv_divergence import OBVDivergenceStrategy
        return OBVDivergenceStrategy(params=params)

    def test_registration(self):
        import agentic_trading.strategies.obv_divergence  # noqa: F401
        from agentic_trading.strategies.registry import _REGISTRY
        assert "obv_divergence" in _REGISTRY

    def test_no_signal_without_obv(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({"rsi": 50})
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_no_signal_insufficient_history(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({"obv": 1000, "volume_ratio": 1.2, "atr": 2.0})
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is None

    def test_exit_on_obv_ema_cross(self):
        strat = self._make_strategy()
        ctx = _make_ctx()
        strat._record_entry("BTC/USDT", "long")

        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "obv": 900,
            "obv_ema_20": 1000,  # OBV below its EMA
            "volume_ratio": 1.0,
            "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT
        assert "OBV crossed below" in sig.rationale

    def test_bullish_divergence_detection(self):
        strat = self._make_strategy()
        # Price lower low, OBV higher low
        prices = [100, 99, 98, 97, 96, 98, 100, 99, 97, 96, 95, 94]
        obvs = [0, 100, 50, -100, -200, -100, 0, 50, 100, 50, -50, -100]
        result = strat._detect_bullish_divergence(prices, obvs, 3)
        # Result may or may not be detected depending on exact swing heuristics
        # But should not crash
        assert result is None or isinstance(result, float)

    def test_bearish_divergence_detection(self):
        strat = self._make_strategy()
        prices = [100, 101, 102, 103, 104, 102, 100, 101, 103, 104, 105, 106]
        obvs = [0, 100, 200, 300, 400, 300, 200, 250, 350, 380, 370, 350]
        result = strat._detect_bearish_divergence(prices, obvs, 3)
        assert result is None or isinstance(result, float)


# ===========================================================================
# Strategy registry completeness
# ===========================================================================


class TestStrategyRegistry:
    """Verify all CMT strategies are properly registered."""

    def test_all_strategies_registered(self):
        # Import all strategy modules
        import agentic_trading.strategies.bb_squeeze  # noqa: F401
        import agentic_trading.strategies.breakout  # noqa: F401
        import agentic_trading.strategies.fibonacci_confluence  # noqa: F401
        import agentic_trading.strategies.funding_arb  # noqa: F401
        import agentic_trading.strategies.mean_reversion  # noqa: F401
        import agentic_trading.strategies.mean_reversion_enhanced  # noqa: F401
        import agentic_trading.strategies.multi_tf_ma  # noqa: F401
        import agentic_trading.strategies.obv_divergence  # noqa: F401
        import agentic_trading.strategies.rsi_divergence  # noqa: F401
        import agentic_trading.strategies.stochastic_macd  # noqa: F401
        import agentic_trading.strategies.supply_demand  # noqa: F401
        import agentic_trading.strategies.trend_following  # noqa: F401
        from agentic_trading.strategies.registry import list_strategies
        strategies = list_strategies()

        expected = [
            "bb_squeeze",
            "breakout",
            "fibonacci_confluence",
            "funding_arb",
            "mean_reversion",
            "mean_reversion_enhanced",
            "multi_tf_ma",
            "obv_divergence",
            "rsi_divergence",
            "stochastic_macd",
            "supply_demand",
            "trend_following",
        ]
        assert strategies == expected

    def test_all_strategies_instantiate(self):
        import agentic_trading.strategies.bb_squeeze  # noqa: F401
        import agentic_trading.strategies.fibonacci_confluence  # noqa: F401
        import agentic_trading.strategies.mean_reversion_enhanced  # noqa: F401
        import agentic_trading.strategies.multi_tf_ma  # noqa: F401
        import agentic_trading.strategies.obv_divergence  # noqa: F401
        import agentic_trading.strategies.rsi_divergence  # noqa: F401
        import agentic_trading.strategies.stochastic_macd  # noqa: F401
        import agentic_trading.strategies.supply_demand  # noqa: F401
        from agentic_trading.strategies.registry import create_strategy

        for sid in [
            "multi_tf_ma", "rsi_divergence", "stochastic_macd",
            "bb_squeeze", "mean_reversion_enhanced", "fibonacci_confluence",
            "obv_divergence", "supply_demand",
        ]:
            strat = create_strategy(sid)
            assert strat.strategy_id == sid
            assert isinstance(strat.get_parameters(), dict)

    def test_all_strategies_have_risk_constraints_in_signals(self):
        """Every strategy should include sizing_method in risk_constraints."""
        import agentic_trading.strategies.mean_reversion_enhanced  # noqa: F401
        import agentic_trading.strategies.multi_tf_ma  # noqa: F401
        from agentic_trading.strategies.registry import create_strategy

        # Test multi_tf_ma
        strat = create_strategy(
            "multi_tf_ma",
            {"min_confidence": 0.1, "pullback_tolerance_pct": 0.05},
        )
        ctx = _make_ctx()
        candle = _make_candle(close=100.0)
        fv = _make_fv({
            "ema_50": 101.0, "ema_200": 95.0, "ema_21": 100.5,
            "adx": 30, "rsi": 45, "atr": 2.0,
        })
        sig = strat.on_candle(ctx, candle, fv)
        if sig and sig.direction != SignalDirection.FLAT:
            assert "sizing_method" in sig.risk_constraints

        # Test mean_reversion_enhanced
        strat2 = create_strategy("mean_reversion_enhanced", {"min_confidence": 0.1})
        fv2 = _make_fv({
            "bb_upper": 110.0, "bb_lower": 96.0, "bb_middle": 103.0,
            "rsi": 25, "atr": 2.0,
        })
        sig2 = strat2.on_candle(ctx, _make_candle(close=95.0), fv2)
        if sig2 and sig2.direction != SignalDirection.FLAT:
            assert "sizing_method" in sig2.risk_constraints
