"""Enumerations used across the trading platform."""

from enum import Enum, auto


class Mode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date
    POST_ONLY = "PostOnly"  # Bybit V5: maker-only, rejected if would take


class PositionMode(str, Enum):
    """Bybit V5 position mode."""

    ONE_WAY = "one_way"     # positionIdx=0
    HEDGE = "hedge"         # positionIdx=1 (buy) / 2 (sell)


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # One-way mode


class MarginMode(str, Enum):
    CROSS = "cross"
    ISOLATED = "isolated"


class InstrumentType(str, Enum):
    SPOT = "spot"
    PERP = "perp"  # Perpetual futures
    FUTURE = "future"  # Dated futures
    FX_SPOT = "fx_spot"  # FX spot (T+2 settlement)
    FX_CFD = "fx_cfd"  # FX contract for difference


class RegimeType(str, Enum):
    TREND = "trend"
    RANGE = "range"
    UNKNOWN = "unknown"


class VolatilityRegime(str, Enum):
    HIGH = "high"
    LOW = "low"
    UNKNOWN = "unknown"


class LiquidityRegime(str, Enum):
    HIGH = "high"
    LOW = "low"
    UNKNOWN = "unknown"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        mapping = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        return mapping[self.value]

    @property
    def seconds(self) -> int:
        return self.minutes * 60


class CircuitBreakerType(str, Enum):
    VOLATILITY = "volatility"
    SPREAD = "spread"
    LIQUIDITY = "liquidity"
    STALENESS = "staleness"
    ERROR_RATE = "error_rate"
    CLOCK_SKEW = "clock_skew"


class RiskAlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AssetClass(str, Enum):
    """Asset class identifier for multi-asset support."""

    CRYPTO = "crypto"
    FX = "fx"


class QtyUnit(str, Enum):
    """How order quantity is denominated."""

    BASE = "base"   # Crypto: 0.5 BTC; FX: 10000 EUR
    LOTS = "lots"   # FX standard: 1 lot = 100,000 base units


class Exchange(str, Enum):
    BINANCE = "binance"
    BYBIT = "bybit"
    OANDA = "oanda"
    LMAX = "lmax"


class ConvictionLevel(str, Enum):
    """Trade plan conviction level."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INVALIDATED = "invalidated"


class SetupGrade(str, Enum):
    """Qualitative setup assessment grade."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class MarketStructureBias(str, Enum):
    """Higher-timeframe market structure bias."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"


class MaturityLevel(str, Enum):
    """Strategy maturity level (Soteria-inspired governance).

    L0 = shadow/log-only, L4 = full autonomy.
    Slow promotion, fast demotion.
    """

    L0_SHADOW = "L0_shadow"
    L1_PAPER = "L1_paper"
    L2_GATED = "L2_gated"
    L3_CONSTRAINED = "L3_constrained"
    L4_AUTONOMOUS = "L4_autonomous"

    @property
    def rank(self) -> int:
        """Numeric rank for comparisons (0–4)."""
        return list(MaturityLevel).index(self)


class ImpactTier(str, Enum):
    """Trade impact classification tier."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GovernanceAction(str, Enum):
    """Actions the governance system can take."""

    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    BLOCK = "block"
    DEMOTE = "demote"
    PAUSE = "pause"
    KILL = "kill"


class AgentType(str, Enum):
    """Agent type identifiers for the multi-agent architecture."""

    MARKET_INTELLIGENCE = "market_intelligence"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"
    EXECUTION = "execution"
    RISK_GATE = "risk_gate"
    RECONCILIATION = "reconciliation"
    GOVERNANCE_CANARY = "governance_canary"
    OPTIMIZER = "optimizer"
    REPORTING = "reporting"
    SURVEILLANCE = "surveillance"
    DATA_QUALITY = "data_quality"
    INCIDENT_RESPONSE = "incident_response"
    CMT_ANALYST = "cmt_analyst"
    PREDICTION_MARKET = "prediction_market"
    EXECUTION_PLANNER = "execution_planner"
    FEATURE_COMPUTATION = "feature_computation"
    CUSTOM = "custom"


class AgentStatus(str, Enum):
    """Lifecycle status of an agent."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class StrategyStage(str, Enum):
    """Strategy lifecycle stage (evidence-gated progression).

    Candidate → Backtest → EvalPack → Paper → Limited → Scale
    Demotion can happen at any live stage.
    """

    CANDIDATE = "candidate"
    BACKTEST = "backtest"
    EVAL_PACK = "eval_pack"
    PAPER = "paper"
    LIMITED = "limited"
    SCALE = "scale"
    DEMOTED = "demoted"


class DegradedMode(str, Enum):
    """System degraded-mode levels (escalate only, never auto de-escalate).

    NORMAL → REDUCE_ONLY → NO_NEW_TRADES → READ_ONLY → KILLED
    """

    NORMAL = "normal"
    REDUCE_ONLY = "reduce_only"
    NO_NEW_TRADES = "no_new_trades"
    READ_ONLY = "read_only"
    KILLED = "killed"

    @property
    def rank(self) -> int:
        """Numeric rank for comparisons (0=NORMAL, 4=KILLED)."""
        return list(DegradedMode).index(self)


class IncidentSeverity(str, Enum):
    """Incident severity levels for the incident manager."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    """Incident lifecycle status."""

    DETECTED = "detected"
    TRIAGED = "triaged"
    RESOLVED = "resolved"


# ---------------------------------------------------------------------------
# Context & Reasoning
# ---------------------------------------------------------------------------


class MemoryEntryType(str, Enum):
    """Types of remembered analyses in the memory store."""

    HTF_ASSESSMENT = "htf_assessment"
    SMC_REPORT = "smc_report"
    CMT_ASSESSMENT = "cmt_assessment"
    TRADE_PLAN = "trade_plan"
    SIGNAL = "signal"
    RISK_EVENT = "risk_event"
    REASONING_TRACE = "reasoning_trace"
    TRADE_OUTCOME = "trade_outcome"
    REFLECTION = "reflection"


class ReasoningPhase(str, Enum):
    """Phases in structured agent reasoning."""

    PERCEPTION = "perception"
    HYPOTHESIS = "hypothesis"
    EVALUATION = "evaluation"
    DECISION = "decision"
    ACTION = "action"
    REFLECTION = "reflection"


class PipelineOutcome(str, Enum):
    """Pipeline run outcomes."""

    SIGNAL_EMITTED = "signal_emitted"
    NO_SIGNAL = "no_signal"
    ERROR = "error"
    SKIPPED = "skipped"


class OptimizationRecommendation(str, Enum):
    """Optimizer recommendation for a strategy's parameters."""

    KEEP = "keep"       # Current params are optimal or near-optimal
    UPDATE = "update"   # New params show meaningful improvement
    DISABLE = "disable"  # Strategy underperforming across all param combos
    SKIP = "skip"       # No param grid or data, cannot optimize
