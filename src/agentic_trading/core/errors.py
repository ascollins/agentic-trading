"""Custom exception hierarchy for the trading platform."""


class TradingError(Exception):
    """Base exception for all trading platform errors."""


# --- Configuration ---
class ConfigError(TradingError):
    """Invalid or missing configuration."""


class SafetyGateError(ConfigError):
    """Safety gate not satisfied (e.g., missing I_UNDERSTAND_LIVE_TRADING)."""


# --- Data ---
class DataError(TradingError):
    """Data ingestion or quality error."""


class StaleDataError(DataError):
    """Data is older than acceptable threshold."""


class DataGapError(DataError):
    """Gap detected in data feed."""


# --- Exchange ---
class ExchangeError(TradingError):
    """Exchange communication error."""


class RateLimitError(ExchangeError):
    """Exchange rate limit hit."""


class AuthenticationError(ExchangeError):
    """Exchange authentication failure."""


class InsufficientBalanceError(ExchangeError):
    """Not enough balance for the operation."""


# --- Order ---
class OrderError(TradingError):
    """Order lifecycle error."""


class DuplicateOrderError(OrderError):
    """Duplicate order detected via dedupe key."""


class OrderRejectedError(OrderError):
    """Order rejected by exchange or risk checks."""


class ReconciliationError(OrderError):
    """State mismatch detected during reconciliation."""


# --- Risk ---
class RiskError(TradingError):
    """Risk check failure."""


class RiskLimitBreached(RiskError):
    """A risk limit has been breached."""


class CircuitBreakerTripped(RiskError):
    """A circuit breaker has been triggered."""

    def __init__(self, breaker_type: str, reason: str):
        self.breaker_type = breaker_type
        self.reason = reason
        super().__init__(f"Circuit breaker [{breaker_type}]: {reason}")


class KillSwitchActive(RiskError):
    """Kill switch is active, all trading halted."""


# --- Strategy ---
class StrategyError(TradingError):
    """Strategy computation error."""


class RegimeDetectionError(StrategyError):
    """Regime detection failure."""


# --- Backtest ---
class BacktestError(TradingError):
    """Backtesting engine error."""


class DeterminismError(BacktestError):
    """Backtest determinism violation detected."""


# --- Governance ---
class GovernanceError(TradingError):
    """Governance check failure."""


class MaturityGateError(GovernanceError):
    """Strategy maturity level insufficient for requested action."""


class ExecutionTokenError(GovernanceError):
    """Execution token invalid, expired, or revoked."""


class GovernanceCanaryFailure(GovernanceError):
    """Governance canary detected infrastructure failure."""


# --- Control Plane ---
class ControlPlaneError(TradingError):
    """Control plane infrastructure error."""


class ControlPlaneUnavailable(ControlPlaneError):
    """A critical control plane component is unavailable (audit, policy, etc)."""
