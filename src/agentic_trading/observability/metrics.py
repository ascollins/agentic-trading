"""Prometheus metrics endpoint.

Exposes trading platform metrics for monitoring via Grafana.
"""

from __future__ import annotations

import threading
from typing import Any

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------

SYSTEM_INFO = Info("trading_system", "Trading system information")

# ---------------------------------------------------------------------------
# Trading metrics
# ---------------------------------------------------------------------------

SIGNALS_TOTAL = Counter(
    "trading_signals_total",
    "Total signals generated",
    ["strategy_id", "symbol", "direction"],
)

ORDERS_TOTAL = Counter(
    "trading_orders_total",
    "Total orders submitted",
    ["symbol", "side", "order_type", "status"],
)

FILLS_TOTAL = Counter(
    "trading_fills_total",
    "Total fills received",
    ["symbol", "side"],
)

# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

EQUITY = Gauge(
    "trading_equity",
    "Current portfolio equity",
)

POSITION_SIZE = Gauge(
    "trading_position_size",
    "Current position size",
    ["symbol", "side"],
)

UNREALIZED_PNL = Gauge(
    "trading_unrealized_pnl",
    "Unrealized PnL",
    ["symbol"],
)

DAILY_PNL = Gauge(
    "trading_daily_pnl",
    "Daily realized PnL",
)

DRAWDOWN = Gauge(
    "trading_drawdown_pct",
    "Current drawdown percentage",
)

GROSS_EXPOSURE = Gauge(
    "trading_gross_exposure",
    "Gross portfolio exposure",
)

# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

CIRCUIT_BREAKER_TRIPS = Counter(
    "trading_circuit_breaker_trips_total",
    "Circuit breaker trip count",
    ["breaker_type"],
)

RISK_CHECK_FAILURES = Counter(
    "trading_risk_check_failures_total",
    "Risk check failures",
    ["check_name"],
)

KILL_SWITCH_ACTIVE = Gauge(
    "trading_kill_switch_active",
    "Kill switch status (1=active, 0=inactive)",
)

# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------

DECISION_LATENCY = Histogram(
    "trading_decision_latency_seconds",
    "Time from candle to order intent",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

ORDER_LATENCY = Histogram(
    "trading_order_latency_seconds",
    "Time from order submission to exchange ack",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

RECONCILIATION_LATENCY = Histogram(
    "trading_reconciliation_latency_seconds",
    "Reconciliation loop duration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# ---------------------------------------------------------------------------
# Data metrics
# ---------------------------------------------------------------------------

CANDLES_PROCESSED = Counter(
    "trading_candles_processed_total",
    "Total candles processed",
    ["symbol", "timeframe"],
)

DATA_STALENESS = Gauge(
    "trading_data_staleness_seconds",
    "Seconds since last data update",
    ["symbol"],
)

WS_RECONNECTIONS = Counter(
    "trading_ws_reconnections_total",
    "WebSocket reconnection count",
    ["exchange"],
)

# ---------------------------------------------------------------------------
# Governance metrics (Soteria-inspired)
# ---------------------------------------------------------------------------

GOVERNANCE_DECISIONS = Counter(
    "trading_governance_decisions_total",
    "Total governance gate decisions",
    ["strategy_id", "action"],
)

GOVERNANCE_BLOCKS = Counter(
    "trading_governance_blocks_total",
    "Governance blocks by reason category",
    ["strategy_id", "reason"],
)

MATURITY_LEVEL = Gauge(
    "trading_strategy_maturity_level",
    "Strategy maturity level (0=L0, 4=L4)",
    ["strategy_id"],
)

HEALTH_SCORE = Gauge(
    "trading_strategy_health_score",
    "Strategy health score (0.0 to 1.0)",
    ["strategy_id"],
)

DRIFT_DEVIATION = Gauge(
    "trading_strategy_drift_pct",
    "Strategy drift from baseline percentage",
    ["strategy_id", "metric"],
)

CANARY_STATUS = Gauge(
    "trading_canary_status",
    "Canary component health (1=healthy, 0=unhealthy)",
    ["component"],
)

GOVERNANCE_LATENCY = Histogram(
    "trading_governance_latency_seconds",
    "Governance gate evaluation latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)

ACTIVE_TOKENS = Gauge(
    "trading_active_execution_tokens",
    "Active execution tokens",
    ["strategy_id"],
)


def start_metrics_server(port: int = 9090, mode: str = "unknown") -> None:
    """Start Prometheus metrics HTTP server in a background thread."""
    SYSTEM_INFO.info({
        "version": "0.1.0",
        "mode": mode,
    })
    start_http_server(port)


# ---------------------------------------------------------------------------
# Convenience helpers for emitting metrics from the trading pipeline
# ---------------------------------------------------------------------------


def record_signal(strategy_id: str, symbol: str, direction: str) -> None:
    """Record a strategy signal emission."""
    SIGNALS_TOTAL.labels(strategy_id=strategy_id, symbol=symbol, direction=direction).inc()


def record_fill(symbol: str, side: str, qty: float = 0.0, price: float = 0.0) -> None:
    """Record an order fill."""
    FILLS_TOTAL.labels(symbol=symbol, side=side).inc()


def record_order(symbol: str, side: str, order_type: str, status: str) -> None:
    """Record an order submission."""
    ORDERS_TOTAL.labels(symbol=symbol, side=side, order_type=order_type, status=status).inc()


def update_equity(value: float) -> None:
    """Update the portfolio equity gauge."""
    EQUITY.set(value)


def update_drawdown(pct: float) -> None:
    """Update the drawdown percentage gauge."""
    DRAWDOWN.set(pct)


def update_position(symbol: str, side: str, size: float) -> None:
    """Update position size gauge for a symbol."""
    POSITION_SIZE.labels(symbol=symbol, side=side).set(size)


def update_daily_pnl(value: float) -> None:
    """Update the daily PnL gauge."""
    DAILY_PNL.set(value)


def update_kill_switch(active: bool) -> None:
    """Update the kill switch gauge."""
    KILL_SWITCH_ACTIVE.set(1 if active else 0)


def record_candle_processed(symbol: str, timeframe: str) -> None:
    """Record a candle being processed."""
    CANDLES_PROCESSED.labels(symbol=symbol, timeframe=timeframe).inc()


def update_data_staleness(symbol: str, seconds: float) -> None:
    """Update data staleness gauge."""
    DATA_STALENESS.labels(symbol=symbol).set(seconds)


def record_ws_reconnection(exchange: str) -> None:
    """Record a WebSocket reconnection."""
    WS_RECONNECTIONS.labels(exchange=exchange).inc()


def record_decision_latency(seconds: float) -> None:
    """Record latency from candle to order intent."""
    DECISION_LATENCY.observe(seconds)


# ---------------------------------------------------------------------------
# Governance helpers
# ---------------------------------------------------------------------------


def record_governance_decision(strategy_id: str, action: str) -> None:
    """Record a governance gate decision."""
    GOVERNANCE_DECISIONS.labels(strategy_id=strategy_id, action=action).inc()


def record_governance_block(strategy_id: str, reason: str) -> None:
    """Record a governance block."""
    GOVERNANCE_BLOCKS.labels(strategy_id=strategy_id, reason=reason).inc()


def update_maturity_level(strategy_id: str, level: int) -> None:
    """Update strategy maturity level gauge (0=L0, 4=L4)."""
    MATURITY_LEVEL.labels(strategy_id=strategy_id).set(level)


def update_health_score(strategy_id: str, score: float) -> None:
    """Update strategy health score gauge."""
    HEALTH_SCORE.labels(strategy_id=strategy_id).set(score)


def update_drift_deviation(
    strategy_id: str, metric: str, pct: float
) -> None:
    """Update drift deviation gauge for a metric."""
    DRIFT_DEVIATION.labels(strategy_id=strategy_id, metric=metric).set(pct)


def update_canary_status(component: str, healthy: bool) -> None:
    """Update canary component health gauge."""
    CANARY_STATUS.labels(component=component).set(1 if healthy else 0)


def record_governance_latency(seconds: float) -> None:
    """Record governance gate evaluation latency."""
    GOVERNANCE_LATENCY.observe(seconds)


def update_active_tokens(strategy_id: str, count: int) -> None:
    """Update active execution tokens gauge."""
    ACTIVE_TOKENS.labels(strategy_id=strategy_id).set(count)


# ---------------------------------------------------------------------------
# Journal & analytics metrics (Edgewonk-inspired)
# ---------------------------------------------------------------------------

JOURNAL_TRADES_TOTAL = Counter(
    "trading_journal_trades_total",
    "Total trades tracked by journal",
    ["strategy_id", "outcome"],
)

JOURNAL_WIN_RATE = Gauge(
    "trading_journal_win_rate",
    "Rolling win rate per strategy",
    ["strategy_id"],
)

JOURNAL_PROFIT_FACTOR = Gauge(
    "trading_journal_profit_factor",
    "Rolling profit factor per strategy",
    ["strategy_id"],
)

JOURNAL_EXPECTANCY = Gauge(
    "trading_journal_expectancy",
    "Rolling expectancy (avg PnL per trade)",
    ["strategy_id"],
)

JOURNAL_AVG_R = Gauge(
    "trading_journal_avg_r_multiple",
    "Average R-multiple per strategy",
    ["strategy_id"],
)

JOURNAL_SHARPE = Gauge(
    "trading_journal_sharpe",
    "Trade-level Sharpe ratio per strategy",
    ["strategy_id"],
)

JOURNAL_MANAGEMENT_EFF = Gauge(
    "trading_journal_management_efficiency",
    "Average management efficiency (actual/MFE)",
    ["strategy_id"],
)

JOURNAL_MAX_DRAWDOWN = Gauge(
    "trading_journal_max_drawdown",
    "Rolling window max drawdown",
    ["strategy_id"],
)

JOURNAL_CONFIDENCE_BRIER = Gauge(
    "trading_journal_confidence_brier",
    "Confidence calibration Brier score (lower=better)",
    ["strategy_id"],
)

JOURNAL_OVERTRADING = Gauge(
    "trading_journal_overtrading",
    "Overtrading status (1=overtrading, 0=normal)",
    ["strategy_id"],
)

JOURNAL_EDGE_PVALUE = Gauge(
    "trading_journal_edge_pvalue",
    "Statistical edge p-value (lower=stronger edge)",
    ["strategy_id"],
)

JOURNAL_RUIN_PROBABILITY = Gauge(
    "trading_journal_ruin_probability",
    "Monte Carlo probability of ruin",
    ["strategy_id"],
)

JOURNAL_KELLY_FRACTION = Gauge(
    "trading_journal_kelly_fraction",
    "Optimal Kelly fraction",
    ["strategy_id"],
)

JOURNAL_OPEN_TRADES = Gauge(
    "trading_journal_open_trades",
    "Currently open trades",
)

JOURNAL_CLOSED_TRADES = Gauge(
    "trading_journal_closed_trades",
    "Total closed trades in journal",
)


# ---------------------------------------------------------------------------
# Journal convenience helpers
# ---------------------------------------------------------------------------


def record_journal_trade(strategy_id: str, outcome: str) -> None:
    """Record a trade closure in the journal."""
    JOURNAL_TRADES_TOTAL.labels(strategy_id=strategy_id, outcome=outcome).inc()


def update_journal_rolling_metrics(strategy_id: str, metrics: dict) -> None:
    """Update rolling journal metrics from a snapshot dict."""
    if "win_rate" in metrics:
        JOURNAL_WIN_RATE.labels(strategy_id=strategy_id).set(metrics["win_rate"])
    if "profit_factor" in metrics and metrics["profit_factor"] != float("inf"):
        JOURNAL_PROFIT_FACTOR.labels(strategy_id=strategy_id).set(metrics["profit_factor"])
    if "expectancy" in metrics:
        JOURNAL_EXPECTANCY.labels(strategy_id=strategy_id).set(metrics["expectancy"])
    if "avg_r" in metrics:
        JOURNAL_AVG_R.labels(strategy_id=strategy_id).set(metrics["avg_r"])
    if "sharpe" in metrics:
        JOURNAL_SHARPE.labels(strategy_id=strategy_id).set(metrics["sharpe"])
    if "avg_management_efficiency" in metrics:
        JOURNAL_MANAGEMENT_EFF.labels(strategy_id=strategy_id).set(
            metrics["avg_management_efficiency"]
        )
    if "max_drawdown" in metrics:
        JOURNAL_MAX_DRAWDOWN.labels(strategy_id=strategy_id).set(metrics["max_drawdown"])


def update_journal_counts(open_count: int, closed_count: int) -> None:
    """Update open/closed trade count gauges."""
    JOURNAL_OPEN_TRADES.set(open_count)
    JOURNAL_CLOSED_TRADES.set(closed_count)


def update_journal_confidence(strategy_id: str, brier_score: float) -> None:
    """Update confidence calibration Brier score."""
    JOURNAL_CONFIDENCE_BRIER.labels(strategy_id=strategy_id).set(brier_score)


def update_journal_overtrading(strategy_id: str, is_overtrading: bool) -> None:
    """Update overtrading status gauge."""
    JOURNAL_OVERTRADING.labels(strategy_id=strategy_id).set(1 if is_overtrading else 0)


def update_journal_edge(strategy_id: str, p_value: float) -> None:
    """Update statistical edge p-value."""
    JOURNAL_EDGE_PVALUE.labels(strategy_id=strategy_id).set(p_value)


def update_journal_monte_carlo(
    strategy_id: str, ruin_prob: float, kelly: float
) -> None:
    """Update Monte Carlo and Kelly metrics."""
    JOURNAL_RUIN_PROBABILITY.labels(strategy_id=strategy_id).set(ruin_prob)
    JOURNAL_KELLY_FRACTION.labels(strategy_id=strategy_id).set(kelly)
