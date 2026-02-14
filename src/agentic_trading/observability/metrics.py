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


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus metrics HTTP server in a background thread."""
    SYSTEM_INFO.info({
        "version": "0.1.0",
        "mode": "unknown",  # Set at startup
    })
    start_http_server(port)
