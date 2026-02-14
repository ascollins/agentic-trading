"""Initial schema: orders, fills, positions, balances, audit, experiments.

Revision ID: 001_initial
Revises: None
Create Date: 2024-01-01 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Orders table
    op.create_table(
        "orders",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("order_id", sa.String(64), nullable=False),
        sa.Column("client_order_id", sa.String(128), nullable=True),
        sa.Column("exchange", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("order_type", sa.String(16), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("time_in_force", sa.String(8), nullable=True),
        sa.Column("quantity", sa.Numeric(24, 8), nullable=False),
        sa.Column("filled_quantity", sa.Numeric(24, 8), server_default="0"),
        sa.Column("price", sa.Numeric(24, 8), nullable=True),
        sa.Column("average_fill_price", sa.Numeric(24, 8), nullable=True),
        sa.Column("stop_price", sa.Numeric(24, 8), nullable=True),
        sa.Column("strategy_id", sa.String(64), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_orders_symbol", "orders", ["symbol"])
    op.create_index("ix_orders_status", "orders", ["status"])
    op.create_index("ix_orders_strategy_id", "orders", ["strategy_id"])
    op.create_index("ix_orders_trace_id", "orders", ["trace_id"])
    op.create_index("ix_orders_created_at", "orders", ["created_at"])
    op.create_index("ix_orders_symbol_status", "orders", ["symbol", "status"])

    # Fills table
    op.create_table(
        "fills",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("fill_id", sa.String(64), nullable=False),
        sa.Column("order_id", UUID(as_uuid=True), sa.ForeignKey("orders.id"), nullable=False),
        sa.Column("exchange", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Numeric(24, 8), nullable=False),
        sa.Column("price", sa.Numeric(24, 8), nullable=False),
        sa.Column("fee", sa.Numeric(24, 8), server_default="0"),
        sa.Column("fee_currency", sa.String(16), nullable=True),
        sa.Column("is_maker", sa.Boolean, server_default="false"),
        sa.Column("trade_id", sa.String(64), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("fill_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_fills_order_id", "fills", ["order_id"])
    op.create_index("ix_fills_symbol", "fills", ["symbol"])
    op.create_index("ix_fills_trace_id", "fills", ["trace_id"])
    op.create_index("ix_fills_fill_timestamp", "fills", ["fill_timestamp"])

    # Position snapshots
    op.create_table(
        "position_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("exchange", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Numeric(24, 8), nullable=False),
        sa.Column("entry_price", sa.Numeric(24, 8), nullable=False),
        sa.Column("mark_price", sa.Numeric(24, 8), nullable=True),
        sa.Column("unrealised_pnl", sa.Numeric(24, 8), nullable=True),
        sa.Column("realised_pnl", sa.Numeric(24, 8), nullable=True),
        sa.Column("leverage", sa.Numeric(8, 2), nullable=True),
        sa.Column("margin_mode", sa.String(16), nullable=True),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_position_snapshots_symbol", "position_snapshots", ["symbol"])
    op.create_index("ix_position_snapshots_snapshot_at", "position_snapshots", ["snapshot_at"])
    op.create_index("ix_position_snapshots_symbol_snapshot_at", "position_snapshots", ["symbol", "snapshot_at"])

    # Balance snapshots
    op.create_table(
        "balance_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("exchange", sa.String(32), nullable=False),
        sa.Column("currency", sa.String(16), nullable=False),
        sa.Column("total", sa.Numeric(24, 8), nullable=False),
        sa.Column("free", sa.Numeric(24, 8), nullable=False),
        sa.Column("locked", sa.Numeric(24, 8), nullable=False),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_balance_snapshots_currency", "balance_snapshots", ["currency"])
    op.create_index("ix_balance_snapshots_snapshot_at", "balance_snapshots", ["snapshot_at"])
    op.create_index("ix_balance_snapshots_currency_exchange_snapshot_at", "balance_snapshots", ["currency", "exchange", "snapshot_at"])

    # Decision audit trail
    op.create_table(
        "decision_audit",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=True),
        sa.Column("symbol", sa.String(32), nullable=True),
        sa.Column("feature_vector", JSONB, nullable=True),
        sa.Column("signal", JSONB, nullable=True),
        sa.Column("signal_direction", sa.String(8), nullable=True),
        sa.Column("signal_confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("risk_check", JSONB, nullable=True),
        sa.Column("risk_passed", sa.Boolean, nullable=True),
        sa.Column("order_intent", JSONB, nullable=True),
        sa.Column("order_result", JSONB, nullable=True),
        sa.Column("fill_result", JSONB, nullable=True),
        sa.Column("final_status", sa.String(24), nullable=True),
        sa.Column("decision_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_decision_audit_trace_id", "decision_audit", ["trace_id"])
    op.create_index("ix_decision_audit_strategy_id", "decision_audit", ["strategy_id"])
    op.create_index("ix_decision_audit_symbol", "decision_audit", ["symbol"])
    op.create_index("ix_decision_audit_decision_at", "decision_audit", ["decision_at"])
    op.create_index("ix_decision_audit_strategy_symbol", "decision_audit", ["strategy_id", "symbol"])

    # Experiment log
    op.create_table(
        "experiment_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("config_snapshot", JSONB, nullable=False),
        sa.Column("status", sa.String(16), server_default="running"),
        sa.Column("sharpe_ratio", sa.Numeric(8, 4), nullable=True),
        sa.Column("sortino_ratio", sa.Numeric(8, 4), nullable=True),
        sa.Column("max_drawdown", sa.Numeric(8, 4), nullable=True),
        sa.Column("total_return", sa.Numeric(12, 4), nullable=True),
        sa.Column("win_rate", sa.Numeric(5, 4), nullable=True),
        sa.Column("profit_factor", sa.Numeric(8, 4), nullable=True),
        sa.Column("total_trades", sa.Integer, nullable=True),
        sa.Column("equity_curve", JSONB, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("git_sha", sa.String(40), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_experiment_log_strategy_id", "experiment_log", ["strategy_id"])
    op.create_index("ix_experiment_log_status", "experiment_log", ["status"])
    op.create_index("ix_experiment_log_started_at", "experiment_log", ["started_at"])


def downgrade() -> None:
    op.drop_table("experiment_log")
    op.drop_table("decision_audit")
    op.drop_table("balance_snapshots")
    op.drop_table("position_snapshots")
    op.drop_table("fills")
    op.drop_table("orders")
