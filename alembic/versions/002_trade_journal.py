"""Add trade_journal table for Edgewonk-inspired trade analytics persistence.

Revision ID: 002_trade_journal
Revises: 001_initial
Create Date: 2024-01-15 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "002_trade_journal"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trade_journal",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trade_id", sa.String(64), unique=True, nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),

        # Phase & outcome
        sa.Column("phase", sa.String(16), nullable=False),
        sa.Column("outcome", sa.String(16), nullable=True),

        # Prices
        sa.Column("avg_entry_price", sa.Numeric(24, 8), nullable=True),
        sa.Column("avg_exit_price", sa.Numeric(24, 8), nullable=True),
        sa.Column("initial_risk_price", sa.Numeric(24, 8), nullable=True),
        sa.Column("planned_target_price", sa.Numeric(24, 8), nullable=True),

        # Quantities
        sa.Column("total_entry_qty", sa.Numeric(24, 8), server_default="0"),
        sa.Column("total_exit_qty", sa.Numeric(24, 8), server_default="0"),

        # PnL
        sa.Column("gross_pnl", sa.Numeric(24, 8), server_default="0"),
        sa.Column("total_fees", sa.Numeric(24, 8), server_default="0"),
        sa.Column("net_pnl", sa.Numeric(24, 8), server_default="0"),

        # Risk metrics
        sa.Column("initial_risk_amount", sa.Numeric(24, 8), nullable=True),
        sa.Column("r_multiple", sa.Numeric(10, 4), server_default="0"),
        sa.Column("mae", sa.Numeric(24, 8), server_default="0"),
        sa.Column("mfe", sa.Numeric(24, 8), server_default="0"),
        sa.Column("mae_price", sa.Numeric(24, 8), server_default="0"),
        sa.Column("mfe_price", sa.Numeric(24, 8), server_default="0"),
        sa.Column("mae_r", sa.Numeric(10, 4), server_default="0"),
        sa.Column("mfe_r", sa.Numeric(10, 4), server_default="0"),
        sa.Column("management_efficiency", sa.Numeric(10, 4), server_default="0"),

        # Signal & governance
        sa.Column("signal_confidence", sa.Numeric(8, 4), server_default="0"),
        sa.Column("health_score_at_entry", sa.Numeric(8, 4), server_default="1"),
        sa.Column("governance_sizing_multiplier", sa.Numeric(8, 4), server_default="1"),

        # Duration
        sa.Column("hold_duration_seconds", sa.Numeric(16, 2), server_default="0"),

        # Timestamps
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),

        # Fill & mark data (JSONB)
        sa.Column("entry_fills_json", JSONB, nullable=True),
        sa.Column("exit_fills_json", JSONB, nullable=True),
        sa.Column("mark_samples_json", JSONB, nullable=True),

        # Classification
        sa.Column("tags", JSONB, nullable=True),
        sa.Column("mistakes", JSONB, nullable=True),

        # Metadata
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Indexes
    op.create_index("ix_tradejournal_trade_id", "trade_journal", ["trade_id"])
    op.create_index("ix_tradejournal_trace_id", "trade_journal", ["trace_id"])
    op.create_index("ix_tradejournal_strategy_id", "trade_journal", ["strategy_id"])
    op.create_index("ix_tradejournal_symbol", "trade_journal", ["symbol"])
    op.create_index("ix_tradejournal_phase", "trade_journal", ["phase"])
    op.create_index("ix_tradejournal_outcome", "trade_journal", ["outcome"])
    op.create_index("ix_tradejournal_opened_at", "trade_journal", ["opened_at"])
    op.create_index("ix_tradejournal_closed_at", "trade_journal", ["closed_at"])
    op.create_index("ix_tradejournal_strategy_symbol", "trade_journal", ["strategy_id", "symbol"])


def downgrade() -> None:
    op.drop_table("trade_journal")
