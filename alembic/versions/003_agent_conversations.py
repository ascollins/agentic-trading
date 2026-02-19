"""Add agent_conversations and agent_messages tables for Soteria reasoning.

Revision ID: 003_agent_conversations
Revises: 002_trade_journal
Create Date: 2024-01-20 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "003_agent_conversations"
down_revision: Union[str, None] = "002_trade_journal"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # agent_conversations — one row per reasoning conversation
    # ------------------------------------------------------------------
    op.create_table(
        "agent_conversations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", sa.String(64), unique=True, nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False, server_default=""),
        sa.Column("timeframe", sa.String(8), nullable=False, server_default=""),
        sa.Column("trigger_event", sa.String(128), nullable=False, server_default=""),
        sa.Column("strategy_id", sa.String(64), nullable=False, server_default=""),
        sa.Column("outcome", sa.String(32), nullable=False, server_default="no_trade"),
        sa.Column("outcome_details", JSONB, nullable=True),
        sa.Column("context_snapshot", JSONB, nullable=True),
        sa.Column("has_veto", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("has_disagreement", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("message_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("trace_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Numeric(12, 2), nullable=True),
        sa.Column("traces_json", JSONB, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Indexes for common queries
    op.create_index("ix_agentconv_conversation_id", "agent_conversations", ["conversation_id"])
    op.create_index("ix_agentconv_symbol", "agent_conversations", ["symbol"])
    op.create_index("ix_agentconv_strategy_id", "agent_conversations", ["strategy_id"])
    op.create_index("ix_agentconv_outcome", "agent_conversations", ["outcome"])
    op.create_index("ix_agentconv_has_veto", "agent_conversations", ["has_veto"])
    op.create_index("ix_agentconv_has_disagreement", "agent_conversations", ["has_disagreement"])
    op.create_index("ix_agentconv_started_at", "agent_conversations", ["started_at"])
    op.create_index(
        "ix_agentconv_symbol_outcome",
        "agent_conversations",
        ["symbol", "outcome"],
    )

    # ------------------------------------------------------------------
    # agent_messages — one row per inter-agent message
    # ------------------------------------------------------------------
    op.create_table(
        "agent_messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("message_id", sa.String(64), unique=True, nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("sender", sa.String(32), nullable=False),
        sa.Column("recipients", JSONB, nullable=True),
        sa.Column("message_type", sa.String(32), nullable=False),
        sa.Column("content", sa.Text, nullable=False, server_default=""),
        sa.Column("structured_data", JSONB, nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False, server_default="0"),
        sa.Column("references", JSONB, nullable=True),
        sa.Column("metadata_json", JSONB, nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Indexes
    op.create_index("ix_agentmsg_message_id", "agent_messages", ["message_id"])
    op.create_index("ix_agentmsg_conversation_id", "agent_messages", ["conversation_id"])
    op.create_index("ix_agentmsg_sender", "agent_messages", ["sender"])
    op.create_index("ix_agentmsg_message_type", "agent_messages", ["message_type"])
    op.create_index("ix_agentmsg_timestamp", "agent_messages", ["timestamp"])
    op.create_index(
        "ix_agentmsg_conv_sender",
        "agent_messages",
        ["conversation_id", "sender"],
    )


def downgrade() -> None:
    op.drop_table("agent_messages")
    op.drop_table("agent_conversations")
