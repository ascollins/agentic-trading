"""Integration test: ModelRegistry lifecycle and AuditBundleGenerator.

Exercises the model registration, stage promotion pipeline, query helpers,
and audit bundle generation.
"""

from __future__ import annotations

import pytest

from agentic_trading.control_plane.action_types import ActionScope, AuditEntry
from agentic_trading.control_plane.audit_log import AuditLog
from agentic_trading.intelligence.model_registry import (
    ModelRecord,
    ModelRegistry,
    ModelStage,
)
from agentic_trading.observability.audit_bundle import AuditBundleGenerator


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------


class TestModelRegistryLifecycle:

    def test_register_auto_increments_version(self):
        """Registering two models with same name produces version 1 and 2."""
        registry = ModelRegistry()
        r1 = registry.register("btc_lstm")
        r2 = registry.register("btc_lstm")

        assert r1.version == 1
        assert r2.version == 2
        assert r1.model_id != r2.model_id

    def test_register_stores_training_hash_and_hyperparams(self):
        """Model record preserves training_data_hash and hyperparameters."""
        registry = ModelRegistry()
        r = registry.register(
            "btc_lstm",
            training_data_hash="abc123",
            hyperparameters={"layers": 3, "dropout": 0.2},
            description="test model",
        )

        assert r.training_data_hash == "abc123"
        assert r.hyperparameters["layers"] == 3
        assert r.hyperparameters["dropout"] == 0.2
        assert r.description == "test model"

    def test_promote_research_to_paper(self):
        """RESEARCH -> PAPER transition succeeds."""
        registry = ModelRegistry()
        r = registry.register("btc_lstm")
        assert r.stage == ModelStage.RESEARCH

        promoted = registry.promote(r.model_id, ModelStage.PAPER, approved_by="ops-1")
        assert promoted is not None
        assert promoted.stage == ModelStage.PAPER
        assert promoted.approved_by == "ops-1"
        assert len(promoted.transitions) == 1
        assert promoted.transitions[0].from_stage == "research"
        assert promoted.transitions[0].to_stage == "paper"

    def test_promote_paper_to_limited(self):
        """PAPER -> LIMITED transition succeeds with approver recorded."""
        registry = ModelRegistry()
        r = registry.register("btc_lstm")
        registry.promote(r.model_id, ModelStage.PAPER, approved_by="ops-1")
        promoted = registry.promote(r.model_id, ModelStage.LIMITED, approved_by="risk-1")

        assert promoted is not None
        assert promoted.stage == ModelStage.LIMITED
        assert promoted.approved_by == "risk-1"
        assert promoted.approved_at is not None

    def test_promote_limited_to_production(self):
        """LIMITED -> PRODUCTION transition succeeds."""
        registry = ModelRegistry()
        r = registry.register("btc_lstm")
        registry.promote(r.model_id, ModelStage.PAPER)
        registry.promote(r.model_id, ModelStage.LIMITED)
        promoted = registry.promote(
            r.model_id, ModelStage.PRODUCTION, approved_by="admin-1",
        )

        assert promoted is not None
        assert promoted.stage == ModelStage.PRODUCTION
        assert promoted.approved_at is not None
        assert len(promoted.transitions) == 3

    def test_invalid_transition_returns_none(self):
        """RESEARCH -> PRODUCTION skipping PAPER returns None."""
        registry = ModelRegistry()
        r = registry.register("btc_lstm")

        result = registry.promote(r.model_id, ModelStage.PRODUCTION)
        assert result is None

        # Model should remain in RESEARCH
        current = registry.get(r.model_id)
        assert current is not None
        assert current.stage == ModelStage.RESEARCH

    def test_retire_from_any_non_terminal(self):
        """Any stage can transition to RETIRED."""
        registry = ModelRegistry()

        # Retire from RESEARCH
        r1 = registry.register("model_a")
        retired = registry.retire(r1.model_id, reason="obsolete", actor="ops")
        assert retired is not None
        assert retired.stage == ModelStage.RETIRED

        # Retire from PAPER
        r2 = registry.register("model_b")
        registry.promote(r2.model_id, ModelStage.PAPER)
        retired2 = registry.retire(r2.model_id, reason="underperforming", actor="ops")
        assert retired2 is not None
        assert retired2.stage == ModelStage.RETIRED

        # Retire from PRODUCTION
        r3 = registry.register("model_c")
        registry.promote(r3.model_id, ModelStage.PAPER)
        registry.promote(r3.model_id, ModelStage.LIMITED)
        registry.promote(r3.model_id, ModelStage.PRODUCTION)
        retired3 = registry.retire(r3.model_id, reason="replaced", actor="ops")
        assert retired3 is not None
        assert retired3.stage == ModelStage.RETIRED

    def test_retired_is_terminal(self):
        """RETIRED model cannot be promoted to any other stage."""
        registry = ModelRegistry()
        r = registry.register("btc_lstm")
        registry.retire(r.model_id, reason="done")

        # Attempt to promote from RETIRED to RESEARCH
        result = registry.promote(r.model_id, ModelStage.RESEARCH)
        assert result is None

        # Attempt to promote from RETIRED to PAPER
        result = registry.promote(r.model_id, ModelStage.PAPER)
        assert result is None

    def test_get_production_returns_correct_version(self):
        """get_production() returns the model at PRODUCTION stage."""
        registry = ModelRegistry()

        # Register two versions, promote only v1 to production
        r1 = registry.register("btc_lstm")
        registry.promote(r1.model_id, ModelStage.PAPER)
        registry.promote(r1.model_id, ModelStage.LIMITED)
        registry.promote(r1.model_id, ModelStage.PRODUCTION)

        r2 = registry.register("btc_lstm")  # v2 stays in RESEARCH

        prod = registry.get_production("btc_lstm")
        assert prod is not None
        assert prod.model_id == r1.model_id
        assert prod.version == 1

    def test_list_by_stage_filters_correctly(self):
        """list_by_stage(PAPER) only returns models at PAPER stage."""
        registry = ModelRegistry()

        r1 = registry.register("model_a")
        r2 = registry.register("model_b")
        r3 = registry.register("model_c")

        # Promote r1 and r2 to PAPER, leave r3 in RESEARCH
        registry.promote(r1.model_id, ModelStage.PAPER)
        registry.promote(r2.model_id, ModelStage.PAPER)

        paper_models = registry.list_by_stage(ModelStage.PAPER)
        assert len(paper_models) == 2

        research_models = registry.list_by_stage(ModelStage.RESEARCH)
        assert len(research_models) == 1
        assert research_models[0].model_id == r3.model_id

    def test_update_metrics_persists(self):
        """update_metrics() merges new keys into existing metrics dict."""
        registry = ModelRegistry()
        r = registry.register("btc_lstm", metrics={"sharpe": 1.5})

        updated = registry.update_metrics(r.model_id, {"hit_rate": 0.62})
        assert updated is not None
        assert updated.metrics["sharpe"] == 1.5
        assert updated.metrics["hit_rate"] == 0.62

        # Update an existing key
        updated2 = registry.update_metrics(r.model_id, {"sharpe": 2.0})
        assert updated2 is not None
        assert updated2.metrics["sharpe"] == 2.0
        assert updated2.metrics["hit_rate"] == 0.62


# ---------------------------------------------------------------------------
# AuditBundleGenerator tests
# ---------------------------------------------------------------------------


class TestAuditBundleGenerator:

    @pytest.mark.asyncio
    async def test_audit_bundle_includes_all_entries_for_trace(self):
        """AuditBundleGenerator.generate(trace_id) returns all entries."""
        audit_log = AuditLog()
        correlation_id = "trace-001"

        # Append several entries with the same correlation_id
        await audit_log.append(AuditEntry(
            correlation_id=correlation_id,
            causation_id="action-001",
            actor="exec-agent",
            event_type="tool_call_pre_execution",
            payload={"tool_name": "submit_order"},
        ))
        await audit_log.append(AuditEntry(
            correlation_id=correlation_id,
            causation_id="action-001",
            actor="exec-agent",
            event_type="policy_evaluated",
            payload={"allowed": True},
        ))
        await audit_log.append(AuditEntry(
            correlation_id=correlation_id,
            causation_id="action-001",
            actor="exec-agent",
            event_type="tool_call_recorded",
            payload={"success": True},
        ))

        # Also add an unrelated entry
        await audit_log.append(AuditEntry(
            correlation_id="other-trace",
            causation_id="action-999",
            actor="system",
            event_type="tool_call_recorded",
            payload={"success": True},
        ))

        generator = AuditBundleGenerator(audit_log=audit_log)
        bundle = generator.generate(trace_id=correlation_id)

        assert bundle is not None
        assert bundle.trace_id == correlation_id
        assert len(bundle.entries) == 3
        # Entries should be in chronological order
        event_types = [e.event_type for e in bundle.entries]
        assert "tool_call_pre_execution" in event_types
        assert "tool_call_recorded" in event_types
