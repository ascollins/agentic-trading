"""Unit tests for the platform-wide Metadata Standard."""

from __future__ import annotations

from datetime import timedelta, timezone  # noqa: UP017

from agentic_trading.core.ids import utc_now as _now
from agentic_trading.meta.metadata import (
    Classification,
    Lineage,
    MetadataScope,
    MetadataStandard,
    MetadataStatus,
    SourceType,
    TrustLevel,
    compute_effective_relevance,
    metadata_for_agent,
    metadata_for_derived,
    metadata_for_human,
    metadata_for_market_data,
)

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestSourceType:
    """SourceType enum tests."""

    def test_has_eight_members(self):
        assert len(SourceType) == 8

    def test_values_are_lowercase(self):
        for member in SourceType:
            assert member.value == member.value.lower()

    def test_expected_members(self):
        expected = {
            "agent", "strategy", "market_data", "human",
            "system", "derived", "external", "policy",
        }
        actual = {m.value for m in SourceType}
        assert actual == expected


class TestClassification:
    """Classification enum tests."""

    def test_has_four_members(self):
        assert len(Classification) == 4

    def test_expected_members(self):
        expected = {"public", "internal", "confidential", "restricted"}
        actual = {m.value for m in Classification}
        assert actual == expected


class TestTrustLevel:
    """TrustLevel enum tests."""

    def test_has_five_members(self):
        assert len(TrustLevel) == 5

    def test_expected_members(self):
        expected = {"verified", "high", "medium", "low", "untrusted"}
        actual = {m.value for m in TrustLevel}
        assert actual == expected


class TestMetadataStatus:
    """MetadataStatus enum tests."""

    def test_has_five_members(self):
        assert len(MetadataStatus) == 5

    def test_expected_members(self):
        expected = {"active", "deprecated", "do_not_use", "archived", "expired"}
        actual = {m.value for m in MetadataStatus}
        assert actual == expected

    def test_values_are_lowercase(self):
        for member in MetadataStatus:
            assert member.value == member.value.lower()


# ---------------------------------------------------------------------------
# Sub-model tests
# ---------------------------------------------------------------------------


class TestLineage:
    """Lineage model tests."""

    def test_defaults(self):
        lineage = Lineage()
        assert lineage.parent_id == ""
        assert lineage.parent_type == ""
        assert lineage.root_id == ""
        assert lineage.derivation == ""
        assert lineage.chain_depth == 0
        assert lineage.chain_ids == []

    def test_chain_depth_tracking(self):
        lineage = Lineage(
            parent_id="parent-1",
            root_id="root-1",
            chain_depth=3,
            chain_ids=["root-1", "mid-1", "parent-1"],
        )
        assert lineage.chain_depth == 3
        assert len(lineage.chain_ids) == 3


class TestMetadataScope:
    """MetadataScope model tests."""

    def test_defaults_empty_lists(self):
        scope = MetadataScope()
        assert scope.strategy_ids == []
        assert scope.symbols == []
        assert scope.exchanges == []
        assert scope.asset_classes == []

    def test_tenant_id_default(self):
        scope = MetadataScope()
        assert scope.tenant_id == "default"


# ---------------------------------------------------------------------------
# MetadataStandard tests
# ---------------------------------------------------------------------------


class TestMetadataStandard:
    """MetadataStandard model tests."""

    def test_creation_with_defaults(self):
        meta = MetadataStandard()
        assert meta.source_id != ""
        assert meta.source_type == SourceType.SYSTEM
        assert meta.author == ""
        assert meta.classification == Classification.INTERNAL
        assert meta.quality_score == 1.0
        assert meta.trust_level == TrustLevel.HIGH
        assert meta.status == MetadataStatus.ACTIVE
        assert meta.expiry is None
        assert meta.deprecated_at is None
        assert meta.deprecated_reason == ""
        assert meta.superseded_by == ""
        assert meta.version == 1
        assert meta.content_hash == ""

    def test_unique_source_ids(self):
        m1 = MetadataStandard()
        m2 = MetadataStandard()
        assert m1.source_id != m2.source_id

    def test_timestamps_are_utc(self):
        meta = MetadataStandard()
        assert meta.created_at.tzinfo is not None
        assert meta.created_at.tzinfo == timezone.utc  # noqa: UP017
        assert meta.updated_at.tzinfo is not None

    def test_is_usable_active(self):
        meta = MetadataStandard(status=MetadataStatus.ACTIVE)
        assert meta.is_usable() is True

    def test_is_usable_do_not_use(self):
        meta = MetadataStandard(status=MetadataStatus.DO_NOT_USE)
        assert meta.is_usable() is False

    def test_is_usable_archived(self):
        meta = MetadataStandard(status=MetadataStatus.ARCHIVED)
        assert meta.is_usable() is False

    def test_is_usable_expired_status(self):
        meta = MetadataStandard(status=MetadataStatus.EXPIRED)
        assert meta.is_usable() is False

    def test_is_usable_expired_by_time(self):
        past = _now() - timedelta(hours=1)
        meta = MetadataStandard(expiry=past)
        assert meta.is_usable() is False

    def test_is_usable_future_expiry(self):
        future = _now() + timedelta(hours=1)
        meta = MetadataStandard(expiry=future)
        assert meta.is_usable() is True

    def test_is_usable_untrusted(self):
        meta = MetadataStandard(trust_level=TrustLevel.UNTRUSTED)
        assert meta.is_usable() is False

    def test_is_deprecated_active(self):
        meta = MetadataStandard(status=MetadataStatus.ACTIVE)
        assert meta.is_deprecated() is False

    def test_is_deprecated_deprecated(self):
        meta = MetadataStandard(status=MetadataStatus.DEPRECATED)
        assert meta.is_deprecated() is True

    def test_deprecate_returns_new_copy(self):
        original = MetadataStandard()
        deprecated = original.deprecate(
            reason="Replaced by v2",
            superseded_by="new-source-123",
        )
        # Original unchanged
        assert original.status == MetadataStatus.ACTIVE
        # New copy is deprecated
        assert deprecated.status == MetadataStatus.DEPRECATED
        assert deprecated.deprecated_reason == "Replaced by v2"
        assert deprecated.superseded_by == "new-source-123"
        assert deprecated.deprecated_at is not None
        # Still usable (deprecated != blocked)
        assert deprecated.is_usable() is True
        assert deprecated.is_deprecated() is True

    def test_mark_do_not_use(self):
        original = MetadataStandard()
        blocked = original.mark_do_not_use(reason="Contains stale data")
        assert original.status == MetadataStatus.ACTIVE
        assert blocked.status == MetadataStatus.DO_NOT_USE
        assert blocked.deprecated_reason == "Contains stale data"
        assert blocked.deprecated_at is not None
        assert blocked.is_usable() is False

    def test_serialization_roundtrip(self):
        meta = MetadataStandard(
            source_type=SourceType.AGENT,
            author="cmt-analyst-01",
            classification=Classification.CONFIDENTIAL,
            quality_score=0.85,
            trust_level=TrustLevel.MEDIUM,
            version=2,
            scope=MetadataScope(
                symbols=["BTCUSDT"],
                exchanges=["bybit"],
            ),
            lineage=Lineage(
                parent_id="parent-1",
                root_id="root-1",
                chain_depth=1,
                chain_ids=["root-1"],
            ),
        )
        json_str = meta.model_dump_json()
        restored = MetadataStandard.model_validate_json(json_str)

        assert restored.source_id == meta.source_id
        assert restored.source_type == SourceType.AGENT
        assert restored.author == "cmt-analyst-01"
        assert restored.quality_score == 0.85
        assert restored.trust_level == TrustLevel.MEDIUM
        assert restored.version == 2
        assert restored.scope.symbols == ["BTCUSDT"]
        assert restored.lineage.chain_depth == 1

    def test_model_dump_includes_all_fields(self):
        meta = MetadataStandard()
        dumped = meta.model_dump()
        expected_fields = {
            "source_id", "source_type", "author",
            "created_at", "updated_at",
            "scope", "classification",
            "quality_score", "trust_level",
            "status", "expiry", "deprecated_at",
            "deprecated_reason", "superseded_by",
            "version", "lineage", "content_hash",
        }
        assert set(dumped.keys()) == expected_fields


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


class TestFactoryHelpers:
    """Factory helper tests."""

    def test_metadata_for_agent(self):
        meta = metadata_for_agent("cmt-analyst-01", "cmt_analyst")
        assert meta.source_type == SourceType.AGENT
        assert meta.author == "cmt-analyst-01"
        assert meta.trust_level == TrustLevel.MEDIUM

    def test_metadata_for_market_data_scope(self):
        meta = metadata_for_market_data("bybit", "BTCUSDT")
        assert meta.source_type == SourceType.MARKET_DATA
        assert meta.author == "bybit"
        assert meta.trust_level == TrustLevel.HIGH
        assert meta.scope.exchanges == ["bybit"]
        assert meta.scope.symbols == ["BTCUSDT"]

    def test_metadata_for_human(self):
        meta = metadata_for_human("analyst-1")
        assert meta.source_type == SourceType.HUMAN
        assert meta.author == "human:analyst-1"
        assert meta.trust_level == TrustLevel.HIGH
        assert meta.classification == Classification.INTERNAL

    def test_metadata_for_derived_lineage(self):
        parent = MetadataStandard(
            source_type=SourceType.MARKET_DATA,
            author="bybit",
            trust_level=TrustLevel.HIGH,
            classification=Classification.INTERNAL,
        )
        child = metadata_for_derived(parent, derivation="aggregated")

        assert child.source_type == SourceType.DERIVED
        assert child.author == "bybit"
        assert child.trust_level == TrustLevel.HIGH
        assert child.classification == Classification.INTERNAL
        assert child.lineage.parent_id == parent.source_id
        assert child.lineage.parent_type == "market_data"
        assert child.lineage.derivation == "aggregated"

    def test_metadata_for_derived_increments_depth(self):
        root = MetadataStandard(source_type=SourceType.HUMAN)
        child = metadata_for_derived(root)
        grandchild = metadata_for_derived(child)

        assert child.lineage.chain_depth == 1
        assert grandchild.lineage.chain_depth == 2

    def test_metadata_for_derived_preserves_root_id(self):
        root = MetadataStandard(source_type=SourceType.HUMAN)
        child = metadata_for_derived(root)
        grandchild = metadata_for_derived(child)

        assert child.lineage.root_id == root.source_id
        assert grandchild.lineage.root_id == root.source_id


# ---------------------------------------------------------------------------
# Trust-weighted relevance
# ---------------------------------------------------------------------------


class TestComputeEffectiveRelevance:
    """compute_effective_relevance() tests."""

    def test_verified_full_quality(self):
        result = compute_effective_relevance(1.0, TrustLevel.VERIFIED, 1.0)
        assert result == 1.0

    def test_untrusted_returns_zero(self):
        result = compute_effective_relevance(0.9, TrustLevel.UNTRUSTED, 1.0)
        assert result == 0.0

    def test_medium_trust_reduces_relevance(self):
        result = compute_effective_relevance(1.0, TrustLevel.MEDIUM, 1.0)
        assert result == 0.8

    def test_low_quality_reduces_relevance(self):
        result = compute_effective_relevance(1.0, TrustLevel.VERIFIED, 0.5)
        assert result == 0.5

    def test_combined_reduction(self):
        result = compute_effective_relevance(0.85, TrustLevel.MEDIUM, 0.9)
        expected = 0.85 * 0.8 * 0.9
        assert abs(result - expected) < 1e-9
