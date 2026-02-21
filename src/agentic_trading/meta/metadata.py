"""Platform-wide Metadata Standard.

Composable metadata contract for all platform entities.  Designed as a
mixin-style model: embed as a field on existing models rather than inheriting.

    class MyEntity(BaseModel):
        ...
        metadata: MetadataStandard | None = None

Defines 12 canonical fields, deprecation lifecycle, trust-weighted relevance,
and lineage provenance chains.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from agentic_trading.core.ids import new_id as _uuid
from agentic_trading.core.ids import utc_now as _now

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """Classification of what produced this data."""

    AGENT = "agent"
    STRATEGY = "strategy"
    MARKET_DATA = "market_data"
    HUMAN = "human"
    SYSTEM = "system"
    DERIVED = "derived"
    EXTERNAL = "external"
    POLICY = "policy"


class Classification(str, Enum):
    """Data sensitivity classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class TrustLevel(str, Enum):
    """Trust level of the source."""

    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNTRUSTED = "untrusted"


class MetadataStatus(str, Enum):
    """Lifecycle status of the metadata-bearing entity."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DO_NOT_USE = "do_not_use"
    ARCHIVED = "archived"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Lineage(BaseModel):
    """Provenance chain — where this data came from."""

    parent_id: str = ""
    parent_type: str = ""
    root_id: str = ""
    derivation: str = ""
    chain_depth: int = 0
    chain_ids: list[str] = Field(default_factory=list)


class MetadataScope(BaseModel):
    """Jurisdiction / applicability scope."""

    strategy_ids: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    exchanges: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)
    tenant_id: str = "default"


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class MetadataStandard(BaseModel):
    """Composable metadata contract for all platform entities.

    Embed as ``metadata: MetadataStandard | None = None`` on any model.
    """

    # Identity
    source_id: str = Field(default_factory=_uuid)
    source_type: SourceType = SourceType.SYSTEM
    author: str = ""

    # Timestamps
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    # Scope & classification
    scope: MetadataScope = Field(default_factory=MetadataScope)
    classification: Classification = Classification.INTERNAL

    # Quality & trust
    quality_score: float = 1.0
    trust_level: TrustLevel = TrustLevel.HIGH

    # Lifecycle
    status: MetadataStatus = MetadataStatus.ACTIVE
    expiry: datetime | None = None
    deprecated_at: datetime | None = None
    deprecated_reason: str = ""
    superseded_by: str = ""

    # Versioning
    version: int = 1

    # Provenance
    lineage: Lineage = Field(default_factory=Lineage)

    # Integrity
    content_hash: str = ""

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def is_usable(self) -> bool:
        """Return ``True`` if the entity is safe for retrieval / decisions."""
        if self.status in (
            MetadataStatus.DO_NOT_USE,
            MetadataStatus.ARCHIVED,
            MetadataStatus.EXPIRED,
        ):
            return False
        if self.expiry is not None and _now() > self.expiry:
            return False
        return self.trust_level != TrustLevel.UNTRUSTED

    def is_deprecated(self) -> bool:
        """Return ``True`` if the entity is deprecated (still usable)."""
        return self.status == MetadataStatus.DEPRECATED

    def deprecate(
        self,
        reason: str,
        superseded_by: str = "",
    ) -> MetadataStandard:
        """Return a copy marked as deprecated."""
        return self.model_copy(
            update={
                "status": MetadataStatus.DEPRECATED,
                "deprecated_at": _now(),
                "deprecated_reason": reason,
                "superseded_by": superseded_by,
                "updated_at": _now(),
            },
        )

    def mark_do_not_use(self, reason: str) -> MetadataStandard:
        """Return a copy marked as do-not-use (hard filter)."""
        return self.model_copy(
            update={
                "status": MetadataStatus.DO_NOT_USE,
                "deprecated_at": _now(),
                "deprecated_reason": reason,
                "updated_at": _now(),
            },
        )


# ---------------------------------------------------------------------------
# Trust-weighted relevance helper
# ---------------------------------------------------------------------------

TRUST_WEIGHTS: dict[TrustLevel, float] = {
    TrustLevel.VERIFIED: 1.0,
    TrustLevel.HIGH: 0.95,
    TrustLevel.MEDIUM: 0.8,
    TrustLevel.LOW: 0.5,
    TrustLevel.UNTRUSTED: 0.0,
}


def compute_effective_relevance(
    base_relevance: float,
    trust_level: TrustLevel,
    quality_score: float,
) -> float:
    """Combine base relevance with trust and quality.

    ``effective = base_relevance * trust_weight * quality_score``
    """
    weight = TRUST_WEIGHTS.get(trust_level, 0.8)
    return base_relevance * weight * quality_score


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def metadata_for_agent(
    agent_id: str,
    agent_type: str = "",
) -> MetadataStandard:
    """Create metadata for an agent-generated artifact."""
    return MetadataStandard(
        source_type=SourceType.AGENT,
        author=agent_id,
        trust_level=TrustLevel.MEDIUM,
    )


def metadata_for_market_data(
    exchange: str,
    symbol: str,
) -> MetadataStandard:
    """Create metadata for market data."""
    return MetadataStandard(
        source_type=SourceType.MARKET_DATA,
        author=exchange,
        trust_level=TrustLevel.HIGH,
        scope=MetadataScope(exchanges=[exchange], symbols=[symbol]),
    )


def metadata_for_human(user_id: str) -> MetadataStandard:
    """Create metadata for human-authored content."""
    return MetadataStandard(
        source_type=SourceType.HUMAN,
        author=f"human:{user_id}",
        trust_level=TrustLevel.HIGH,
        classification=Classification.INTERNAL,
    )


def metadata_for_derived(
    parent: MetadataStandard,
    derivation: str = "transformed",
) -> MetadataStandard:
    """Create metadata for a derived entity, inheriting from *parent*."""
    chain = list(parent.lineage.chain_ids) + [parent.source_id]
    return MetadataStandard(
        source_type=SourceType.DERIVED,
        author=parent.author,
        trust_level=parent.trust_level,
        classification=parent.classification,
        scope=parent.scope.model_copy(),
        lineage=Lineage(
            parent_id=parent.source_id,
            parent_type=parent.source_type.value,
            root_id=parent.lineage.root_id or parent.source_id,
            derivation=derivation,
            chain_depth=parent.lineage.chain_depth + 1,
            chain_ids=chain,
        ),
    )
