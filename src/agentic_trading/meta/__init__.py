"""Platform-wide Metadata Standard.

Public API
----------
Models:
    MetadataStandard, Lineage, MetadataScope

Enums:
    SourceType, Classification, TrustLevel, MetadataStatus

Factories:
    metadata_for_agent, metadata_for_market_data,
    metadata_for_human, metadata_for_derived

Helpers:
    compute_effective_relevance, TRUST_WEIGHTS
"""

from agentic_trading.meta.metadata import (
    TRUST_WEIGHTS,
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

__all__ = [
    # Enums
    "Classification",
    "MetadataStatus",
    "SourceType",
    "TrustLevel",
    # Models
    "Lineage",
    "MetadataScope",
    "MetadataStandard",
    # Factories
    "metadata_for_agent",
    "metadata_for_derived",
    "metadata_for_human",
    "metadata_for_market_data",
    # Helpers
    "TRUST_WEIGHTS",
    "compute_effective_relevance",
]
