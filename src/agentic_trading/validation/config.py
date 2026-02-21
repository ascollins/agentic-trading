"""Validation framework configuration.

Per-output-type configuration for validator behaviour,
thresholds, and remediation policies.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .critique_validator import CritiqueTriggerConfig
from .remediation import RemediationPolicy


class ValidationConfig(BaseModel):
    """Top-level validation framework configuration."""

    enabled: bool = True
    # Evidence validation
    max_uncited_ratio: float = 0.3
    # Critique
    critique: CritiqueTriggerConfig = Field(
        default_factory=CritiqueTriggerConfig
    )
    # Remediation policies (per output type)
    remediation_policies: list[RemediationPolicy] = Field(
        default_factory=list
    )
    # Output types that skip validation entirely
    skip_types: list[str] = Field(default_factory=list)
