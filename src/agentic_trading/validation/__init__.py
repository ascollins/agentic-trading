"""Validation Framework — multi-layer hallucination detection for LLM outputs.

Public API
----------
Models:
    ValidationResult, ValidationIssue, ValidationSeverity, ValidationLayer,
    ClaimAnnotation, ClaimType, CritiqueResult,
    RemediationAction, RemediationState, RemediationRecord

Validators:
    IValidator, SchemaValidator, EvidenceValidator,
    BusinessRuleValidator, CritiqueValidator

Business Rules:
    BusinessRule, BusinessRuleSet, BusinessRuleType,
    build_signal_rules, build_cmt_rules

Pipeline:
    ValidationPipeline

Remediation:
    RemediationEngine, RemediationPolicy

Config:
    ValidationConfig, CritiqueTriggerConfig

Errors:
    ValidationError, SchemaValidationError, EvidenceValidationError,
    BusinessRuleValidationError, CritiqueValidationError,
    RemediationExhaustedError
"""

from agentic_trading.validation.business_rules import (
    BusinessRule,
    BusinessRuleSet,
    BusinessRuleType,
    BusinessRuleValidator,
    build_cmt_rules,
    build_signal_rules,
)
from agentic_trading.validation.config import ValidationConfig
from agentic_trading.validation.critique_validator import (
    CritiqueTriggerConfig,
    CritiqueValidator,
)
from agentic_trading.validation.errors import (
    BusinessRuleValidationError,
    CritiqueValidationError,
    EvidenceValidationError,
    RemediationExhaustedError,
    SchemaValidationError,
    ValidationError,
)
from agentic_trading.validation.evidence_validator import EvidenceValidator
from agentic_trading.validation.models import (
    ClaimAnnotation,
    ClaimType,
    CritiqueResult,
    RemediationAction,
    RemediationRecord,
    RemediationState,
    ValidationIssue,
    ValidationLayer,
    ValidationResult,
    ValidationSeverity,
)
from agentic_trading.validation.pipeline import ValidationPipeline
from agentic_trading.validation.protocol import IValidator
from agentic_trading.validation.remediation import (
    RemediationEngine,
    RemediationPolicy,
)
from agentic_trading.validation.schema_validator import SchemaValidator

__all__ = [
    # Enums
    "BusinessRuleType",
    "ClaimType",
    "RemediationAction",
    "RemediationState",
    "ValidationLayer",
    "ValidationSeverity",
    # Models
    "BusinessRule",
    "BusinessRuleSet",
    "ClaimAnnotation",
    "CritiqueResult",
    "RemediationRecord",
    "ValidationIssue",
    "ValidationResult",
    # Protocol
    "IValidator",
    # Validators
    "BusinessRuleValidator",
    "CritiqueValidator",
    "EvidenceValidator",
    "SchemaValidator",
    # Pipeline
    "ValidationPipeline",
    # Remediation
    "RemediationEngine",
    "RemediationPolicy",
    # Config
    "CritiqueTriggerConfig",
    "ValidationConfig",
    # Factories
    "build_cmt_rules",
    "build_signal_rules",
    # Errors
    "BusinessRuleValidationError",
    "CritiqueValidationError",
    "EvidenceValidationError",
    "RemediationExhaustedError",
    "SchemaValidationError",
    "ValidationError",
]
