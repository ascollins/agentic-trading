"""Shared fixtures for governance tests."""

import pytest

from agentic_trading.core.config import (
    CanaryConfig,
    DriftDetectorConfig,
    ExecutionTokenConfig,
    GovernanceConfig,
    HealthScoreConfig,
    ImpactClassifierConfig,
    MaturityConfig,
)
from agentic_trading.governance.canary import GovernanceCanary
from agentic_trading.governance.drift_detector import DriftDetector
from agentic_trading.governance.gate import GovernanceGate
from agentic_trading.governance.health_score import HealthTracker
from agentic_trading.governance.impact_classifier import ImpactClassifier
from agentic_trading.governance.maturity import MaturityManager
from agentic_trading.governance.tokens import TokenManager


@pytest.fixture
def maturity_config():
    return MaturityConfig()


@pytest.fixture
def health_config():
    return HealthScoreConfig()


@pytest.fixture
def canary_config():
    return CanaryConfig()


@pytest.fixture
def impact_config():
    return ImpactClassifierConfig()


@pytest.fixture
def drift_config():
    return DriftDetectorConfig()


@pytest.fixture
def token_config():
    return ExecutionTokenConfig()


@pytest.fixture
def governance_config():
    return GovernanceConfig(enabled=True)


@pytest.fixture
def maturity_manager(maturity_config):
    return MaturityManager(maturity_config)


@pytest.fixture
def health_tracker(health_config):
    return HealthTracker(health_config)


@pytest.fixture
def impact_classifier(impact_config):
    return ImpactClassifier(impact_config)


@pytest.fixture
def drift_detector(drift_config):
    return DriftDetector(drift_config)


@pytest.fixture
def token_manager(token_config):
    return TokenManager(token_config)


@pytest.fixture
def governance_gate(
    governance_config,
    maturity_manager,
    health_tracker,
    impact_classifier,
    drift_detector,
    token_manager,
):
    return GovernanceGate(
        config=governance_config,
        maturity=maturity_manager,
        health=health_tracker,
        impact=impact_classifier,
        drift=drift_detector,
        tokens=None,  # Tokens not required by default
        event_bus=None,
    )
