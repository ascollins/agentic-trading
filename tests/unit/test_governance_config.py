"""Tests for governance configuration."""

from agentic_trading.core.config import (
    CanaryConfig,
    DriftDetectorConfig,
    ExecutionTokenConfig,
    GovernanceConfig,
    HealthScoreConfig,
    ImpactClassifierConfig,
    MaturityConfig,
    Settings,
)


class TestGovernanceConfig:
    """GovernanceConfig defaults and nesting."""

    def test_governance_disabled_by_default(self):
        cfg = GovernanceConfig()
        assert cfg.enabled is False

    def test_maturity_defaults(self):
        cfg = MaturityConfig()
        assert cfg.default_level == "L1_paper"
        assert cfg.promotion_min_trades == 50
        assert cfg.l3_sizing_cap == 0.25

    def test_health_score_defaults(self):
        cfg = HealthScoreConfig()
        assert cfg.window_size == 50
        assert cfg.debt_per_loss == 1.0
        assert cfg.min_sizing_multiplier == 0.1

    def test_canary_defaults(self):
        cfg = CanaryConfig()
        assert cfg.check_interval_seconds == 30
        assert "event_bus" in cfg.components
        assert cfg.failure_threshold == 3

    def test_impact_defaults(self):
        cfg = ImpactClassifierConfig()
        assert cfg.high_notional_usd == 50_000.0
        assert cfg.critical_notional_usd == 200_000.0

    def test_drift_defaults(self):
        cfg = DriftDetectorConfig()
        assert cfg.deviation_threshold_pct == 30.0
        assert cfg.pause_threshold_pct == 50.0
        assert "win_rate" in cfg.metrics_tracked

    def test_token_defaults(self):
        cfg = ExecutionTokenConfig()
        assert cfg.default_ttl_seconds == 300
        assert cfg.require_tokens is False


class TestGovernanceInSettings:
    """GovernanceConfig in Settings."""

    def test_settings_has_governance(self):
        settings = Settings()
        assert hasattr(settings, "governance")
        assert isinstance(settings.governance, GovernanceConfig)

    def test_settings_governance_disabled(self):
        settings = Settings()
        assert settings.governance.enabled is False

    def test_settings_with_governance_override(self):
        settings = Settings(governance=GovernanceConfig(enabled=True))
        assert settings.governance.enabled is True

    def test_nested_config_override(self):
        cfg = GovernanceConfig(
            enabled=True,
            maturity=MaturityConfig(promotion_min_trades=100),
        )
        assert cfg.maturity.promotion_min_trades == 100
