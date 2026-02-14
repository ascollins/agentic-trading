"""Tests for governance.health_score — Strategy Health Tracker."""

import pytest

from agentic_trading.core.config import HealthScoreConfig
from agentic_trading.governance.health_score import HealthTracker


class TestHealthTracker:
    """Basic health tracking operations."""

    def test_initial_score_pristine(self, health_tracker):
        """New strategies should have a perfect health score."""
        assert health_tracker.get_score("s1") == 1.0

    def test_initial_sizing_multiplier_full(self, health_tracker):
        assert health_tracker.get_sizing_multiplier("s1") == 1.0

    def test_loss_accumulates_debt(self, health_tracker):
        """A loss should increase debt and reduce score."""
        health_tracker.record_outcome("s1", won=False, r_multiple=-1.0)
        assert health_tracker.get_debt("s1") > 0
        assert health_tracker.get_score("s1") < 1.0

    def test_win_adds_credit(self, health_tracker):
        """A win should add credit."""
        health_tracker.record_outcome("s1", won=True, r_multiple=2.0)
        assert health_tracker.get_credit("s1") > 0

    def test_win_reduces_debt_first(self, health_tracker):
        """Credits should clear debt before accumulating."""
        health_tracker.record_outcome("s1", won=False, r_multiple=-1.0)
        initial_debt = health_tracker.get_debt("s1")
        health_tracker.record_outcome("s1", won=True, r_multiple=1.0)
        assert health_tracker.get_debt("s1") < initial_debt

    def test_multiple_losses_degrade_score(self, health_tracker):
        """Multiple losses should progressively degrade the score."""
        scores = []
        for _ in range(5):
            health_tracker.record_outcome("s1", won=False, r_multiple=-1.0)
            scores.append(health_tracker.get_score("s1"))
        # Each subsequent score should be lower
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]


class TestDebtCreditModel:
    """Debt/credit accumulation details."""

    def test_debt_capped_at_max(self, health_tracker):
        """Debt should not exceed max_debt."""
        for _ in range(20):
            health_tracker.record_outcome("s1", won=False, r_multiple=-2.0)
        assert health_tracker.get_debt("s1") <= health_tracker._config.max_debt

    def test_score_floors_at_zero(self, health_tracker):
        """Score should never go below 0."""
        for _ in range(20):
            health_tracker.record_outcome("s1", won=False, r_multiple=-2.0)
        assert health_tracker.get_score("s1") >= 0.0

    def test_loss_magnitude_scales_debt(self):
        """Larger losses should accumulate more debt."""
        cfg = HealthScoreConfig(debt_per_loss=1.0, max_debt=100.0)
        tracker = HealthTracker(cfg)
        tracker.record_outcome("s1", won=False, r_multiple=-3.0)
        debt_big = tracker.get_debt("s1")

        tracker2 = HealthTracker(cfg)
        tracker2.record_outcome("s1", won=False, r_multiple=-1.0)
        debt_small = tracker2.get_debt("s1")

        assert debt_big > debt_small

    def test_sizing_multiplier_has_floor(self):
        """Sizing should never go below min_sizing_multiplier."""
        cfg = HealthScoreConfig(
            debt_per_loss=5.0,
            max_debt=10.0,
            min_sizing_multiplier=0.2,
        )
        tracker = HealthTracker(cfg)
        for _ in range(10):
            tracker.record_outcome("s1", won=False, r_multiple=-2.0)
        assert tracker.get_sizing_multiplier("s1") >= 0.2

    def test_recovery_after_losses(self, health_tracker):
        """Wins after losses should gradually restore the score."""
        for _ in range(5):
            health_tracker.record_outcome("s1", won=False, r_multiple=-1.0)
        low_score = health_tracker.get_score("s1")

        for _ in range(10):
            health_tracker.record_outcome("s1", won=True, r_multiple=1.0)
        high_score = health_tracker.get_score("s1")

        assert high_score > low_score


class TestHealthReset:
    """Admin reset functionality."""

    def test_reset_restores_pristine(self, health_tracker):
        for _ in range(5):
            health_tracker.record_outcome("s1", won=False, r_multiple=-1.0)
        assert health_tracker.get_score("s1") < 1.0
        health_tracker.reset("s1")
        assert health_tracker.get_score("s1") == 1.0
        assert health_tracker.get_debt("s1") == 0.0

    def test_window_trades_count(self, health_tracker):
        for i in range(3):
            health_tracker.record_outcome("s1", won=True, r_multiple=1.0)
        assert health_tracker.get_window_trades("s1") == 3

    def test_health_update_event_returned(self, health_tracker):
        event = health_tracker.record_outcome("s1", won=True, r_multiple=1.0)
        assert event.strategy_id == "s1"
        assert event.score > 0
        assert event.window_trades == 1
