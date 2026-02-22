"""Test UI route status codes, content types, and key HTML elements.

Uses create_ui_app() with all-None parameters (graceful degradation mode).
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from agentic_trading.ui.app import create_ui_app


def _make_app():
    """Create a UI app with no dependencies (graceful degradation)."""
    return create_ui_app()


# ------------------------------------------------------------------
# Full-page routes
# ------------------------------------------------------------------


class TestUIPageRoutes:
    def test_home_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_home_contains_mission_control_heading(self):
        client = TestClient(_make_app())
        resp = client.get("/")
        assert "Mission Control" in resp.text

    def test_strategies_redirects_to_home(self):
        client = TestClient(_make_app(), follow_redirects=False)
        resp = client.get("/strategies")
        assert resp.status_code == 302
        assert "/#section-strategies" in resp.headers["location"]

    def test_activity_redirects_to_home(self):
        client = TestClient(_make_app(), follow_redirects=False)
        resp = client.get("/activity")
        assert resp.status_code == 302
        assert "/#section-activity" in resp.headers["location"]

    def test_risk_redirects_to_home(self):
        client = TestClient(_make_app(), follow_redirects=False)
        resp = client.get("/risk")
        assert resp.status_code == 302
        assert "/#section-risk" in resp.headers["location"]

    def test_settings_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_backtest_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/backtest")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_health_returns_ok(self):
        client = TestClient(_make_app())
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ------------------------------------------------------------------
# HTMX partial routes
# ------------------------------------------------------------------


class TestUIPartialRoutes:
    def test_partial_scorecard(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/home/scorecard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_partial_portfolio(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/home/portfolio")
        assert resp.status_code == 200

    def test_partial_positions(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/home/positions")
        assert resp.status_code == 200

    def test_partial_system(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/home/system")
        assert resp.status_code == 200

    def test_partial_approvals(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/home/approvals")
        assert resp.status_code == 200

    def test_partial_banner(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/home/banner")
        assert resp.status_code == 200

    def test_partial_status_bar(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/status-bar")
        assert resp.status_code == 200

    def test_partial_strategies_list(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/strategies/list")
        assert resp.status_code == 200

    def test_partial_activity_timeline_all(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/activity/timeline?type=all")
        assert resp.status_code == 200

    def test_partial_activity_timeline_trades(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/activity/timeline?type=trades")
        assert resp.status_code == 200

    def test_partial_risk_gauges(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/risk/gauges")
        assert resp.status_code == 200

    def test_partial_kill_switch(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/risk/kill-switch")
        assert resp.status_code == 200

    def test_equity_curve_api(self):
        client = TestClient(_make_app())
        resp = client.get("/api/equity-curve")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ------------------------------------------------------------------
# Action routes (graceful degradation — no backend components)
# ------------------------------------------------------------------


class TestUIActionRoutes:
    def test_approve_without_manager_returns_200(self):
        client = TestClient(_make_app())
        resp = client.post("/actions/approve/fake-id")
        assert resp.status_code == 200

    def test_deny_without_manager_returns_200(self):
        client = TestClient(_make_app())
        resp = client.post("/actions/deny/fake-id")
        assert resp.status_code == 200

    def test_kill_switch_activate_without_manager(self):
        client = TestClient(_make_app())
        resp = client.post("/actions/kill-switch/activate")
        assert resp.status_code == 200

    def test_kill_switch_deactivate_without_manager(self):
        client = TestClient(_make_app())
        resp = client.post("/actions/kill-switch/deactivate")
        assert resp.status_code == 200

    def test_close_position_without_adapter(self):
        client = TestClient(_make_app())
        resp = client.post("/actions/positions/close/BTC%2FUSDT")
        assert resp.status_code == 200
        # Should contain an error message about missing adapter
        assert "No adapter" in resp.text or "positions" in resp.text.lower()

    def test_promote_without_lifecycle(self):
        client = TestClient(_make_app())
        resp = client.post("/actions/promote/trend_following")
        assert resp.status_code == 200

    def test_resolve_incident_without_manager(self):
        client = TestClient(_make_app())
        resp = client.post("/actions/resolve-incident/fake-id")
        assert resp.status_code == 200

    def test_action_routes_return_toast_headers(self):
        """All action routes should include HX-Trigger toast headers."""
        client = TestClient(_make_app())
        # Test a sample of action routes for HX-Trigger header
        resp = client.post("/actions/approve/fake-id")
        assert "HX-Trigger" in resp.headers
        assert "showToast" in resp.headers["HX-Trigger"]

        resp = client.post("/actions/deny/fake-id")
        assert "HX-Trigger" in resp.headers

        resp = client.post("/actions/kill-switch/activate")
        assert "HX-Trigger" in resp.headers

        resp = client.post("/actions/positions/close/BTC%2FUSDT")
        assert "HX-Trigger" in resp.headers


# ------------------------------------------------------------------
# Error pages
# ------------------------------------------------------------------


class TestUIErrorPages:
    def test_404_returns_html(self):
        client = TestClient(_make_app(), raise_server_exceptions=False)
        resp = client.get("/nonexistent-page")
        assert resp.status_code == 404
        assert "text/html" in resp.headers["content-type"]
        assert "Page Not Found" in resp.text

    def test_404_contains_back_link(self):
        client = TestClient(_make_app(), raise_server_exceptions=False)
        resp = client.get("/nonexistent-page")
        assert 'href="/"' in resp.text


# ------------------------------------------------------------------
# Circuit breakers route
# ------------------------------------------------------------------


class TestCircuitBreakersRoute:
    def test_circuit_breakers_partial_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/risk/circuit-breakers")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_circuit_breakers_shows_armed_defaults(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/risk/circuit-breakers")
        assert "Daily Loss" in resp.text


# ------------------------------------------------------------------
# New dashboard section routes (graceful degradation — all None deps)
# ------------------------------------------------------------------


class TestRiskPnlRoutes:
    def test_risk_pnl_summary_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/risk-pnl/summary")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_risk_pnl_summary_shows_default_values(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/risk-pnl/summary")
        # With no deps, builder returns zeroed defaults — template renders data view
        assert "DAILY LOSS BUDGET" in resp.text
        assert "EXPOSURE" in resp.text


class TestActionQueueRoutes:
    def test_action_approvals_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/action-queue/approvals")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_action_incidents_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/action-queue/incidents")
        assert resp.status_code == 200

    def test_action_decisions_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/action-queue/decisions")
        assert resp.status_code == 200

    def test_action_alerts_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/action-queue/alerts")
        assert resp.status_code == 200


class TestModelScorecardRoutes:
    def test_model_registry_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/model-scorecard/models")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_drift_indicators_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/model-scorecard/drift")
        assert resp.status_code == 200

    def test_effectiveness_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/model-scorecard/effectiveness")
        assert resp.status_code == 200

    def test_exec_quality_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/model-scorecard/exec-quality")
        assert resp.status_code == 200


class TestPreTradeControlsRoutes:
    def test_pre_trade_checks_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/pre-trade/checks")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_pre_trade_throttles_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/pre-trade/throttles")
        assert resp.status_code == 200


class TestMicroEdgeRoutes:
    def test_r_distribution_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/micro-edge/distribution")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_edge_analysis_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/micro-edge/edge-analysis")
        assert resp.status_code == 200

    def test_surveillance_returns_200(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/micro-edge/surveillance")
        assert resp.status_code == 200

    def test_r_distribution_api_returns_json(self):
        client = TestClient(_make_app())
        resp = client.get("/api/r-distribution")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_surveillance_shows_empty_state(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/micro-edge/surveillance")
        assert "NO OPEN CASES" in resp.text

    def test_edge_analysis_shows_dash_placeholders(self):
        client = TestClient(_make_app())
        resp = client.get("/partials/micro-edge/edge-analysis")
        # With no journal, edge lists are empty — template shows "--" dashes
        assert "EDGE ANALYSIS" in resp.text
        assert "BY STRATEGY" in resp.text
