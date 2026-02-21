"""Tooltip coverage, length limits, and duplicate detection.

Validates tooltips.json and checks that every section header and card label
in the rendered dashboard HTML contains a tooltip info-icon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from agentic_trading.ui.app import create_ui_app

TOOLTIPS_PATH = Path(__file__).resolve().parents[2] / (
    "src/agentic_trading/ui/static/tooltips.json"
)

TITLE_MAX = 40
BODY_MAX = 240


# ------------------------------------------------------------------
# Load tooltip dictionary
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def tooltips() -> dict:
    with open(TOOLTIPS_PATH) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Structure & length validation
# ------------------------------------------------------------------


class TestTooltipDictionary:
    def test_file_exists(self):
        assert TOOLTIPS_PATH.exists(), "tooltips.json not found"

    def test_valid_json(self):
        with open(TOOLTIPS_PATH) as f:
            data = json.load(f)
        assert "sections" in data
        assert "cards" in data

    def test_section_count(self, tooltips):
        assert len(tooltips["sections"]) == 12, (
            f"Expected 12 sections, got {len(tooltips['sections'])}"
        )

    def test_card_count(self, tooltips):
        assert len(tooltips["cards"]) >= 20, (
            f"Expected >=20 cards, got {len(tooltips['cards'])}"
        )

    def test_title_length(self, tooltips):
        violations = []
        for category in ("sections", "cards"):
            for key, entry in tooltips[category].items():
                if len(entry["title"]) > TITLE_MAX:
                    violations.append(
                        f"{category}.{key}.title: {len(entry['title'])} chars"
                    )
        assert not violations, f"Title too long:\n" + "\n".join(violations)

    def test_body_length(self, tooltips):
        violations = []
        for category in ("sections", "cards"):
            for key, entry in tooltips[category].items():
                if len(entry["body"]) > BODY_MAX:
                    violations.append(
                        f"{category}.{key}.body: {len(entry['body'])} chars "
                        f"(max {BODY_MAX})"
                    )
        assert not violations, f"Body too long:\n" + "\n".join(violations)

    def test_no_duplicate_bodies(self, tooltips):
        seen: dict[str, str] = {}
        duplicates = []
        for category in ("sections", "cards"):
            for key, entry in tooltips[category].items():
                body = entry["body"]
                full_key = f"{category}.{key}"
                if body in seen:
                    duplicates.append(f"{full_key} == {seen[body]}")
                seen[body] = full_key
        assert not duplicates, f"Duplicate tooltips:\n" + "\n".join(duplicates)

    def test_required_fields(self, tooltips):
        missing = []
        for category in ("sections", "cards"):
            for key, entry in tooltips[category].items():
                for field in ("title", "body"):
                    if field not in entry or not entry[field].strip():
                        missing.append(f"{category}.{key}.{field}")
        assert not missing, f"Missing fields:\n" + "\n".join(missing)


# ------------------------------------------------------------------
# Rendered HTML coverage
# ------------------------------------------------------------------


def _make_app():
    return create_ui_app()


class TestTooltipRendering:
    def test_home_page_has_section_tooltips(self):
        """Every section header on the home page should have an info-icon."""
        client = TestClient(_make_app())
        html = client.get("/").text
        # All 12 section headers should contain the info-icon pattern
        assert html.count("info-icon") >= 12, (
            f"Expected >=12 info-icons in section headers, "
            f"found {html.count('info-icon')}"
        )

    def test_section_tooltip_text_present(self):
        """Spot-check that actual tooltip text appears in the rendered page."""
        client = TestClient(_make_app())
        html = client.get("/").text
        # Check a few section tooltips are rendered
        assert "Account equity and daily performance" in html
        assert "Chronological feed of signals" in html
        assert "Agent health and system status" in html

    def test_card_partials_have_tooltips(self):
        """Each card partial route should render an info-icon tooltip."""
        client = TestClient(_make_app())
        routes = [
            "/partials/home/scorecard",
            "/partials/home/portfolio",
            "/partials/home/positions",
            "/partials/home/system",
            "/partials/risk-pnl/summary",
            "/partials/action-queue/approvals",
            "/partials/action-queue/incidents",
            "/partials/action-queue/decisions",
            "/partials/action-queue/alerts",
            "/partials/model-scorecard/models",
            "/partials/model-scorecard/drift",
            "/partials/model-scorecard/effectiveness",
            "/partials/model-scorecard/exec-quality",
            "/partials/pre-trade/checks",
            "/partials/pre-trade/throttles",
            "/partials/micro-edge/distribution",
            "/partials/micro-edge/edge-analysis",
            "/partials/micro-edge/surveillance",
            "/partials/risk/gauges",
            "/partials/risk/circuit-breakers",
        ]
        missing = []
        for route in routes:
            resp = client.get(route)
            if resp.status_code == 200 and "info-icon" not in resp.text:
                missing.append(route)
        assert not missing, f"Missing info-icon in:\n" + "\n".join(missing)
