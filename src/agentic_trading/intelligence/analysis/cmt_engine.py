"""CMT Analysis Engine — Claude API integration for qualitative analysis.

Constructs prompts from platform-computed features, calls the Anthropic
API with the CMT skill as system prompt, and parses structured JSON
responses into validated Pydantic models.

Local computation (indicators, risk math) is handled by the platform.
The Claude API is used for judgment calls: pattern recognition, Wyckoff
phase classification, confluence synthesis, behavioral bias assessment,
trade thesis construction, and system health interpretation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .cmt_models import (
    CMTAssessmentRequest,
    CMTAssessmentResponse,
    CMTConfluenceScore,
    CMTLayerResult,
    CMTTradePlan,
    CMTTarget,
)

logger = logging.getLogger(__name__)

# JSON schema instruction appended to every user prompt.
_RESPONSE_SCHEMA = """\
Respond with ONLY valid JSON matching this schema (no markdown fences):
{
  "symbol": "<string>",
  "timeframes_analyzed": ["<string>", ...],
  "layers": [
    {
      "layer": <1-9>,
      "name": "<layer name>",
      "direction": "bullish|bearish|neutral",
      "confidence": "high|medium|low",
      "score": <float>,
      "key_findings": ["<string>", ...],
      "warnings": ["<string>", ...]
    }
  ],
  "confluence": {
    "trend_alignment": <-2 to 2>,
    "key_level_proximity": <0 to 2>,
    "pattern_signal": <-2 to 2>,
    "indicator_consensus": <-2 to 2>,
    "sentiment_alignment": <-1 to 1>,
    "volatility_regime": <-1 to 1>,
    "macro_alignment": <-1 to 1>
  },
  "trade_plan": null | {
    "direction": "LONG|SHORT",
    "entry_price": <float>,
    "entry_trigger": "<string>",
    "stop_loss": <float>,
    "stop_reasoning": "<string>",
    "targets": [{"price": <float>, "pct": <float>, "source": "<string>"}],
    "rr_ratio": <float>,
    "blended_rr": <float>,
    "position_size_pct": <float>,
    "invalidation": "<string>",
    "thesis": "<string>"
  },
  "thesis": "<string>",
  "system_health": "green|amber|red",
  "watchlist_action": "enter|monitor|no_action",
  "no_trade_reason": "<string or empty>"
}"""


class CMTAnalysisEngine:
    """Orchestrates the Claude API-powered CMT analysis pipeline.

    Parameters
    ----------
    skill_path:
        Directory containing SKILL.md and references/.
    api_key_env:
        Name of the environment variable holding the Anthropic API key.
    model:
        Claude model identifier.
    max_daily_calls:
        Maximum API calls per calendar day (budget guard).
    min_confluence:
        Minimum confluence total to include a trade plan.
    """

    def __init__(
        self,
        *,
        skill_path: str = "skills/cmt-analyst",
        api_key_env: str = "ANTHROPIC_API_KEY",
        model: str = "claude-sonnet-4-5-20250929",
        max_daily_calls: int = 50,
        min_confluence: int = 5,
    ) -> None:
        self._model = model
        self._api_key_env = api_key_env
        self._max_daily_calls = max_daily_calls
        self._min_confluence = min_confluence

        # Load skill as system prompt
        self._system_prompt = self._load_skill(skill_path)

        # Daily call budget
        self._calls_today = 0
        self._budget_reset_day = 0  # Day-of-year of last reset

        # Per-symbol rate limiting
        self._last_call_per_symbol: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Skill loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_skill(skill_path: str) -> str:
        """Load SKILL.md content as the system prompt."""
        base = Path(skill_path)
        skill_file = base / "SKILL.md"
        if not skill_file.exists():
            logger.warning("SKILL.md not found at %s, using empty prompt", skill_file)
            return ""
        return skill_file.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------

    def _check_budget(self) -> bool:
        """Return True if we have remaining API call budget for today."""
        today = time.gmtime().tm_yday
        if today != self._budget_reset_day:
            self._calls_today = 0
            self._budget_reset_day = today
        return self._calls_today < self._max_daily_calls

    def _record_call(self) -> None:
        """Increment the daily call counter."""
        self._calls_today += 1

    @property
    def calls_remaining_today(self) -> int:
        today = time.gmtime().tm_yday
        if today != self._budget_reset_day:
            return self._max_daily_calls
        return max(0, self._max_daily_calls - self._calls_today)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, request: CMTAssessmentRequest) -> str:
        """Construct the user prompt from platform data.

        Serialises indicator values, HTF assessment, SMC confluence,
        regime state, and portfolio state into a compact JSON prompt
        that the LLM can process.
        """
        sections: list[str] = [
            f"## Analysis Request: {request.symbol}",
            f"Timeframes: {', '.join(request.timeframes)}",
        ]

        if request.ohlcv_summary:
            sections.append(
                "### OHLCV Summary\n```json\n"
                + json.dumps(request.ohlcv_summary, default=str)
                + "\n```"
            )

        if request.indicator_values:
            sections.append(
                "### Pre-Computed Indicator Values\n```json\n"
                + json.dumps(
                    {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in request.indicator_values.items()},
                    default=str,
                )
                + "\n```"
            )

        if request.htf_assessment:
            sections.append(
                "### Higher-Timeframe Assessment\n```json\n"
                + json.dumps(request.htf_assessment, default=str)
                + "\n```"
            )

        if request.smc_confluence:
            sections.append(
                "### SMC Confluence Analysis\n```json\n"
                + json.dumps(request.smc_confluence, default=str)
                + "\n```"
            )

        if request.regime_state:
            sections.append(
                "### Market Regime\n```json\n"
                + json.dumps(request.regime_state, default=str)
                + "\n```"
            )

        if request.portfolio_state:
            sections.append(
                "### Portfolio State\n```json\n"
                + json.dumps(request.portfolio_state, default=str)
                + "\n```"
            )

        if request.performance_metrics:
            sections.append(
                "### System Performance Metrics\n```json\n"
                + json.dumps(request.performance_metrics, default=str)
                + "\n```"
            )

        sections.append(
            "### Instructions\n"
            "Apply the full 9-layer CMT analytical framework to this data.\n"
            f"Minimum confluence score for trade plan: {self._min_confluence}.\n"
            "If confluence < threshold, set trade_plan to null and explain "
            "in no_trade_reason.\n\n"
            + _RESPONSE_SCHEMA
        )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    async def call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call the Anthropic API and return the raw text response.

        Uses ``anthropic.AsyncAnthropic`` if available, falling back
        to ``httpx`` direct HTTP call.

        Raises
        ------
        RuntimeError
            If the API key is not set or the call fails.
        """
        api_key = os.environ.get(self._api_key_env, "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — cannot call CMT analysis engine"
            )

        try:
            import anthropic  # type: ignore[import-untyped]

            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text  # type: ignore[union-attr]
        except ImportError:
            logger.warning(
                "anthropic SDK not installed, falling back to httpx"
            )
        except Exception:
            logger.exception("Anthropic SDK call failed, trying httpx fallback")

        # httpx fallback
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(self, raw: str) -> CMTAssessmentResponse:
        """Parse a raw JSON string into a validated CMTAssessmentResponse.

        Handles minor formatting issues (markdown fences, trailing
        commas) gracefully.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [
                ln for ln in lines
                if not ln.strip().startswith("```")
            ]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse CMT response JSON: %s", exc)
            return CMTAssessmentResponse(
                symbol="unknown",
                thesis="Parse error — raw response could not be decoded",
                system_health="amber",
            )

        # Build layer results
        layers: list[CMTLayerResult] = []
        for layer_data in data.get("layers", []):
            layers.append(CMTLayerResult(**layer_data))

        # Build confluence score
        conf_data = data.get("confluence", {})
        confluence = CMTConfluenceScore(**conf_data)
        confluence.compute_total()
        confluence.check_veto()
        confluence.threshold_met = (
            confluence.total >= self._min_confluence and not confluence.veto
        )

        # Build trade plan if present
        trade_plan: CMTTradePlan | None = None
        tp_data = data.get("trade_plan")
        if tp_data is not None:
            targets = [CMTTarget(**t) for t in tp_data.pop("targets", [])]
            trade_plan = CMTTradePlan(**tp_data, targets=targets)

        return CMTAssessmentResponse(
            symbol=data.get("symbol", "unknown"),
            timeframes_analyzed=data.get("timeframes_analyzed", []),
            layers=layers,
            confluence=confluence,
            trade_plan=trade_plan,
            thesis=data.get("thesis", ""),
            system_health=data.get("system_health", "green"),
            watchlist_action=data.get("watchlist_action", "no_action"),
            no_trade_reason=data.get("no_trade_reason", ""),
        )

    # ------------------------------------------------------------------
    # Full assessment pipeline
    # ------------------------------------------------------------------

    async def assess(
        self,
        request: CMTAssessmentRequest,
    ) -> CMTAssessmentResponse | None:
        """Run the full CMT analysis pipeline for a symbol.

        Returns ``None`` if the API budget is exhausted or the call
        fails (graceful degradation — the agent continues without
        crashing).
        """
        if not self._check_budget():
            logger.warning(
                "CMT API budget exhausted (%d/%d calls today)",
                self._calls_today,
                self._max_daily_calls,
            )
            return None

        user_prompt = self.build_prompt(request)

        try:
            raw = await self.call_api(self._system_prompt, user_prompt)
        except Exception:
            logger.exception("CMT API call failed for %s", request.symbol)
            return None

        self._record_call()
        self._last_call_per_symbol[request.symbol] = time.monotonic()

        response = self.parse_response(raw)
        response.symbol = request.symbol

        logger.info(
            "CMT assessment for %s: confluence=%.1f, threshold_met=%s, "
            "health=%s, calls_remaining=%d",
            request.symbol,
            response.confluence.total,
            response.confluence.threshold_met,
            response.system_health,
            self.calls_remaining_today,
        )

        return response

    async def assess_with_thinking(
        self,
        request: CMTAssessmentRequest,
    ) -> tuple[CMTAssessmentResponse | None, str]:
        """Run CMT analysis and capture extended thinking.

        Returns ``(response, raw_thinking_text)``.  Falls back to
        standard ``assess()`` if extended thinking is unavailable.
        """
        if not self._check_budget():
            logger.warning(
                "CMT API budget exhausted (%d/%d calls today)",
                self._calls_today,
                self._max_daily_calls,
            )
            return None, ""

        user_prompt = self.build_prompt(request)

        try:
            raw, thinking = await self._call_api_with_thinking(
                self._system_prompt, user_prompt
            )
        except Exception:
            logger.exception(
                "CMT API call with thinking failed for %s, "
                "falling back to standard call",
                request.symbol,
            )
            # Fallback to standard assess
            response = await self.assess(request)
            return response, ""

        self._record_call()
        self._last_call_per_symbol[request.symbol] = time.monotonic()

        response = self.parse_response(raw)
        response.symbol = request.symbol

        logger.info(
            "CMT assessment (with thinking) for %s: confluence=%.1f, "
            "threshold_met=%s, health=%s, thinking_len=%d",
            request.symbol,
            response.confluence.total,
            response.confluence.threshold_met,
            response.system_health,
            len(thinking),
        )

        return response, thinking

    async def _call_api_with_thinking(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, str]:
        """Call Anthropic API with extended thinking enabled.

        Returns ``(response_text, thinking_text)``.
        Falls back to standard ``call_api`` if extended thinking
        is not supported.
        """
        api_key = os.environ.get(self._api_key_env, "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — cannot call CMT analysis engine"
            )

        try:
            import anthropic  # type: ignore[import-untyped]

            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model=self._model,
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 8000,
                },
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            thinking_text = ""
            response_text = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    response_text = block.text

            return response_text, thinking_text
        except Exception:
            logger.warning(
                "Extended thinking call failed, falling back to standard API"
            )
            raw = await self.call_api(system_prompt, user_prompt)
            return raw, ""
