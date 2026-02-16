# Agentic Trading Dashboard — UX Improvement Plan

---

## Relevant Files Found

| File | What it contains |
|---|---|
| `configs/grafana/dashboards/trading.json` | Main trading dashboard (30 panels): portfolio, strategies, governance, latency, trade journal analytics |
| `configs/grafana/dashboards/narration.json` | Live narration dashboard (4 panels): real-time plain-English commentary, avatar button, status, disclaimer |
| `src/agentic_trading/observability/metrics.py` | 50+ metrics definitions across 8 categories — the data feeding both dashboards |
| `configs/prometheus.yml` | Data collection configuration (scrapes every 5 seconds) |
| `configs/grafana/provisioning/` | Datasource and dashboard auto-setup files |
| `docker-compose.yml` | Infrastructure: defines Grafana on port 3001, Prometheus on 9091 |

No screenshots or incident runbooks were found in the repository.

---

## Dashboard Understanding (In Human Terms)

### What it helps people decide
The main dashboard answers: "Is our automated trading system healthy, making money, and behaving safely?" It's a single scrolling page that mixes everything together — portfolio performance, strategy documentation, system speed, safety guardrails, and deep statistical analysis.

### Who it seems designed for
An engineer who built the system. The language, layout, and information density all point to someone who already knows what "p95 latency," "Brier score," "R-multiple," and "drift deviation" mean. Non-technical stakeholders (Product, Risk, Leadership) would struggle to use this dashboard without help.

### The "first question" it answers
"How much money do we have, and are we making or losing today?" — Portfolio Equity and Daily PnL are in the top-left corner.

### The "last question" it answers
"Are our strategies statistically sound over time?" — buried 98 rows down with Kelly Fraction, Probability of Ruin, and Statistical Edge panels.

### What's unclear or overwhelming at first glance
1. **The page is a single endless scroll** — roughly 98 rows of content with no navigation or visual hierarchy. A new viewer has to scroll through everything to find what they care about.
2. **Strategy documentation lives inside the monitoring dashboard** — four large text blocks (Trend Following, Mean Reversion, Breakout, Funding Arbitrage) take up 20 rows of screen space. These are reference material, not real-time monitoring.
3. **Panel titles use engineering jargon** — "Drift Deviation %," "Canary Status," "Active Execution Tokens," "Confidence Calibration (Brier Score)," "Kelly Fraction," "p95/p99" all require domain expertise.
4. **No status summary or "traffic light" view** — you have to mentally combine 6+ separate panels to determine "is everything OK?"
5. **No contextual help** — panels don't explain what a "good" or "bad" value looks like, even though the thresholds are configured in the JSON.

---

# 1) Executive Summary

## What works today
- **Real data is being collected.** 50+ metrics are instrumented across trading, governance, risk, and analytics. The plumbing is solid.
- **Good threshold thinking.** Drawdown, win rate, profit factor, and several other panels already have color-coded thresholds (green/yellow/red). The building blocks of a traffic-light system exist.
- **Narration dashboard is excellent.** The "Live Narration" dashboard is a genuinely user-friendly innovation — plain-English commentary on system decisions, color-coded actions, and an avatar briefing. This is the best part of the current setup and should be the model for the rest.
- **Kill Switch visibility.** The emergency stop indicator (green OFF / red ACTIVE) is prominently placed and clearly labeled. This is exactly the right pattern.

## Top 3 user pain points

1. **"I can't find what I need."** Everything is on one 98-row scrolling page with no navigation, no sections you can jump to, and no way to filter by role. A Risk Analyst looking for drawdown data has to scroll past strategy documentation and signal charts. A Leadership viewer looking for "are we profitable?" has to scan past governance internals.

2. **"I don't understand what these numbers mean."** Panel titles like "Drift Deviation %," "Confidence Calibration (Brier Score)," "Avg R-Multiple," "Active Execution Tokens," and "Probability of Ruin (MC)" require specialized knowledge. There are no descriptions, tooltips, or benchmarks telling viewers what "good" looks like.

3. **"I can't tell if things are OK without studying every panel."** There's no summary health indicator. To answer "is the system healthy right now?" you'd need to check Kill Switch + Drawdown + Canary Status + Health Scores + Overtrading Alert + Latency panels individually and mentally combine the results.

## Top 3 improvements with expected impact

| # | Improvement | Expected impact |
|---|---|---|
| 1 | **Add a "System Health" summary row at the top** — a single row with 4-5 status indicators (Overall Health, Portfolio Status, Risk Status, System Speed, Data Freshness) that turn green/yellow/red automatically. | Reduces "time to understand system status" from 2-3 minutes of scrolling to 5 seconds. Everyone from Leadership to On-Call can glance and know. |
| 2 | **Split into role-based views** — separate the single dashboard into 3 linked dashboards: (a) Overview for everyone, (b) Strategy Performance for Trading Ops, (c) Risk & Governance for Risk/Compliance. Move strategy documentation to a separate reference page. | Each persona sees only what they need. Page load is faster. Cognitive overload drops dramatically. |
| 3 | **Rename every panel in plain English and add "what good looks like"** — replace jargon titles with question-based titles (e.g., "Confidence Calibration (Brier Score)" → "How accurate are strategy predictions?") and add target ranges in panel descriptions. | Non-technical stakeholders can use the dashboard independently. Fewer "what does this mean?" questions to engineers. |

---

# 2) Users & Jobs-to-be-Done

## Persona 1: Trading Ops Lead
**Who they are:** Day-to-day manager of the trading platform. Checks the system multiple times per day. Wants to know if strategies are performing and if anything needs adjustment.

**Goals:**
- Confirm strategies are generating signals and executing trades
- Spot underperforming strategies quickly
- Decide whether to pause, adjust, or promote a strategy

**What they need to see first:**
- Today's profit/loss and equity curve
- Which strategies are active and their win rates
- Any strategies that have been blocked or paused by governance

**What success looks like:**
- Can answer "how are we doing today?" in under 10 seconds
- Can identify which strategy to investigate in under 30 seconds
- Doesn't need to ask an engineer to interpret the dashboard

## Persona 2: Risk Analyst
**Who they are:** Responsible for monitoring risk limits, drawdown, and ensuring the system stays within safety boundaries. May not be deeply technical but understands financial risk concepts.

**Goals:**
- Verify the system is within risk limits (drawdown, exposure, position sizes)
- Get alerted when safety mechanisms activate (kill switch, circuit breakers, governance blocks)
- Review strategy behavior drift — is a strategy behaving differently than expected?

**What they need to see first:**
- Kill switch status and drawdown gauge
- Governance blocks and their reasons
- Risk of ruin and maximum drawdown per strategy

**What success looks like:**
- Can confirm "all risk limits are within bounds" in under 15 seconds
- Can immediately see if any safety mechanism has been triggered
- Can identify which strategy is drifting from its expected behavior

## Persona 3: On-Call Responder
**Who they are:** Engineer or operator on rotation who needs to respond to alerts quickly. Needs to diagnose problems, not review strategy performance.

**Goals:**
- Quickly determine if the system is operational
- Identify the source of a problem (data feed issue? slow exchange? governance bottleneck?)
- Decide whether to escalate or take immediate action (kill switch)

**What they need to see first:**
- System health summary (all green = go back to sleep)
- Data freshness — is the market data feed still working?
- System speed — are decisions and orders being processed quickly?
- Canary status — are all background components healthy?

**What success looks like:**
- Can determine "is this a real problem?" in under 10 seconds
- Can identify the failing component in under 30 seconds
- Has a clear path to escalation or action

## Persona 4: Executive / Leadership Viewer
**Who they are:** Senior stakeholder who checks in occasionally (daily or weekly). Wants a high-level picture without details. Asks "are we on track?" and "should I be worried?"

**Goals:**
- See overall profitability at a glance
- Know if the system is operating safely
- Understand trends over time (are we improving?)

**What they need to see first:**
- Portfolio equity trend (are we growing?)
- Daily P&L (did we make money today?)
- One-line system status (healthy / warning / critical)

**What success looks like:**
- Can answer "how is the trading platform doing?" in a 30-second glance
- Doesn't encounter any jargon they can't understand
- Feels confident the system has proper safety controls

---

# 3) Dashboard "Story" (Information Flow)

The ideal dashboard should guide every viewer through four stages, from most urgent to most detailed:

## Stage 1: "Are we OK?" (0-5 seconds)
**What the viewer sees:** A single summary row at the very top with 4-5 large status indicators.

| Indicator | Green | Yellow | Red |
|---|---|---|---|
| **System Status** | All systems running | Degraded performance | Kill switch active or component down |
| **Portfolio Health** | Positive daily P&L, drawdown < 5% | Drawdown 5-10% or small daily loss | Drawdown > 10% or large daily loss |
| **Risk Guardrails** | All within limits | Approaching limits | Limits breached, trades blocked |
| **System Speed** | Decisions under 100ms | Decisions 100-500ms | Decisions over 500ms |
| **Data Feed** | All market data fresh | Minor delays | Stale data (>60 seconds old) |

**If everything is green,** the Executive and On-Call Responder can stop here. If something is yellow or red, they scroll down.

## Stage 2: "What changed?" (5-30 seconds)
**What the viewer sees:** Key numbers and trends that highlight recent changes.

- Portfolio equity curve (last hour, overlaid with today's P&L)
- Active positions count and gross exposure
- Recent governance blocks or alerts (anything that was stopped or flagged)
- Any strategy that changed status (started, paused, overtrading)

**This answers:** "Something is yellow — what specifically happened?"

## Stage 3: "Where is the issue?" (30 seconds - 2 minutes)
**What the viewer sees:** Detailed breakdowns organized by category. This is where the current dashboard's good data lives, but reorganized:

- **Strategy Performance Tab:** Win rates, profit factors, expectancy per strategy. Comparison table, not individual gauges.
- **Risk Deep-Dive Tab:** Drawdown per strategy, risk-of-ruin, Kelly fraction, governance decisions and blocks over time.
- **System Internals Tab:** Decision speed, order speed, governance speed, data staleness per symbol, canary component status.

**This answers:** "The problem is in Strategy X" or "The exchange connection is slow."

## Stage 4: "What should we do next?" (Action guidance)
**What the viewer sees:** Contextual guidance near problem areas.

- Next to a high drawdown: "Consider reducing position size or pausing strategy."
- Next to governance blocks: "Review blocked trade rationale. Escalate if unexpected."
- Next to slow latency: "Check exchange API status. Consider failover."
- Next to overtrading alert: "Strategy is trading more than expected. Review signal frequency."

**This answers:** "Now I know the problem — what's my next step?"

---

# 4) Usability Critique (Panel by Panel)

## Row 0-8: Portfolio Overview

### "Portfolio Equity" (timeseries, 12-wide)
- **Question it answers:** How is our total account value changing over time?
- **What's confusing:** Good title. But placed next to "Kill Switch" which creates a visual priority mismatch — equity is informational while Kill Switch is an emergency alert.
- **What's missing:** No benchmark line (starting equity), no annotations for when trades happened. No percentage-change view option.
- **Rename suggestion:** Keep "Portfolio Equity" — add subtitle "Total account value over time"

### "Daily PnL" (stat, 6-wide)
- **Question it answers:** How much did we make or lose today?
- **What's confusing:** Fine as-is. Good use of currency formatting.
- **What's missing:** No context — is $500 good or bad? A percentage of equity would add meaning.
- **Rename suggestion:** "Today's Profit / Loss" — add "as of [time]" context

### "Kill Switch" (stat, 6-wide)
- **Question it answers:** Has the emergency stop been activated?
- **What's confusing:** Nothing — this is well-designed with clear green/red color coding.
- **What's missing:** Should be more prominent. If this is active, it's the most important thing on the dashboard.
- **Rename suggestion:** "Emergency Stop" — more universally understood than "Kill Switch"

### "Gross Exposure" (stat, 6-wide)
- **Question it answers:** How much total money is at risk right now?
- **What's confusing:** The word "exposure" is financial jargon. Non-financial users won't know what this means.
- **What's missing:** No context vs. limits. Is $50,000 exposure a lot for this account?
- **Rename suggestion:** "Total Capital at Risk" — add "Limit: $X" as reference

### "System Mode" (stat, 6-wide)
- **Question it answers:** Is the system running, and in what mode?
- **What's confusing:** Shows "RUNNING" in green but the underlying metric ("trading_system_info") only maps the value "1" to "RUNNING." Other states aren't defined, so this could show raw numbers if something breaks.
- **What's missing:** No visual for "paper" vs "live" mode distinction, which is critical context.
- **Rename suggestion:** "Trading Mode" — show "LIVE / PAPER / BACKTEST" prominently

## Row 8-16: Signal Activity

### "Signal Count by Strategy" (bar chart, 12-wide)
- **Question it answers:** Which strategies are generating the most trading signals?
- **What's confusing:** Shows cumulative counts since system start. After running for days, all bars will be large, making it hard to see recent activity. Green/red/purple color coding for long/short/flat is good.
- **What's missing:** A rate view ("signals per hour") would be more useful than cumulative totals.
- **Rename suggestion:** "Strategy Signal Activity" — consider "Signals per Hour by Strategy"

### "Signal Activity by Symbol" (bar chart, 6-wide)
- **Question it answers:** Which markets are getting the most trading attention?
- **What's confusing:** Same cumulative-total problem. Squished to 6-wide, making bar labels hard to read.
- **What's missing:** Context for whether this distribution is expected.
- **Rename suggestion:** "Market Activity by Symbol"

### "Candles Ingested" (stat, 6-wide)
- **Question it answers:** Is market data flowing in?
- **What's confusing:** "Candles" is trading-specific jargon. "Ingested" is engineering jargon. Non-technical viewers have no idea what this means.
- **What's missing:** A rate or "last updated" timestamp would be far more useful than a raw count.
- **Rename suggestion:** "Market Data Feed" — show "Last update: X seconds ago" instead of a raw count

## Row 16-38: Strategy Playbook (TEXT PANELS)

### "Trend Following," "Mean Reversion," "Breakout," "Funding Arbitrage" (4 text panels, 20 rows total)
- **Question they answer:** How does each strategy work?
- **What's confusing:** This is static reference documentation sitting in the middle of a real-time monitoring dashboard. It takes up ~20% of the total page height and provides zero real-time value.
- **What's missing:** This content is valuable but belongs on a separate page (a wiki, linked doc, or separate Grafana dashboard). In its place, the main dashboard should show real-time performance per strategy.
- **Rename suggestion:** Move entirely to a separate "Strategy Reference Guide" dashboard with a link from the main dashboard.

## Row 38-46: Latency & Drawdown

### "Drawdown %" (gauge, 8-wide)
- **Question it answers:** How far has our portfolio dropped from its peak?
- **What's confusing:** Good gauge visualization with green/yellow/orange/red thresholds. However, the max scale is set to 30%, which may not match actual risk limits.
- **What's missing:** Should show the actual configured risk limit line.
- **Rename suggestion:** "Portfolio Drawdown" — add subtitle "Distance from peak equity"

### "Decision Latency (p95 / p99)" (timeseries, 8-wide)
- **Question it answers:** How fast is the system making trading decisions?
- **What's confusing:** "p95" and "p99" are statistical terms that most non-engineers won't understand. "95th percentile" means "95% of decisions are faster than this number." The underlying query uses `histogram_quantile(0.95, rate(...))` which is pure engineering syntax that appears in the legend.
- **What's missing:** A threshold line showing "this is too slow" — the system has decision latency buckets up to 1 second, but the panel doesn't show what's acceptable.
- **Rename suggestion:** "Decision Speed" — show "Typical" (p95) and "Slowest" (p99) as legend labels

### "Order Latency (p95 / p99)" (timeseries, 8-wide)
- **Same issues as Decision Latency** — jargon in title and legend.
- **Rename suggestion:** "Order Confirmation Speed" — "How long until the exchange confirms our orders"

## Row 46-72: Governance (Soteria)

### Section header: "--- GOVERNANCE ---"
- **What's confusing:** "Governance Framework (Soteria)" means nothing to anyone outside the engineering team. Soteria is an internal code name.
- **Rename suggestion:** "Safety & Controls" or "Trading Guardrails"

### "Governance Decisions" (timeseries, 12-wide)
- **Question it answers:** How often are safety checks approving or blocking trades?
- **What's confusing:** Shows "rate per second" (`rate(...[5m])`) which produces small decimal numbers. "0.002 ops" is meaningless to most people. Should show counts.
- **What's missing:** A simple "X approved / Y blocked in the last hour" would be far more actionable.
- **Rename suggestion:** "Safety Check Results" — "How many trades were approved vs blocked"

### "Governance Blocks" (timeseries, 12-wide)
- **Question it answers:** Why were trades blocked?
- **What's confusing:** Same rate-per-second issue. Legend shows raw reason codes.
- **What's missing:** The reasons should be human-readable. "MATURITY_GATE" → "Strategy not ready for live trading."
- **Rename suggestion:** "Blocked Trades — Reasons"

### "Strategy Health Scores" (timeseries, 8-wide)
- **Question it answers:** Are strategies behaving normally?
- **What's confusing:** "Health score 0.0 to 1.0" has no intuitive meaning. What's healthy? The metric name in the query is `trading_health_score` but the actual Prometheus metric is `trading_strategy_health_score` — possible mismatch.
- **What's missing:** Threshold lines (above 0.7 = healthy, below 0.4 = needs attention).
- **Rename suggestion:** "Strategy Health" — use text mappings: "Healthy / Needs Attention / Critical"

### "Drift Deviation %" (timeseries, 8-wide)
- **Question it answers:** Are strategies behaving differently than their historical baseline?
- **What's confusing:** "Drift" is a machine learning / monitoring concept. Most users won't understand what "drifting" means in a trading context.
- **What's missing:** An explanation of what deviation is being measured and what the consequence of drift is.
- **Rename suggestion:** "Strategy Behavior Change" — "How much each strategy has deviated from its normal pattern"

### "Active Execution Tokens" (stat, 4-wide)
- **Question it answers:** How many strategies currently have permission to trade?
- **What's confusing:** "Execution tokens" is internal system jargon. This is a governance concept where strategies need a "token" to trade — like a permission slip.
- **What's missing:** Context for what this number should be.
- **Rename suggestion:** "Strategies Allowed to Trade" — shows how many of your strategies currently have live trading permission

### "Canary Status" (stat, 4-wide)
- **Question it answers:** Are background health-check components working?
- **What's confusing:** "Canary" refers to "canary in a coal mine" — an early warning system. This is engineering jargon.
- **What's missing:** Which canary components exist and which ones might be failing.
- **Rename suggestion:** "Early Warning System" — show "All Clear" / "Warning" instead of "HEALTHY" / "FAILING"

### "Governance Gate Latency" (timeseries, 12-wide)
- **Same p95/p99 jargon issue** as the decision/order latency panels.
- **Rename suggestion:** "Safety Check Speed" — "How fast the guardrail checks are running"

### "Maturity Levels" (timeseries, 12-wide)
- **Question it answers:** What promotion level has each strategy reached?
- **What's confusing:** "Maturity Level" with values 0-4 is opaque. The legend tries to show "{{strategy}} = {{level}}" but the level label doesn't resolve to a human name.
- **What's missing:** L0-L4 should be named (e.g., L0=New, L1=Testing, L2=Paper, L3=Limited Live, L4=Full Live).
- **Rename suggestion:** "Strategy Promotion Status" — "Which trading stage each strategy has reached"

## Row 72-98: Trade Journal & Analytics

### Section header: "--- TRADE JOURNAL & ANALYTICS ---"
- **What's confusing:** "Edgewonk-inspired" is a product reference that means nothing to most users.
- **Rename suggestion:** "Trade Performance Analytics"

### "Win Rate (Rolling)" (timeseries, 8-wide)
- **Question it answers:** What percentage of trades are profitable?
- **What's confusing:** "Rolling" is moderately technical but acceptable. Good threshold coloring.
- **Rename suggestion:** "Win Rate by Strategy" — add subtitle "% of trades that made money"

### "Profit Factor (Rolling)" (timeseries, 8-wide)
- **Question it answers:** Are the winners bigger than the losers?
- **What's confusing:** "Profit Factor" is a specific trading metric (total gains / total losses). Not intuitive.
- **What's missing:** A reference line at 1.0 (breakeven) with a label.
- **Rename suggestion:** "Gains vs. Losses Ratio" — add description "Above 1.0 = gains outweigh losses"

### "Avg R-Multiple" (timeseries, 8-wide)
- **Question it answers:** On average, how much do we gain per unit of risk taken?
- **What's confusing:** "R-Multiple" is professional trading jargon. R = the initial risk amount. An R-multiple of 2.0 means you made 2x your initial risk.
- **Rename suggestion:** "Average Reward per Risk" — add description "How much we earn for each dollar risked"

### "Expectancy ($/trade)" (stat, 6-wide)
- **Question it answers:** On average, how much does each trade make or lose?
- **What's confusing:** Reasonably clear for trading-literate users.
- **Rename suggestion:** "Average Profit per Trade"

### "Sharpe (Trade-level)" (stat, 6-wide)
- **Question it answers:** How consistent are the returns relative to their variability?
- **What's confusing:** "Sharpe ratio" is finance jargon. "(Trade-level)" adds another layer of confusion — it means calculated per trade, not annualized.
- **Rename suggestion:** "Return Consistency" — add description "Higher = more reliable profits"

### "Management Efficiency" (gauge, 6-wide)
- **Question it answers:** How well are we capturing potential profits in our trades?
- **What's confusing:** "Management Efficiency" doesn't convey the actual meaning (actual profit captured vs. maximum possible profit on each trade).
- **Rename suggestion:** "Profit Capture Rate" — "What % of each trade's potential profit we actually captured"

### "Open / Closed Trades" (stat, 6-wide)
- **Question it answers:** How many trades are currently active vs. completed?
- **What's confusing:** Clear enough.
- **Rename suggestion:** Keep as-is, but consider "Active Trades / Completed Trades"

### "Rolling Max Drawdown" (gauge, 8-wide)
- **Question it answers:** What's the worst loss streak for each strategy?
- **What's confusing:** Threshold values are in absolute USD ($1000 yellow, $5000 red) which may not scale with account size.
- **What's missing:** Should be percentage-based to be meaningful across account sizes.
- **Rename suggestion:** "Worst Loss Streak by Strategy"

### "Confidence Calibration (Brier Score)" (timeseries, 8-wide)
- **Question it answers:** When a strategy says it's 80% confident, is it actually right 80% of the time?
- **What's confusing:** "Brier Score" is a specialized statistics concept. Even the parenthetical doesn't help — most people don't know what "confidence calibration" means in this context.
- **What's missing:** A clear explanation that lower is better, and what the scale means.
- **Rename suggestion:** "Prediction Accuracy" — add description "How well each strategy's confidence scores match actual results. Lower = more accurate."

### "Statistical Edge (p-value)" (timeseries, 8-wide)
- **Question it answers:** Is the strategy's performance due to skill or just luck?
- **What's confusing:** "p-value" is statistics jargon. The threshold at 0.05 is the conventional significance level, but only statisticians know this.
- **Rename suggestion:** "Is This Skill or Luck?" — add description "Below 0.05 (green) = likely real skill. Above 0.25 (red) = could be random chance."

### "Overtrading Alert" (stat, 6-wide)
- **Question it answers:** Is a strategy trading too frequently?
- **What's confusing:** Clear labeling with NORMAL/OVERTRADING.
- **Rename suggestion:** Keep "Overtrading Alert" — it's clear.

### "Probability of Ruin (MC)" (stat, 6-wide)
- **Question it answers:** What are the chances this strategy blows up the account?
- **What's confusing:** "(MC)" stands for Monte Carlo simulation — opaque to non-technical users.
- **Rename suggestion:** "Risk of Total Loss" — add description "Simulated probability that this strategy could deplete the account"

### "Kelly Fraction" (stat, 6-wide)
- **Question it answers:** What's the mathematically optimal bet size for this strategy?
- **What's confusing:** "Kelly Fraction" is advanced portfolio theory. Named after John Kelly's 1956 formula — most people won't know this.
- **Rename suggestion:** "Recommended Position Size" — add description "Mathematically optimal allocation based on win rate and payoff"

### "Trades per Strategy" (timeseries, 6-wide)
- **Question it answers:** How many trades has each strategy taken, and with what outcome?
- **What's confusing:** Small panel (6-wide) for a time series that includes strategy × outcome combinations, leading to many overlapping lines.
- **Rename suggestion:** "Trade Count by Strategy" — consider switching to a table or stacked bar

---

# 5) Proposed Redesign (No-Code UX Plan)

## New Layout: Three Linked Dashboards

Instead of one 98-row page, create three focused dashboards connected by links:

### Dashboard A: "Trading Overview" (Everyone's Home Page)
This is what you see when you open Grafana. It answers "are we OK?" in 5 seconds and "what changed?" in 30 seconds.

**Row 1: Health Summary Bar (always visible at top, 1 row tall)**
| Panel | Width | Type | Content |
|---|---|---|---|
| System Status | 5 | Stat | Green "Running" / Red "Stopped" / Orange "Degraded" — combines kill switch + canary + mode |
| Portfolio Health | 5 | Stat | Green/Yellow/Red based on drawdown thresholds + daily PnL direction |
| Risk Guardrails | 5 | Stat | Green "All Clear" / Yellow "Approaching Limits" / Red "Limits Breached" — combines governance blocks + risk check failures |
| Data Feed | 5 | Stat | Green "Live" / Yellow "Delayed" / Red "Stale" — based on data staleness metric |
| Emergency Stop | 4 | Stat | Current kill switch status — big red button visual if active |

**Row 2: Key Numbers (4 stats, 1 row)**
| Panel | Width | Type | Content |
|---|---|---|---|
| Today's P&L | 6 | Stat | Daily profit/loss in USD + percentage of equity |
| Portfolio Value | 6 | Stat | Current equity |
| Capital at Risk | 6 | Stat | Gross exposure + % of equity |
| Active Trades | 6 | Stat | Open trade count |

**Row 3-4: Portfolio Trend (1 large chart)**
| Panel | Width | Type | Content |
|---|---|---|---|
| Portfolio Value Over Time | 24 | Time series | Equity curve with daily PnL as bar overlay |

**Row 5-6: Activity Summary (2 panels)**
| Panel | Width | Type | Content |
|---|---|---|---|
| Strategy Scorecard | 12 | Table | One row per strategy: Name, Status, Win Rate, Today's P&L, Signal Count, Health. Sortable. |
| Recent Safety Events | 12 | Table | Last 10 governance blocks/alerts: Time, Strategy, Reason (plain English), Action Taken |

**Row 7: Navigation Links**
| Panel | Width | Type | Content |
|---|---|---|---|
| Navigation | 24 | Text/Links | "Deep Dives: [Strategy Performance →] [Risk & Governance →] [System Health →] [Strategy Reference Guide →] [Live Narration →]" |

**Total height: ~7 rows (~56 units).** Fits on one screen without scrolling.

---

### Dashboard B: "Strategy Performance" (Trading Ops Deep Dive)
Linked from Overview. For the Trading Ops Lead who needs to evaluate and compare strategies.

**Row 1: Strategy Comparison Table (the most useful panel)**
| Column | Metric | What it tells you |
|---|---|---|
| Strategy | Name | Which strategy |
| Status | Health score mapped to text | Healthy / Needs Attention / Critical |
| Win Rate | % | How often it wins |
| Gains vs. Losses | Profit factor | Are winners bigger? (>1 = yes) |
| Avg Profit/Trade | Expectancy | Dollar value per trade |
| Return per Risk | Avg R-multiple | Efficiency of risk-taking |
| Profit Capture | Management efficiency | How well we ride winners |
| Prediction Accuracy | Brier score (inverted for readability) | Is it calibrated? |
| Trades Today | Count | Volume |
| Overtrading? | Boolean | Flag |

**Row 2-3: Win Rate Trends** (time series, full width)

**Row 4-5: Gains vs. Losses Ratio Trends** (time series, full width)

**Row 6-7: Strategy Signal Breakdown** (bar chart by strategy and direction)

**Row 8-9: Market Activity** (bar chart by symbol)

**Row 10-11: Trade Count by Outcome** (stacked bar — wins vs losses by strategy)

---

### Dashboard C: "Risk & Governance" (Risk Analyst Deep Dive)
Linked from Overview. For the Risk Analyst and compliance functions.

**Row 1: Risk Summary**
| Panel | Width | Content |
|---|---|---|
| Portfolio Drawdown | 8 | Gauge with limit line |
| Risk of Total Loss | 8 | Per-strategy probability |
| Recommended Position Size | 8 | Kelly fraction per strategy |

**Row 2-3: Drawdown Over Time** (time series — worst loss streak per strategy)

**Row 4-5: Safety Check Results** (time series — approved vs blocked, in counts not rates)

**Row 6-7: Blocked Trade Reasons** (bar chart with human-readable reason labels)

**Row 8-9: Strategy Behavior Change** (drift deviation over time)

**Row 10-11: Strategy Promotion Status** (maturity levels with named stages)

**Row 12-13: Is This Skill or Luck?** (p-value time series with significance line at 0.05)

---

### Dashboard D: "System Health" (On-Call Deep Dive)
For the On-Call Responder diagnosing operational issues.

**Row 1-2: Decision Speed + Order Speed + Safety Check Speed** (3 time series side by side)

**Row 3: Data Feed Status** (per-symbol staleness gauges)

**Row 4: Early Warning System** (per-component canary status)

**Row 5: Strategies Allowed to Trade** (execution token count)

**Row 6: Market Data Processing** (candles per symbol, as a rate)

---

### Dashboard E: "Strategy Reference Guide" (Moved from main dashboard)
A separate, non-monitoring dashboard for documentation. Contains the four strategy description panels currently embedded in the trading dashboard, plus the 8 new CMT strategies. No real-time data. Updated manually when strategies change.

---

## Naming Conventions

### Rules
1. **Use questions or outcomes, not metric names.** "How fast are decisions?" not "Decision Latency (p95/p99)"
2. **Avoid abbreviations.** Write "Profit and Loss" the first time; "P&L" after that is fine.
3. **Avoid internal code names.** No "Soteria," "Edgewonk," "Canary," "Kelly." Use the concept instead.
4. **Every panel gets a description.** One sentence explaining what the number means and what "good" looks like.
5. **Use consistent color rules everywhere.** Green = good/within limits. Yellow = caution/approaching limits. Red = action needed/limits breached.

### Before → After Examples

| Before (current) | After (proposed) |
|---|---|
| Kill Switch | Emergency Stop |
| Gross Exposure | Total Capital at Risk |
| System Mode | Trading Mode (Live / Paper / Backtest) |
| Candles Ingested | Market Data Feed |
| Decision Latency (p95 / p99) | Decision Speed |
| Order Latency (p95 / p99) | Order Confirmation Speed |
| --- GOVERNANCE --- / Governance Framework (Soteria) | Safety & Controls |
| Governance Decisions | Safety Check Results |
| Governance Blocks | Blocked Trades — Reasons |
| Strategy Health Scores | Strategy Health |
| Drift Deviation % | Strategy Behavior Change |
| Active Execution Tokens | Strategies Allowed to Trade |
| Canary Status | Early Warning System |
| Governance Gate Latency | Safety Check Speed |
| Maturity Levels | Strategy Promotion Status |
| --- TRADE JOURNAL & ANALYTICS --- / Edgewonk-inspired | Trade Performance Analytics |
| Profit Factor (Rolling) | Gains vs. Losses Ratio |
| Avg R-Multiple | Average Reward per Risk |
| Sharpe (Trade-level) | Return Consistency |
| Management Efficiency | Profit Capture Rate |
| Confidence Calibration (Brier Score) | Prediction Accuracy |
| Statistical Edge (p-value) | Is This Skill or Luck? |
| Probability of Ruin (MC) | Risk of Total Loss |
| Kelly Fraction | Recommended Position Size |

### Chart Recommendations

| Current type | Problem | Recommended change |
|---|---|---|
| Cumulative counter bar charts (Signal Count, Symbol Activity) | Numbers only grow, hard to spot recent changes | Switch to rate-based (per hour) or use "increase over time range" |
| Rate-based time series (`rate(...[5m])`) for governance | Shows tiny decimals (0.002 ops) meaningless to users | Switch to `increase()` or cumulative count for the time window |
| Individual stat panels for journal metrics (Expectancy, Sharpe, Kelly, etc.) | 6+ small panels in a row — cognitive overload | Consolidate into a sortable table with one row per strategy |
| Maturity Levels as time series | Values are integers 0-4, time series looks like staircases | Switch to a table with stage names or a step-indicator visual |
| "p95" / "p99" in legends | Meaningless to non-engineers | Use "Typical" (p95) and "Slowest" (p99) |

---

# 6) Prioritized Backlog

| Priority | Item | User problem solved | Expected benefit | Effort | Owner |
|---|---|---|---|---|---|
| **P0** | **Add health summary row at top of main dashboard** | "I can't tell if things are OK without checking 6 panels" | 5-second status check for ALL personas; reduces false escalations | M | UX + Eng |
| **P0** | **Rename all panels using plain-English naming conventions** | "I don't understand what these numbers mean" | Non-technical users can self-serve; fewer questions to engineers | S | UX |
| **P0** | **Add descriptions to every panel** explaining what the number means and what "good" looks like | "I see a number but don't know if it's good or bad" | Users make correct interpretations without training | S | UX + Product |
| **P0** | **Move strategy documentation to a separate reference dashboard** | "I have to scroll through 20 rows of text to reach monitoring data" | Main dashboard is 40% shorter; monitoring-focused | S | UX |
| **P1** | **Create "Trading Overview" dashboard** with the layout described in Section 5 | "I can't find what I need — everything is on one page" | Role-appropriate landing page; 80% of questions answered on first screen | L | UX + Eng |
| **P1** | **Create strategy scorecard table panel** (one row per strategy with key metrics) | "I have to check 8 separate panels to compare strategies" | Side-by-side comparison in one glance; much faster strategy evaluation | M | Eng |
| **P1** | **Convert governance rate metrics to counts** | "0.002 ops means nothing to me" | Numbers are intuitive (e.g., "12 trades blocked in the last hour") | S | Eng |
| **P1** | **Replace p95/p99 legend labels with "Typical" / "Slowest"** | "What does p99 mean?" | Latency panels are instantly readable by non-engineers | S | UX |
| **P1** | **Add threshold/limit reference lines to gauges and time series** | "I see 7% drawdown but don't know if that's OK" | Users can self-assess without looking up risk limits | S | UX + Eng |
| **P2** | **Create "Strategy Performance" deep-dive dashboard** | Trading Ops needs detailed strategy analysis separate from risk/ops panels | Persona-appropriate view; faster strategy evaluation workflow | M | UX + Eng |
| **P2** | **Create "Risk & Governance" deep-dive dashboard** | Risk Analysts wade through strategy and system data to find risk metrics | Focused risk view; faster compliance checks | M | UX + Eng |
| **P2** | **Create "System Health" deep-dive dashboard** | On-call has to check 6+ panels across different sections to diagnose issues | Focused operational view; faster incident response | M | UX + Eng |
| **P2** | **Add human-readable reason labels for governance blocks** | "MATURITY_GATE means what?" | Block reasons are immediately actionable | S | Eng |
| **P2** | **Convert signal count charts to rate-based (per hour)** | Cumulative counters are meaningless after the first day | Shows actual current activity, not historical totals | S | Eng |
| **P2** | **Add "what to do next" guidance text near alert panels** | "Something is red — now what?" | Faster response times; fewer escalations for known issues | S | UX + Product |
| **P2** | **Add Grafana variables for strategy and symbol filtering** | "I only care about one strategy right now" | Users can focus on what matters; less visual noise | M | Eng |
| **P2** | **Add maturity level stage names (L0=New, L1=Testing, etc.)** | "Level 2 means what?" | Promotion stages are instantly understandable | S | UX + Eng |

---

# 7) Success Metrics & Acceptance Criteria

## How We'll Measure Success

| Metric | How to measure | Target |
|---|---|---|
| **Time-to-answer: "Is the system OK?"** | Timed user test: ask 5 users "is the system healthy?" and measure how long it takes | Under 10 seconds (currently estimated 2-3 minutes) |
| **Time-to-answer: "Which strategy is underperforming?"** | Timed user test: ask users to identify the worst strategy | Under 30 seconds (currently estimated 1-2 minutes) |
| **Self-service rate** | Track how often non-engineers ask engineers "what does this mean?" after launch (survey or Slack mentions) | 50% reduction in dashboard-related questions within 4 weeks |
| **Confidence score** | Post-launch survey: "How confident are you that you can independently use this dashboard?" (1-5 scale) | Average 4.0+ across all personas |
| **Escalation accuracy** | Track whether escalations from the dashboard are valid (true positive rate) | 80%+ valid escalation rate (i.e., fewer false alarms) |
| **Dashboard adoption** | Grafana usage analytics: unique viewers per week | All 4 persona types accessing at least weekly |

## Acceptance Criteria for the Redesigned Dashboard

### Must-have (P0 items complete)
- [ ] A user with no trading background can determine system health status in under 10 seconds from the landing page
- [ ] Every panel title is in plain English — no unexplained jargon, abbreviations, or internal code names
- [ ] Every panel has a description field explaining what the metric means and what "good" looks like
- [ ] Strategy documentation is on a separate dashboard, not mixed with monitoring panels
- [ ] Kill switch / Emergency Stop is the most visually prominent alert on the page

### Should-have (P1 items complete)
- [ ] A "Trading Overview" landing dashboard exists with the health summary bar, key numbers, equity chart, strategy scorecard table, and navigation links
- [ ] Governance metrics show human-readable counts, not rates-per-second
- [ ] Latency panels use "Typical" and "Slowest" instead of "p95" and "p99"
- [ ] Gauges show configured limit lines for context

### Nice-to-have (P2 items complete)
- [ ] Separate deep-dive dashboards exist for Strategy Performance, Risk & Governance, and System Health
- [ ] Grafana variables allow filtering by strategy and symbol
- [ ] Action guidance text appears near red/alert panels
- [ ] All bar charts show rates (per hour), not cumulative totals
- [ ] Maturity levels display stage names, not numbers

### Non-functional requirements
- [ ] All dashboards load in under 3 seconds
- [ ] Dashboard JSON files remain version-controlled in `configs/grafana/dashboards/`
- [ ] New dashboards use the same provisioning mechanism (auto-deployed with Docker)
- [ ] The narration dashboard remains unchanged (it's already excellent)
