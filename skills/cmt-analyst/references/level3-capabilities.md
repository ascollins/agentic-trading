# Level III Capabilities Reference — The Integration of Technical Analysis

Source: CMT Level III 2022 Curriculum (29 chapters, 6 sections)

L3 is about integration: taking everything from L1 and L2 and applying it to real-world portfolio management, risk control, and multi-asset decision-making. The emphasis shifts from "can you identify this pattern?" to "given this complete picture, what do you DO?"

---

## Section I: Risk Management (L3 Chs 1-7)

### System Design & Testing (L3 Ch 1)
- **Value and challenges of systematic trading**: Systems provide consistency, remove emotion, enable backtesting. Challenges: curve-fitting, regime changes, implementation slippage, psychological discipline to follow the system during drawdowns.
- **Discretionary vs. nondiscretionary**: Discretionary = analyst makes final call (flexible but bias-prone). Nondiscretionary = rules execute automatically (consistent but inflexible). Hybrid approach: systematic framework with discretionary overrides only for extreme conditions.
- **Mindset and discipline**: The hardest part isn't designing the system — it's FOLLOWING it. Especially during drawdowns, when every instinct says to override the rules. The system is only as good as your discipline to execute it.
- **System design procedures**: (1) Define objective (absolute return, risk-adjusted, relative), (2) select market/timeframe, (3) specify entry/exit rules, (4) define position sizing, (5) test, (6) validate out-of-sample, (7) implement with monitoring.
- **Technical trading system types**: Trend-following, mean-reversion, breakout, pattern-based, composite/multi-factor.
- **Risk management protocols**: Non-negotiable. Every system requires: maximum position size, portfolio heat limit, stop-loss logic, correlation awareness, drawdown circuit-breakers.

### Money & Portfolio Risk Management (L3 Ch 2)
- **Trading strategies vs. money management strategies**: Entry/exit rules determine WHEN to trade. Money management determines HOW MUCH to trade. Money management has greater impact on long-term results than signal quality.
- **Theory of runs**: Consecutive losses (runs) are inevitable. A 50% win-rate system will experience runs of 7+ losses. Position sizing must survive the worst expected run.
- **Martingale vs. anti-martingale**: Martingale (increase size after loss) = recipe for ruin. Anti-martingale (increase size after win) = allows compounding while protecting during drawdowns. Always use anti-martingale logic.
- **Risk of ruin**: Mathematical probability of account reaching zero (or a predefined loss threshold). Function of: win rate, average win/loss ratio, and fraction risked per trade. Risk of ruin must be <1% for any viable system.
- **Optimal f**: The Kelly-derived fraction of capital that maximises geometric growth rate. In practice, use half-Kelly or less because: (1) parameters are estimated, not known, (2) ruin is permanent, (3) drawdowns at full Kelly are psychologically devastating.
- **Diversifiable vs. correlated risk**: Diversification reduces unsystematic (asset-specific) risk but NOT systematic (market) risk. In crises, correlations spike — "diversification fails when you need it most." True portfolio protection requires hedging, not just diversification.
- **Stop types**: Fixed dollar, percentage, ATR-based, time-based, pattern-based, trailing. Each has trade-offs between protection and whipsaw frequency.
- **Minimum capital**: Calculate based on: maximum position size × number of concurrent positions × margin requirements + drawdown cushion. Undercapitalisation is the most common reason systems fail in practice.

### System Evaluation & Testing (L3 Chs 3-4)
- **Testing factors**: Define objectives (what constitutes "success"), parameters (what variables to test), and test data (representative, sufficient length, includes different regimes).
- **In-sample vs. out-of-sample**: Develop on in-sample (60-70% of data). Validate on out-of-sample (30-40%). NEVER go back and re-optimise after seeing out-of-sample results.
- **Visualising optimisation results**: 3D surface plots of parameter combinations. Look for "plateaus" (robust — performance stable across parameter range) not "peaks" (fragile — performance depends on exact parameters).
- **Genetic algorithms**: Evolutionary optimisation. Population of parameter sets → evaluate fitness → select best → crossover/mutation → new generation. Advantage: explores solution space without exhaustive search. Risk: may converge on local optima.
- **Robustness**: A system is robust if: (1) it works across a range of parameters, (2) it works across multiple markets, (3) it works across different time periods, (4) out-of-sample performance resembles in-sample. If ANY of these fail, the system is likely curve-fitted.
- **Performance and risk metrics**: Choose metrics aligned with objectives. Maximising return ≠ maximising risk-adjusted return. Calmar ratio may be more relevant than Sharpe for a trader focused on drawdown control.
- **Price shocks**: Plan for gap events, circuit breakers, flash crashes. Stress-test systems against historical shocks (1987, 2008, 2020, etc.). Stops may not fill at expected levels during shocks.
- **Trend-following vs. mean-reversion trade-offs**: Trend systems: lower win rate (~35-45%), larger average win, suffer in ranges. Mean-reversion: higher win rate (~55-65%), smaller average win, suffer in trends. Optimal approach often combines both, with an environmental model to determine regime.

### Risk Control (L3 Ch 5)
- **Sharpe Ratio**: (Rp - Rf) / σp. Measures excess return per unit of total risk. >1.0 = acceptable, >2.0 = excellent. Limitation: penalises upside volatility equally with downside.
- **Information Ratio**: (Rp - Rb) / σ(Rp-Rb). Measures active return per unit of tracking error. Relevant for benchmark-relative strategies.
- **Treynor Ratio**: (Rp - Rf) / βp. Measures excess return per unit of SYSTEMATIC risk. Use when evaluating a component within a diversified portfolio.
- **Calmar Ratio**: Annualised return / Maximum drawdown. Practical for traders — directly measures the "pain to gain" relationship. >3.0 = excellent.
- **Sortino Ratio**: (Rp - Rf) / Downside deviation. Like Sharpe but only penalises DOWNSIDE volatility. Better for strategies with asymmetric return profiles (e.g., trend-following with occasional large wins).
- **Value at Risk (VaR)**: Maximum expected loss at a given confidence level over a given period. Example: "95% VaR of $10,000 means there's a 5% chance of losing more than $10,000." Methods: parametric (assumes normal distribution — underestimates tail risk), historical simulation, Monte Carlo. Limitation: VaR doesn't tell you HOW BAD losses can be beyond the VaR threshold. Use CVaR/Expected Shortfall for tail risk.
- **Stop and target methods**: Fixed percentage, ATR-multiple, support/resistance-based, trailing, time-based. Compare methods' impact on system metrics.
- **Position sizing approaches**: Fixed fractional (risk X% per trade), fixed ratio, volatility-adjusted (position size = risk / ATR), Kelly/Optimal f. Each produces different risk/return profiles.
- **Compounding positions**: Adding to winners (pyramiding) vs. averaging down (adding to losers). Anti-martingale says add to winners only. Pyramiding rules: each addition should be smaller than the previous, and move stop to protect accumulated profit.
- **Risk of ruin calculation**: RoR = ((1 - Edge) / (1 + Edge))^Units. Where Edge = (Win% × Avg Win - Loss% × Avg Loss) / Avg Loss. Units = account / risk per trade. Must be <1%.
- **Optimal f formula**: f* = (bp - q) / b. Where b = avg win / avg loss, p = probability of win, q = probability of loss. Maximum geometric growth at f*, but drawdowns at f* can exceed 50% — use half-Kelly or less in practice.

### Statistical Analysis (L3 Ch 6)
- Assess random vs. nonrandom trends in performance: A system showing nonrandom performance sequences (clustering of wins/losses) may have regime dependency — investigate.
- Sampling and sample statistics: Larger samples give more reliable estimates. Minimum ~30 trades for any meaningful statistical inference, preferably 100+.
- Relative frequency: Empirical probability from observed data. Use for win rate estimation.
- Six elements of statistical inference: population, sample, statistic, parameter, sampling distribution, inference.
- Theoretical vs. empirical probabilities: Theoretical = calculated from assumptions. Empirical = observed from data. Market probabilities are ALWAYS empirical — never assume theoretical distributions.

### Hypothesis Tests & Confidence Intervals (L3 Ch 7)
- **Necessary vs. sufficient conditions**: "RSI < 30 is necessary for a buy signal" (must be true but may not be enough alone). "RSI < 30 AND bullish divergence is sufficient" (triggers the trade). Build systems with both.
- **Null hypothesis**: Always frame as the skeptical position: "This trading rule has no predictive power." The burden of proof is on demonstrating the rule works, not on disproving it.
- **Why frame null as target**: You're trying to REJECT the null (no edge). If you can't reject it at an appropriate significance level, the rule may be spurious.
- Confidence intervals for system parameters: Express win rate, average profit, Sharpe ratio etc. as ranges, not point estimates. The width of the interval reflects uncertainty.

---

## Section II: Asset Relationships (L3 Chs 8-15)

### Regression in Intermarket Analysis (L3 Ch 8)
- Multiple regression for multi-factor asset models: Predict asset returns from multiple inputs (rates, commodities, currencies, etc.)
- Tolerance calculations: Detect multicollinearity among predictor variables. High multicollinearity = unreliable coefficients.
- Meaningful predictor selection: Theory-first, then statistical validation. Avoid data-mining predictors.

### Asset Correlation Studies (L3 Chs 9-13)
- **International indices & commodities** (L3 Ch 9): Cross-market correlations for global diversification and timing. Regression methods to quantify relationships.
- **S&P 500 relationships** (L3 Ch 10): Correlations between SPX and international markets, bonds, commodities, dollar. These correlations SHIFT across regimes — monitor dynamically.
- **European indices** (L3 Ch 11): Intra-European correlations. Decoupling patterns during sovereign stress. Euro-denominated vs. local-currency returns.
- **Gold** (L3 Ch 12): Gold-dollar inverse correlation. Gold-stocks conditional (negative in risk-off, uncorrelated in normal markets, positive in stagflation). Gold as crisis hedge and debasement trade.
- **Intraday correlations** (L3 Ch 13): Correlation characteristics change across timeframes. Daily correlations may differ significantly from hourly or minute-level. Index futures show tighter intraday correlations during US hours, looser during Asian/European hours.

### Intermarket Indicators (L3 Ch 14)
- Construct relative strength studies across asset classes
- Compare intermarket indicators: yield spreads, commodity ratios, currency crosses
- Prepare recommendations from correlation data: Overweight assets in favorable relative position, underweight unfavorable
- Integration: Don't just observe intermarket — use it to inform individual market analysis

### Relative Rotation Graphs — RRG (L3 Ch 15)
- **Concept**: Plot relative strength (x-axis: RS-Ratio) against its momentum (y-axis: RS-Momentum) for multiple assets simultaneously.
- **Four quadrants**:
  1. **Leading** (upper right): Strong AND improving relative strength. Leaders.
  2. **Weakening** (lower right): Strong BUT deteriorating momentum. Former leaders beginning to fade.
  3. **Lagging** (lower left): Weak AND deteriorating. Laggards.
  4. **Improving** (upper left): Weak BUT gaining momentum. Potential future leaders.
- **Clockwise rotation**: The natural pattern — Leading → Weakening → Lagging → Improving → Leading. Healthy markets show orderly rotation.
- **Application**: Sector selection (overweight Leading, underweight Lagging), market comparison (which country/region is leading), portfolio rebalancing (rotate from Weakening to Improving), timing (enter Improving sectors before they reach Leading).
- **Derived indicators**: RS-Ratio trend and RS-Momentum trend provide additional signal granularity.

---

## Section III: Portfolio Management (L3 Chs 12-14/16-18)

### Momentum Investing (L3 Ch 12/16)
- Historical evidence: Momentum strategies have positive returns across markets, asset classes, and time periods. One of the most robust anomalies in finance.
- Common myths debunked: "Momentum is just market risk" (no — works long/short). "Momentum is just small-cap" (no — works across cap ranges). "Momentum is just risk" (no — risk-adjustment reduces but doesn't eliminate alpha).
- Implementation: Rank assets by trailing returns (3-12 months typically), buy top decile, sell/short bottom decile, rebalance monthly.

### Macro-Finance Environment (L3 Ch 13/17)
- **Business cycle**: Expansion → peak → contraction → trough. Each phase favors different sectors and asset classes.
- **Financial cycle** (credit cycle): Can diverge from business cycle. Credit expansion can prolong business expansion. Credit contraction can deepen recessions.
- **Sector rotation model**: Map current cycle phase → overweight favorable sectors. Early expansion: financials, consumer discretionary, technology. Mid expansion: industrials, materials. Late expansion: energy, utilities. Contraction: healthcare, consumer staples, utilities.
- **Leading, coincident, lagging indicators**: Leading (yield curve, building permits, stock market, PMI) predict turns. Coincident (employment, GDP, industrial production) confirm current phase. Lagging (unemployment rate, CPI, interest rates) confirm what already happened.
- **Integration with TA**: Use macro-finance to set the DIRECTION and RISK BUDGET. Use technical analysis to TIME entries and manage RISK.

### Portfolio Risk & Performance Attribution (L3 Ch 14/18)
- **Total risk = volatility = standard deviation of returns**: The foundational equation.
- **Three formulations of total risk**: (1) Historical standard deviation, (2) forward-looking (implied from options), (3) conditional (regime-dependent — risk is higher in certain environments).
- **Diversification reduces only firm-specific risk**: Systematic (market) risk cannot be diversified away. Adding more stocks beyond ~30 provides minimal additional diversification benefit.
- **Beta**: Sensitivity to market returns. Beta = 1.0 = market risk. >1.0 = more volatile than market. <1.0 = less volatile. Use beta to understand portfolio's market exposure.
- **Sharpe ratio for portfolios**: Evaluates the portfolio as a whole. Allows comparison between portfolios with different risk levels.
- **Treynor ratio for components**: Evaluates individual positions within a diversified portfolio. Uses beta instead of total risk, because firm-specific risk is diversified away at portfolio level.
- **Performance attribution**: Decompose returns into: market timing (being in/out at right times), sector selection (right sectors), security selection (right stocks within sectors), and interaction effects.

---

## Section IV: Behavioral Finance — Advanced (L3 Chs 15-21)

### Behavioral Biases — Applied (L3 Ch 15)
- **Cognitive vs. emotional biases**: Cognitive (systematic processing errors — can be corrected with awareness). Emotional (feeling-driven — harder to correct, must be managed with systems).
- **Counter-bias plans**: For each identified bias, formulate a specific mitigation. Example: Against overconfidence → mandate position size limits. Against loss aversion → use predefined stop-loss orders.
- **Capitalising on others' biases**: Market participants collectively exhibit biases. Momentum exists because of herding and anchoring. Reversals occur because of loss aversion reaching breaking point. Build strategies that exploit these.

### Investor Psychology (L3 Ch 16)
This chapter maps behavioral elements to EVERY phase of price action:
- **Pattern development**: Accumulation and distribution patterns form because of conflicting biases — some anchored to prior prices, others herding into new trends. The tug-of-war creates the range.
- **Trend persistence**: Trends continue because of herding (joining the trend), confirmation bias (seeking supporting evidence), and anchoring (using recent prices as the "normal" reference).
- **Consolidation**: Occurs when biases equilibrate — bulls and bears equally anchored. Volume declines as conviction fades. New information is needed to break the balance.
- **Trend reversal**: Happens when loss aversion reaches critical mass — enough holders "give up" simultaneously, creating capitulation (at bottoms) or profit-taking cascade (at tops). Overconfidence at extremes amplifies the reversal.

### Group Decision-Making (L3 Ch 17/21)
- Committee investment decisions are systematically WORSE than individual decisions due to: groupthink, social conformity, authority bias, diffusion of responsibility, information cascades.
- **Practical implication**: Consensus institutional positioning = high probability of being wrong at extremes. When "everyone agrees," the trade is crowded.
- **Mitigation**: (1) Independent analysis before group discussion, (2) designated devil's advocate, (3) anonymous voting, (4) external challenge from uncorrelated sources.

### Bubble Anatomy & De-Bubbling (L3 Chs 18-19)
- **Five stages** (Kindleberger/Minsky framework):
  1. **Displacement**: New technology, deregulation, or paradigm. Fundamentally sound initial price advance.
  2. **Boom**: Positive feedback loop begins. Early adopters profit. Media attention increases. New buyers enter.
  3. **Euphoria**: "This time is different." Leverage increases dramatically. Valuation metrics dismissed. Widespread public participation.
  4. **Profit-taking**: Smart money exits. Prices plateau. Volume shifts. Narrative maintained but cracks appear.
  5. **Panic**: Prices collapse. Leverage unwinds. Forced selling. Capitulation. New narrative = "obvious in hindsight."
- **Stage identification**: At any point, assess which stage the current market best resembles. The difficulty: stages 2-3 are hardest to distinguish in real-time (profit motive creates bias).
- **De-bubbling strategies**: Three cross-section approaches that benefit from deflationary/de-bubbling environments: (1) Quality (strong balance sheets outperform), (2) Value (cheap assets bought from forced sellers), (3) Low volatility (defensive positioning outperforms).
- **Alpha generation from de-bubbling**: Short overvalued sectors, long defensive/quality. Increase cash allocation in late stage 3. Deploy capital aggressively in stage 5 capitulation.

### Behavioral Techniques (L3 Ch 20/24)
- **Event reactions**: Planned news (earnings, FOMC, data releases) vs. price shocks (geopolitical, natural disasters). Planned events are partially priced in — the surprise component drives reaction. Price shocks are unpriced — full reaction.
- **Volatility ratio for event estimation**: Compare post-event range to pre-event range. High ratio = market was NOT positioned for the outcome. Low ratio = event was well-anticipated.
- **COT Index construction**: Normalise commercial/speculator positioning over a lookback (typically 26 or 52 weeks). COT Index = (Current net position - Minimum) / (Maximum - Minimum). Values >90% or <10% indicate extreme positioning — contrarian signals.

---

## Section V: Volatility Analysis — Advanced (L3 Chs 21-23)

### VIX as Stock Market Indicator (L3 Ch 21)
- VIX-SPX co-movement: Normally inverse. When both rise simultaneously = unusual stress. When both fall = unusual complacency.
- VIX futures basis: Futures premium to spot = contango (normal). Futures discount to spot = backwardation (fear exceeding forward expectations). Backwardation extremes = potential contrarian buy signal for equities.
- VIX in forecasting: Formulate forecasts that include not just direction but expected RANGE. Example: "SPX likely to rally, with VIX at 18 suggesting ±5.2% 30-day expected move. Upside target within one standard deviation is achievable."

### Hedging with VIX Derivatives (L3 Ch 22)
- **Rationale**: Traditional stops can gap through during crashes. VIX products provide convex protection — they accelerate in value precisely when portfolio losses accelerate.
- **VIX options**: Long calls for tail-risk hedging. Far OTM calls are cheap but only pay off in severe events. Closer-to-ATM calls are more expensive but provide broader protection.
- **VIX futures**: Long for hedging (but negative carry due to contango roll cost). Short for income (but unlimited risk if VIX spikes). Manage carefully.
- **Strategy construction**: Size VIX hedge to target a specific portfolio protection level. Example: 2% portfolio in VIX calls provides ~20-30% protection in a -10% equity drawdown (approximate — varies by strike/expiry/VIX level).

### Advanced Techniques (L3 Ch 23)
- **Volatility measures comparison**: HV (backward-looking), IV (forward-looking from options), realised vs. implied spread (volatility risk premium).
- **Volatility for profit targets and stops**: ATR-multiple targets/stops adapt to current market conditions automatically.
- **Volatility as a system filter**: Suppress signals during extremely low vol (false breakouts likely) or extremely high vol (whipsaws likely). Middle-vol environments tend to produce the cleanest signals.
- **Fractal analysis**: Markets exhibit self-similarity across timeframes. The Hurst exponent measures tendency to trend (H > 0.5) or mean-revert (H < 0.5). H = 0.5 indicates random walk.
- **Chaos theory**: Sensitive dependence on initial conditions. Markets are deterministic but chaotic — small changes in inputs produce large changes in outputs. Practical implication: long-term prediction is impossible, but short-term pattern recognition may have value.
- **Entropy**: Measure of disorder/randomness. Low entropy = orderly, predictable (trending). High entropy = disordered, unpredictable (ranging/chaotic). Shannon entropy applied to price distributions.
- **Neural networks**: Pattern recognition tools. Advantages: can identify non-linear relationships. Disadvantages: black box (difficult to interpret), prone to overfitting, require large training data. Use as a supplement to, not replacement for, traditional analysis.
- **Genetic algorithms**: Evolutionary optimisation for system parameters. Population of solutions → fitness evaluation → selection → crossover/mutation → new generation. Better than brute-force for large parameter spaces but can find local optima.

---

## Section VI: Classical Methods — Integration (L3 Chs 24-29)

### Pattern Recognition Advanced (L3 Ch 24)
- **Pivot points**: Traditional pivots from prior period OHLC. Provide calculated S/R levels. Pivot = (H + L + C) / 3. S1/R1, S2/R2, S3/R3 calculated from pivot.
- **DeMark calculations**: TD Lines, TD Points, TD Sequential for projected S/R and exhaustion counting. More complex than traditional pivots but adaptable to current price action.
- **Intraday idiosyncratic patterns**: Different markets have different intraday behaviors. Equities show opening/closing surges. FX shows session-transition patterns. Know the market's personality.
- **Opening gaps as signals**: Gap size, direction relative to prior close, and fill/no-fill behavior provide systematic trading signals.

### Multiple Timeframe Integration (L3 Ch 25)
- **Elder's Triple Screen**: (1) Weekly trend direction → determines bias, (2) Daily oscillator → finds pullback in weekly direction, (3) Intraday → pinpoints entry.
- **Krausz's Six Rules**: See Layer 1. The critical principle: higher timeframe ALWAYS overrules lower. A daily buy signal against a weekly downtrend is a counter-trend trade at best.
- **Pring's KST and Special K**: Multi-timeframe momentum synthesis. Combines multiple rates of change across different lookback periods into a single indicator.

### Progressive Candlestick Charting (L3 Chs 26-28)
- **Nine price action guidelines** (L3 Ch 26): Context-dependent rules for interpreting candle signals. Integrate with Western technical analysis for highest reliability.
- **Progressive charting** (L3 Ch 27): Read candles as they develop, not just after close. Anticipate completions to prepare orders. Evaluate: trend context, reversal signals, confirmation, risk parameters.
- **Real-world application** (L3 Ch 28): Propose specific entry/exit points from candle evidence combined with S/R, indicators, and risk management. Assess trend persistence vs. reversal probability. The four questions: What is the trend? Is there a reversal signal? Is there confirmation? What are the risk parameters?

### Conclusions — The 12 Major Conclusions (L3 Ch 29)
- Technical indicators work when properly employed across market environments.
- ARIMA forecasts generate tradeable signals when combined with other methods.
- Linear regression produces trend signals and RS signals.
- Integration is key — no single method is sufficient. The full CMT body of knowledge is designed to be used AS A SYSTEM, not as isolated tools.
