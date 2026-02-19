# Level II Capabilities Reference — The Theory & Analysis of Technical Analysis

Source: CMT Level II 2022 Curriculum (39 chapters, 6 sections)

L2 elevates L1 foundational knowledge into deeper analytical methods. Every L1 concept gets advanced application, plus new domains: Market Profile, applied cycles, advanced candlestick forecasting, prospect theory, correlation/regression/ARIMA, intermarket analysis, market models, scientific method, and backtesting.

---

## Section I: Chart Development & Analysis (L2 Chs 1-16)

### Market Profile (L2 Ch 2)
- Time-Price Opportunity (TPO) charts: Display how much TIME was spent at each price level.
- Point of Control (POC): Price with highest TPO count — "fair value" where most trading occurred.
- Value Area: Price range containing 70% of TPOs. Value Area High (VAH) and Value Area Low (VAL) act as S/R.
- Single prints: Prices visited but not revisited — indicate fast directional moves. Act as S/R on retest.
- Poor highs/lows: Minimal TPOs at session extremes — suggest unfinished business, likely to be retested.
- Balance vs. imbalance: Symmetrical bell-shaped profile = balance (range). Elongated/skewed profile = imbalance (trending).
- Opening types: Open-Drive (strong conviction), Open-Test-Drive (tests one side then drives), Open-Rejection-Reverse (tests and fails), Open-Auction.
- Use Market Profile for: determining fair value vs. unfair pricing, identifying institutional activity zones, and timing entries at value area boundaries.

### Advanced Trend Systems (L2 Chs 3-6)
- **MA selection framework**: Match MA type (SMA/EMA/WMA) and period to the trend you're trading. Short-term trader = faster MA, longer-term investor = slower MA. No single "best" MA — it depends on the objective.
- **Drop-off effect**: When a large price leaves the MA window, it can cause spurious signals. Be aware of what's exiting the lookback, especially on SMA.
- **Two-trend and three-trend systems**: Dual MA (e.g., 50/200) for trend identification. Triple MA (e.g., 10/50/200) for regime and timing. Exit rules: close below shorter MA (faster exit) vs. short MA crosses long MA (slower, avoids whipsaws).
- **Trend system combinations**: Bollinger Bands + Keltner Channels for squeeze detection. Percentage bands for fixed-threshold systems. Volatility bands (ATR-based) for adaptive systems. Combine band signals with momentum confirmation.
- **10-day MA rule**: Systematic trend-following using price position relative to 10-day MA. Simple, effective, testable.
- **Three reasons trend systems work**: (1) Trends persist due to behavioral biases, (2) trend-following captures the fat right tail of return distributions, (3) systematic execution removes emotional interference.

### Four-Phase Price-Volume Model (L2 Ch 8)
The operational framework for volume interpretation:
1. **Accumulation**: Price flat/basing. Volume shifting from selling to neutral/buying. Smart money entering. OBV diverging positively from price.
2. **Markup**: Price rising. Volume expanding on advances, contracting on pullbacks. Public participation increasing. OBV confirming trend.
3. **Distribution**: Price flat at highs. Volume shifting from buying to selling. Smart money exiting. OBV diverging negatively from price.
4. **Markdown**: Price falling. Volume expanding on declines. Public selling/panic. OBV confirming downtrend.

Phase identification drives strategy selection: trade with trend in markup/markdown, trade range in accumulation/distribution.

### Advanced Breadth (L2 Ch 9)
- **Arms Index (TRIN)**: (Adv/Dec) / (Adv Vol/Dec Vol). TRIN <1.0 = bullish (advancing volume dominant). TRIN >1.0 = bearish. Extreme readings (<0.5 or >2.0) = contrarian signals.
- **Thrust Oscillator**: Measures breadth impulse. Strong thrust from oversold conditions = high-probability rally signal. Used as a filter — only take buy signals after thrust confirms.
- Incorporate breadth and volume into systematic methods: Breadth as a qualifying filter for trend signals (only take trend signals when breadth confirms).

### Applied Cycle Analysis (L2 Chs 15-16)
- **Mid-cycle dip**: Predictable pullback at approximately the cycle midpoint. Buying opportunity within a bullish cycle.
- **3/4-cycle high**: Often the final significant high before cycle-driven decline.
- **Cycle inversion**: Peak occurs where trough is expected (or vice versa). Powerful reversal signal — indicates the dominant force has shifted.
- **Rounded tops and V-bottoms**: Tops round because cycle peaks disperse across time. Bottoms are sharp because cycle troughs synchronise (Hurst's synchronicity principle).
- **CMA envelope**: Centered moving average eliminates phase lag. Creates a channel that reveals the underlying cycle clearly.
- **Valid Trend Line (VTL)**: Connects cycle troughs. Break of VTL = potential cycle direction change. Analogous to breaking a trendline in price, but applied to the cycle.
- **Spectrogram**: Frequency-domain visualisation. Identifies the dominant cycle period. The dominant cycle is the one with highest amplitude — this is the cycle most likely to influence near-term price action.
- **Harmonic phasing**: Smaller cycles relate to larger ones by integer ratios (usually 2:1). The 20-week cycle contains two ~10-week cycles, which contain four ~5-week cycles. Use this hierarchy for multi-timeframe cycle alignment.

### Advanced Candlestick Analysis (L2 Chs 12-14, L3 Chs 26-28)
- **Forecasting and trading techniques** (L2 Ch 14): Combine candle patterns with Western chart analysis (S/R, trendlines, indicators) for higher-probability signals. Use candles for risk management — a reversal candle at a key level = tighten stop or take profit. Multi-timeframe candle analysis: weekly candle pattern overrides daily.
- **Progressive charting** (L3 Ch 27): Evaluate candle patterns AS they develop, not just after close. Anticipate pattern completion to prepare orders. Four questions to answer at every candle: (1) What is the trend? (2) Is there a reversal signal? (3) Is there confirmation? (4) What are the risk parameters?
- **Nine price action guidelines** (L3 Ch 26): Essential rules for candlestick interpretation in context of broader technical environment.
- **Real-world integration** (L3 Ch 28): Propose entry/exit points based on candle patterns, price action, and risk together. Assess trend persistence from candle evidence.

---

## Section II: Volatility (L2 Chs 17-19)

### Options for the Technical Analyst
- Options as a window into market expectations, not just a trading instrument
- Greeks as sensitivity measures: Delta (directional exposure), Gamma (rate of change of delta), Theta (time decay), Vega (volatility sensitivity)
- IV as the market's forecast of future volatility. Calculate single-day IV = Annual IV / √252 (equities) or √365 (crypto).
- Inputs to option pricing model: underlying price, strike, time to expiry, risk-free rate, IV. The ONLY unknown is IV — everything else is observable.
- HV vs. IV: HV tells you what happened. IV tells you what the market expects. When IV > HV significantly, options are expensive (good for selling). When IV < HV, options are cheap (good for buying).

### VIX Deep Dive
- VIX is impacted by put-call parity and options supply/demand — not just fear
- VIX as sentiment: Rising VIX with falling SPX = increasing fear. Falling VIX with rising SPX = complacency building. VIX diverging from SPX = potential regime change.
- VIX in forecasts: Include expected volatility as a variable, not just direction. High VIX environments favour mean-reversion strategies. Low VIX environments favour trend-following.
- Expected move calculation: 30-day = SPX × VIX/100 / √12. One standard deviation expected range.

---

## Section III: Behavioral Finance (L2 Chs 20-24)

### Prospect Theory (L2 Ch 20)
- Utility theory assumes rational actors. Prospect theory shows actual behavior deviates systematically.
- **Loss aversion**: Losses are felt ~2x more than equivalent gains. Creates: premature profit-taking, holding losers too long, excessive overhead supply at prior highs.
- **Reference dependence**: Outcomes evaluated relative to a reference point (usually purchase price), not absolute wealth. This is why support at cost basis is so persistent.
- **Probability weighting**: People overweight small probabilities (lottery effect → buying far OTM options) and underweight moderate-to-high probabilities.
- **Greatest limitation**: Descriptive, not predictive. Explains WHY biases exist but not WHEN they'll dominate.

### Perception Biases (L2 Ch 21)
Four distortions that affect market participants:
1. **Availability**: Overweighting recent or vivid events. After a crash, investors overestimate crash probability. After a rally, they underestimate risk.
2. **Representativeness**: Pattern-matching from insufficient data. "This looks like 2008" based on superficial similarity.
3. **Anchoring**: Fixating on reference points. Prior all-time high becomes a magnet. 52-week high/low anchors valuation.
4. **Overconfidence**: Overestimating accuracy of predictions. Manifests as: excessive position sizing, inadequate diversification, ignoring contradicting evidence.

### Inertial Effects (L2 Ch 22)
Three forces that sustain existing behavior:
1. **Status quo bias**: Preference for current state. Investors stay in losing positions partly because changing requires effort and decision.
2. **Endowment effect**: Overvaluing what you own. Creates reluctance to sell, supporting trend persistence.
3. **Regret aversion**: Fear of making a decision that proves wrong. Creates inaction at critical junctures.

### Sentiment from Derivatives (L2 Ch 24)
- Futures OI interpretation: Rising OI + rising price = new longs (bullish). Rising OI + falling price = new shorts (bearish). Falling OI + rising price = short covering (less bullish). Falling OI + falling price = long liquidation (less bearish).
- COT report analysis: Commercial (hedger) positioning as the "smart money" signal at extremes. Large speculator positioning as a trend indicator. Small speculator positioning as a contrarian indicator at extremes.
- Put/call ratio: Use both equity-only and total (including index). Equity P/C more reflective of retail sentiment. Index P/C more reflective of institutional hedging.
- Volatility data from options: IV skew, term structure shape, and RV-IV spread as sentiment indicators.

---

## Section IV: Statistical Applications (L2 Chs 25-28)

### Inferential Statistics (L2 Ch 25)
- Descriptive → Inferential: Moving from "what happened" to "what can we conclude."
- Hypothesis testing framework: State null hypothesis → choose significance level → compute test statistic → compare to critical value → conclude.
- **Base rate fallacy**: Don't confuse P(signal | true) with P(true | signal). A rare signal from a mediocre system can appear highly significant.
- Confidence intervals: Express uncertainty honestly. A system with 55% ± 8% win rate might not be profitable.

### Correlation (L2 Ch 26)
- **Pearson's r**: Measures linear correlation. Requires: (1) linearity, (2) approximate normality, (3) no extreme outliers. Range: -1 to +1.
- **Spearman's rho**: Rank-based correlation. Handles non-linear monotonic relationships. More robust to outliers.
- Always visualise with scatterplot before computing correlation. Anscombe's quartet demonstrates why.
- **r² (coefficient of determination)**: Proportion of variance explained. r = 0.7 means r² = 0.49, so only 49% of variation is explained. Market relationships rarely exceed r = 0.5 reliably.

### Regression (L2 Chs 27-28)
- **Linear regression**: Y = a + bX. Use for: trend quantification (regression channel), relationship estimation, signal generation (price deviating from regression = overbought/oversold).
- **Multiple regression**: Y = a + b₁X₁ + b₂X₂ + ... Used for multi-factor models. Check for multicollinearity (tolerance calculations).
- **Predictor variable selection**: Start with theory-driven candidates. Avoid kitchen-sink regression. More variables ≠ better model.
- **ARIMA**: Autoregressive Integrated Moving Average. Captures trend (I component), mean-reversion (AR component), and smoothing (MA component). Residuals from ARIMA model = trading signals: positive residual = price above model expectation (overbought), negative = below (oversold).
- **Regression for RS**: Apply linear regression to relative strength line. Slope quantifies rate of outperformance. Acceleration/deceleration detectable from regression curvature.

---

## Section V: Methods and Market Selection (L2 Chs 29-34)

### Market and Issue Selection (L2 Ch 29)
- **Trading style hierarchy**: Buy-and-hold → position trading → swing trading → day trading. Technical analysis applies differently at each level. Shorter timeframes = more noise, require tighter risk management.
- **Top-down vs. bottom-up**: Top-down starts with macro (economy → sector → stock). Bottom-up starts with individual charts. Integrated approach uses both.
- **Secular vs. cyclical**: Secular trends (10-30 years) provide the backdrop. Cyclical moves (1-5 years) within seculars provide the trades. Don't fight the secular trend.

### Intermarket Analysis (L2 Ch 30)
- Business cycle rotation: Bonds lead → stocks follow → commodities lag. Bond prices peak at recession trough, stock prices bottom mid-recession, commodities bottom in early expansion.
- Correlation for diversification: True diversification requires uncorrelated assets. Correlation matrices change over time — must be monitored dynamically.
- Asset allocation from intermarket: Overweight assets in favorable cycle position, underweight those in unfavorable position.

### Relative Strength Strategies (L2 Ch 31)
- Four methods of calculating RS: (1) ratio of prices, (2) ratio of returns, (3) alpha (excess return over benchmark), (4) information ratio
- Momentum strategy: Buy relative strength leaders, sell laggards. Rebalance periodically.
- Hedging in RS models: Use non-correlated assets to reduce systematic risk while maintaining RS alpha.
- Long-only RS model: Overweight leaders, equal-weight or underweight laggards. Simpler implementation for most investors.

### Market Models (L2 Chs 32-33)
- **Ned Davis' Fab Five**: Composite model combining internal and external indicators. Internal = market-derived (breadth, momentum). External = economic-derived (Fed policy, interest rates, sentiment).
- **Zweig's model**: Four indicators originally, modified to five. Combines trend-following internal signals with mean-reverting external signals. Each indicator classified as internal/external AND trend-following/mean-reverting.
- **Bond model**: Separate from equity model. Yield curve, credit spreads, and monetary policy as primary inputs.
- **Environmental model**: First determine the TYPE of market (trending/ranging, risk-on/risk-off), THEN select the appropriate strategy. Don't apply trend-following in a range or mean-reversion in a trend.

### Scientific Method & EMH (L2 Chs 34, 36-37)
- **Alpha vs. beta**: Beta = market return (free). Alpha = skill-based excess return (earned). Technical analysis claims to generate alpha through timing and selection.
- **EMH three forms revisited**: Weak (prices reflect past prices — TA shouldn't work). Semi-strong (all public info). Strong (all info including insider). Behavioral finance challenges all three. Adaptive Markets Hypothesis resolves the paradox: efficiency varies over time and across markets.
- **Nonrandom price motion** (L2 Ch 37): The foundational premise of TA. Evidence: momentum effect, mean-reversion at extremes, calendar anomalies, feedback loops. Two foundations of behavioral finance: (1) limits to arbitrage and (2) systematic investor irrationality.
- **Scientific method application** (L2 Ch 36): Observation → hypothesis → prediction → test → conclusion. Null hypothesis framing. Three consequences of adopting scientific method in TA.

---

## Section VI: System Design and Testing (L2 Chs 35, 38-39)

### Backtesting Statistics (L2 Ch 35)
- Four features of time-series price data that create challenges: non-stationarity, autocorrelation, heteroskedasticity, fat tails (leptokurtosis).
- Use log returns for proper statistical treatment (additive, closer to normal distribution).
- Three statistical concerns: (1) data-mining bias (enough variables tested, some will appear significant by chance), (2) look-ahead bias (using information not available at the time), (3) survivorship bias (only testing assets that still exist).
- Signal testing vs. backtesting: Signal testing asks "does this indicator predict direction?" Backtesting asks "does this strategy make money?" Signal testing should always precede backtesting.

### Data Mining Bias (L2 Ch 38)
- Data-mining bias: Testing hundreds of rules on the same data will produce some that appear profitable purely by chance.
- Data-snooping bias: Related — reusing data for development and validation. Out-of-sample testing is the only remedy.
- Apply Bonferroni correction or similar adjustments for multiple comparisons.
- A rule's significance should be judged not just by its in-sample performance but by how many rules were tested to find it.

### System Design (L2 Ch 39)
- Five initial decisions: (1) what market, (2) what timeframe, (3) entry rules, (4) exit rules, (5) money management.
- Four system types: trend-following, mean-reversion, pattern-based, composite.
- Discretionary vs. nondiscretionary trade-offs: Discretionary allows adaptation but introduces bias. Nondiscretionary removes emotion but requires robust design.
- Evaluation metrics: Profit factor (gross profit / gross loss), percent profitable, average trade net profit.
- Optimisation methods: Brute force, genetic algorithms. Define "robustness" = acceptable performance across parameter range, not just at optimal point.
- Risk-adjusted metrics: Sharpe, Sterling, Sortino ratios for comparing systems on a risk-adjusted basis.
