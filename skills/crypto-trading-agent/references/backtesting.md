# Strategy Backtesting & Optimisation Reference

## Table of Contents
1. Backtesting Philosophy
2. Data Preparation
3. Strategy Codification
4. Testing Methodology
5. Performance Metrics
6. Optimisation Techniques
7. Walk-Forward Analysis
8. Common Pitfalls

---

## 1. Backtesting Philosophy

A backtest is a hypothesis test, not a fortune-telling exercise. The goal is to answer:
"Does this strategy have a statistically meaningful edge, and is that edge robust across
different market conditions?"

### What a Backtest Can Tell You
- Whether the core logic has historically produced positive expected value
- How the strategy behaves in different regimes (bull, bear, range)
- Maximum historical drawdown and recovery time
- Whether the strategy's edge degrades over time

### What a Backtest Cannot Tell You
- Whether the strategy will work in the future
- How real-world slippage, fees, and funding will affect results
- How you'll behave psychologically when the strategy is in a drawdown
- Whether market microstructure has changed since the test period

---

## 2. Data Preparation

### Data Sources for Bybit Backtesting

**Kline/OHLCV data**: Use the Bybit V5 API `/v5/market/kline` endpoint.
Intervals available: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720 (minutes), D, W, M.

**Minimum data requirements by strategy type**:
- Scalping (1-5M): Minimum 3 months of data, ideally 6+
- Intraday (15M-1H): Minimum 6 months, ideally 12+
- Swing (4H-Daily): Minimum 12 months, ideally 24+
- Position (Daily-Weekly): Minimum 24 months, ideally covering a full bull/bear cycle (36-48)

### Data Cleaning Checklist
- [ ] Remove duplicate timestamps
- [ ] Handle missing candles (especially during maintenance/outages)
- [ ] Verify OHLC relationships (High ≥ Open, Close; Low ≤ Open, Close)
- [ ] Check for zero-volume candles (may indicate exchange issues)
- [ ] Align timestamps to UTC
- [ ] Account for any pair migrations or relistings

### Synthetic Data Considerations
For pairs with limited history, consider:
- Testing the logic on BTC or ETH (longer history) then applying to the target pair
- Using correlated pairs as proxies
- Never backtesting on synthetic/interpolated data

---

## 3. Strategy Codification

### From Discretionary to Systematic

Every strategy component must be converted to unambiguous rules. For SMC-based strategies:

**Market structure identification**:
```python
# Example: Swing point detection
def find_swing_highs(df, lookback=5):
    """A swing high is a candle whose high is higher than 
    the highs of the N candles on either side."""
    swing_highs = []
    for i in range(lookback, len(df) - lookback):
        if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback+1].max():
            swing_highs.append({
                'index': i,
                'timestamp': df['timestamp'].iloc[i],
                'price': df['high'].iloc[i]
            })
    return swing_highs
```

**Order block detection**:
```python
def find_bullish_ob(df, displacement_multiplier=2.0):
    """Find bullish order blocks: last bearish candle before 
    a bullish displacement."""
    avg_range = (df['high'] - df['low']).rolling(20).mean()
    obs = []
    for i in range(1, len(df) - 1):
        # Current candle is bearish
        if df['close'].iloc[i] < df['open'].iloc[i]:
            # Next candle is a bullish displacement
            next_range = df['high'].iloc[i+1] - df['low'].iloc[i+1]
            if (df['close'].iloc[i+1] > df['open'].iloc[i+1] and 
                next_range > avg_range.iloc[i] * displacement_multiplier):
                obs.append({
                    'index': i,
                    'timestamp': df['timestamp'].iloc[i],
                    'high': df['high'].iloc[i],  # OB top
                    'low': df['low'].iloc[i],      # OB bottom
                    'body_high': df['open'].iloc[i],  # Refined OB
                    'body_low': df['close'].iloc[i],
                    'mitigated': False
                })
    return obs
```

**FVG detection**:
```python
def find_bullish_fvg(df):
    """Bullish FVG: candle[i-1].high < candle[i+1].low"""
    fvgs = []
    for i in range(1, len(df) - 1):
        gap_bottom = df['high'].iloc[i-1]
        gap_top = df['low'].iloc[i+1]
        if gap_bottom < gap_top:
            fvgs.append({
                'index': i,
                'timestamp': df['timestamp'].iloc[i],
                'top': gap_top,
                'bottom': gap_bottom,
                'ce': (gap_top + gap_bottom) / 2,  # Consequent encroachment
                'filled': False
            })
    return fvgs
```

---

## 4. Testing Methodology

### Train/Test Split

**Minimum split**: 70% in-sample (training) / 30% out-of-sample (testing)
**Preferred split**: 60% in-sample / 20% validation / 20% out-of-sample

The out-of-sample data must NEVER be used for parameter tuning. It's your final exam.

### Regime-Aware Testing

Split your data by market regime and test each separately:

1. **Bull trending**: Identify periods where price was above 200 EMA with HH/HL structure
2. **Bear trending**: Price below 200 EMA with LH/LL structure
3. **Ranging/Choppy**: Periods where price crossed the 200 EMA multiple times
4. **High volatility events**: Major moves (>10% in 24h), liquidation cascades

A good strategy doesn't need to be profitable in all regimes. But you need to know which
regimes it fails in so you can filter trades or adjust sizing accordingly.

### Transaction Cost Modelling

For Bybit perpetual futures:
- **Maker fee**: 0.02% (limit orders)
- **Taker fee**: 0.055% (market orders)
- **Funding rate**: Variable, typically 0.01% per 8h (can spike to 0.1%+ in extreme conditions)
- **Slippage estimate**: 0.02-0.05% for major pairs (BTC, ETH), 0.05-0.2% for mid-caps

Include round-trip costs (entry + exit) in every backtest trade:
```python
total_cost_per_trade = entry_fee + exit_fee + (funding_rate * holding_periods * 3)
# For a taker entry, maker exit, held for 2 days:
# 0.055% + 0.02% + (0.01% * 2 * 3) = 0.135%
```

---

## 5. Performance Metrics

### Required Metrics (Report All of These)

| Metric | Formula / Description | Acceptable Range |
|--------|----------------------|-----------------|
| Net Profit | Total P&L after costs | Positive, obviously |
| Win Rate | Winning trades / Total trades | 35-55% for trend-following, 55-70% for mean-reversion |
| Average R:R | Average winner size / Average loser size | > 1.5 for trend-following |
| Profit Factor | Gross profit / Gross loss | > 1.5 (below 1.3 is marginal) |
| Sharpe Ratio | (Mean return - Risk-free) / StdDev of returns | > 1.0 (annualised). > 2.0 is excellent. |
| Sortino Ratio | (Mean return - Risk-free) / Downside deviation | > 1.5. Better than Sharpe for asymmetric strategies. |
| Max Drawdown | Largest peak-to-trough decline | < 20% for conservative, < 30% for aggressive |
| Max Drawdown Duration | Longest time from peak to recovery | Context-dependent but flag if > 3 months |
| Calmar Ratio | Annualised return / Max drawdown | > 1.0 is good, > 2.0 is excellent |
| Trade Frequency | Trades per week/month | Must be sufficient for statistical significance |
| Expectancy | (Win% × Avg Win) - (Loss% × Avg Loss) | Positive with confidence interval |

### Statistical Significance

A backtest is meaningless without sufficient trades:
- Minimum 30 trades for any directional conclusion
- Minimum 100 trades for reliable metric estimation
- Minimum 200+ trades for confidence in the strategy's edge

Calculate the t-statistic for the strategy's mean return:
```
t = (mean_return × sqrt(n_trades)) / std_return
```
If t > 2.0 (approximately p < 0.05), the edge is statistically significant.

---

## 6. Optimisation Techniques

### Parameter Sensitivity Analysis

Before optimising, understand how sensitive the strategy is to each parameter:

1. Take each parameter in isolation
2. Vary it across a range while holding others constant
3. Plot the performance metric (Sharpe, profit factor) against the parameter value
4. Look for a "plateau" — a range of values that all produce similar results

A robust strategy has wide plateaus. A fragile strategy has sharp peaks — small parameter
changes cause large performance swings. This is a major overfitting red flag.

### Grid Search (Simple)

Test all combinations within defined ranges:
```python
params = {
    'ob_displacement_mult': [1.5, 2.0, 2.5, 3.0],
    'swing_lookback': [3, 5, 7, 10],
    'risk_per_trade': [0.005, 0.01, 0.015, 0.02],
    'rr_minimum': [1.5, 2.0, 2.5, 3.0]
}
# Total combinations: 4 × 4 × 4 × 4 = 256 backtests
```

### Walk-Forward Optimisation (Preferred)

See Section 7 below. This is the gold standard for avoiding overfitting.

### What to Optimise vs What to Fix

**Fix these** (based on market structure logic, not optimisation):
- Market structure rules (BOS/CHoCH definitions)
- Risk per trade (this is a personal/account decision)
- Maximum leverage

**Optimise these** (parameters that may vary by market/timeframe):
- Displacement multiplier for OB detection
- Swing lookback period
- Minimum R:R threshold
- FVG fill percentage for entry trigger
- EMA periods for dynamic S/R

---

## 7. Walk-Forward Analysis

### Why Walk-Forward?

Walk-forward analysis simulates how the strategy would perform in real-time by:
1. Optimising on a window of historical data
2. Testing on the next unseen window
3. Sliding the window forward and repeating

This prevents curve-fitting because the test data is always truly out-of-sample.

### Walk-Forward Process

```
Total data: Jan 2023 — Dec 2025

Step 1: Optimise on Jan 2023 - Jun 2024 → Test on Jul 2024 - Sep 2024
Step 2: Optimise on Apr 2023 - Sep 2024 → Test on Oct 2024 - Dec 2024
Step 3: Optimise on Jul 2023 - Dec 2024 → Test on Jan 2025 - Mar 2025
Step 4: Optimise on Oct 2023 - Mar 2025 → Test on Apr 2025 - Jun 2025
...

In-sample window: 18 months (rolling)
Out-of-sample window: 3 months
Overlap: 50% (window slides by half the IS period)
```

### Walk-Forward Efficiency (WFE)

```
WFE = Out-of-Sample Performance / In-Sample Performance

WFE > 0.5 (50%): Strategy retains meaningful edge out of sample — good
WFE 0.3-0.5: Edge is present but degraded — proceed with caution
WFE < 0.3: Likely overfit — re-examine strategy logic
```

---

## 8. Common Pitfalls

### Overfitting Indicators
- Backtest Sharpe > 3.0 (suspiciously good)
- More than 5 tuneable parameters
- Performance collapses with small parameter changes
- Works perfectly on one pair/timeframe but fails on others
- WFE below 0.3

### Survivorship Bias
In crypto, many tokens have been delisted, rugged, or gone to zero. If you're testing on
tokens that "survived" to today, your results are biased upward. Always include delisted
and failed tokens in your universe if testing a token-selection strategy.

### Look-Ahead Bias
Ensure no future data leaks into historical decisions:
- Use `shift(1)` or equivalent to ensure signals are based on closed candles only
- Open interest and funding rate data should be lagged by at least one period
- Don't use high/low of the current candle for entry decisions (you don't know the
  high/low until the candle closes)

### Liquidity Assumptions
A strategy that trades $50K positions on a mid-cap alt might look great in a backtest
but fail in practice because the orderbook couldn't absorb that size without 2% slippage.
Always check the average daily volume of your trading pair against your position size.
Rule of thumb: your position should be < 1% of the average hourly volume.
