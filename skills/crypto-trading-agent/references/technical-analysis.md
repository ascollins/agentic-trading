# Technical Analysis — Indicators, Patterns & Multi-Timeframe Framework

## Table of Contents
1. Indicator Hierarchy
2. Moving Averages & Dynamic S/R
3. Momentum Indicators
4. Volume Analysis
5. Fibonacci Framework
6. Candlestick Patterns at Key Levels
7. Multi-Timeframe Indicator Alignment

---

## 1. Indicator Hierarchy

Indicators are confirmation tools, not primary signals. In this framework, market structure
and SMC analysis provide the directional thesis; indicators confirm or deny it.

**Tier 1 — Structure Tools** (always use):
- EMAs (21/50/200) for trend identification and dynamic S/R
- Volume (raw and profiled) for conviction assessment
- ATR for volatility context and stop placement

**Tier 2 — Momentum Confirmation** (use to confirm setups):
- RSI (14-period) for divergence and oversold/overbought
- MACD (12/26/9) for momentum direction and crossovers
- Stochastic RSI for LTF entry timing in OB/FVG zones

**Tier 3 — Supplementary** (use selectively):
- Bollinger Bands for volatility compression/expansion detection
- VWAP (session and anchored) for institutional fair value
- ADX for trend strength quantification
- Ichimoku cloud for HTF trend context (Weekly/Daily only)

---

## 2. Moving Averages & Dynamic Support/Resistance

### The EMA Framework

| EMA | Role | Interpretation |
|-----|------|---------------|
| 21 EMA | Short-term trend | In a strong trend, price pulls back to 21 EMA and bounces. If price is below 21 EMA, short-term momentum has shifted. |
| 50 EMA | Medium-term trend | The "value zone." In healthy trends, pullbacks to 50 EMA are high-probability entries. Acts as dynamic OB. |
| 200 EMA | Long-term trend | The macro trend filter. Above 200 EMA = bullish environment. Below = bearish. Rarely a precision entry, but always a bias filter. |

### EMA Alignment States

**Bull power stack** (21 > 50 > 200, all sloping up):
- Highest probability for long entries
- Pullbacks to 21 or 50 EMA are buyable
- Only look for shorts if HTF structure breaks

**Bear power stack** (21 < 50 < 200, all sloping down):
- Highest probability for short entries
- Rallies to 21 or 50 EMA are sellable
- Only look for longs if HTF structure breaks

**Compression** (EMAs converging, flattening):
- Regime transition likely
- Reduce position sizes
- Wait for EMA expansion to signal new direction

**Twisted** (EMAs interleaved, no clear order):
- Ranging/choppy market
- No directional trades — range-play only or sit out

### 200 EMA as a Macro Filter

For crypto specifically, the 200 EMA on the Daily chart is a widely watched level.
When BTC is above its Daily 200 EMA, the entire crypto market tends toward risk-on behaviour.
Below it, risk-off. Use this as a portfolio-level filter:
- Above Daily 200 EMA: full position sizing, higher leverage tolerance
- Below Daily 200 EMA: reduced sizing, lower leverage, tighter stops

---

## 3. Momentum Indicators

### RSI (Relative Strength Index)

**Primary use**: Divergence detection, not oversold/overbought signals alone.

**Bullish divergence**: Price makes a lower low, RSI makes a higher low.
Indicates selling momentum is weakening. Highest value when it occurs at:
- An HTF demand zone or OB
- After a liquidity sweep (SSL sweep)
- In an area with confluence from FVGs

**Bearish divergence**: Price makes a higher high, RSI makes a lower high.
Indicates buying momentum is weakening. Highest value at supply zones after BSL sweeps.

**Hidden divergence** (trend continuation):
- Bullish hidden: Price makes HL, RSI makes LL → trend continuation long signal
- Bearish hidden: Price makes LH, RSI makes HH → trend continuation short signal

**RSI ranges by regime**:
- Trending bullish: RSI typically oscillates between 40-80. Dips to 40-50 = pullback entries.
- Trending bearish: RSI typically oscillates between 20-60. Rallies to 50-60 = short entries.
- Ranging: RSI oscillates full range 30-70. Classic overbought/oversold applies here.

### MACD

Use MACD histogram for momentum assessment, not as a standalone signal:
- Expanding histogram in the direction of the trend = momentum confirmation
- Shrinking histogram = momentum fading, prepare for pullback or reversal
- Histogram divergence from price = early warning of regime transition

### Stochastic RSI

Best used on LTF (15M/5M) for entry timing within a confirmed setup:
- Stoch RSI < 20 while price is in a bullish OB/FVG zone = precision long entry
- Stoch RSI > 80 while price is in a bearish OB/FVG zone = precision short entry
- Crossovers only matter when they occur at these key structural levels

---

## 4. Volume Analysis

### Raw Volume Interpretation

| Price Action | Volume | Interpretation |
|-------------|--------|---------------|
| Impulse up | High | Genuine buying — trend likely to continue |
| Impulse up | Low | Weak rally — likely to be retraced |
| Pullback down | Low | Healthy correction — sellers not committed, dip buyable |
| Pullback down | High | Concerning — potential distribution, watch for CHoCH |
| Breakout | Very high (3x avg) | Valid breakout — momentum entry possible |
| Breakout | Average or below | Suspect breakout — likely inducement/trap |
| Climactic spike | Extreme (5x+ avg) | Exhaustion — potential reversal zone |

### Volume Profile

Volume profile shows the distribution of volume at price levels over a given period.
Key concepts:

- **Point of Control (POC)**: The price level with the highest traded volume. Acts as a
  magnet for price and a key support/resistance level.
- **Value Area High (VAH)**: The upper boundary of the 70% volume zone.
- **Value Area Low (VAL)**: The lower boundary of the 70% volume zone.
- **High Volume Node (HVN)**: Price levels with concentrated volume — act as support/resistance.
- **Low Volume Node (LVN)**: Price levels with thin volume — price moves quickly through these.
  LVNs often coincide with FVGs.

### VWAP (Volume Weighted Average Price)

VWAP represents the "fair price" based on volume-weighted transactions:
- **Session VWAP**: Resets daily. Institutional traders use this to benchmark execution quality.
  Price above VWAP = buyers in control that session. Below = sellers.
- **Anchored VWAP**: Anchored to a significant event (swing low, BOS, news event). Shows the
  average price of all participants since that event. Useful for identifying trapped traders.

---

## 5. Fibonacci Framework

### Retracement Levels for Entries

| Level | Name | Usage |
|-------|------|-------|
| 0.382 | Shallow | Strong trends — price barely pulls back. Entry in impulse phase. |
| 0.500 | Equilibrium | The 50% level. Where discounted and premium price zones meet. |
| 0.618 | Golden ratio | The most commonly respected Fib level. Often coincides with OBs. |
| 0.650 | — | Part of the "golden pocket" (0.618-0.65). Strong entry zone. |
| 0.705 | OTE start | Optimal Trade Entry zone begins. |
| 0.786 | OTE end | Deepest "healthy" retracement. Beyond this, the move may be failing. |

### The Optimal Trade Entry (OTE) Zone: 0.705 - 0.786

The OTE zone represents the deepest pullback that's still consistent with a continuation.
When price retraces into this zone AND there's an OB or FVG present, it's the highest
probability entry in the SMC framework.

**How to use**:
1. Identify the impulse leg (swing low to swing high for longs)
2. Draw Fibonacci retracement from the swing low to swing high
3. The 0.705 - 0.786 zone is your OTE
4. If an OB or FVG exists within this zone, that's your precision entry
5. Stop loss goes below the swing low (full invalidation)

### Extension Levels for Targets

| Level | Usage |
|-------|-------|
| -0.272 | First extension target. Conservative TP1. |
| -0.618 | Standard full target. Good TP2. |
| -1.000 | Measured move (1:1 extension). Aggressive TP3. |
| -1.618 | Extended target. Only in strong trending conditions. |

---

## 6. Candlestick Patterns at Key Levels

Candlestick patterns only matter when they form at structurally significant levels (OBs,
FVGs, supply/demand zones, key Fib levels). A hammer in the middle of a range means nothing.
A hammer at an HTF demand zone after a liquidity sweep is a powerful signal.

### High-Probability Patterns at SMC Levels

**Bullish patterns** (at demand zones / OBs / after SSL sweep):
- **Bullish engulfing**: Body fully engulfs prior candle's body. The larger the engulfing
  candle relative to surrounding candles, the stronger the signal.
- **Hammer / Pin bar**: Long lower wick (2x+ body length) showing rejection of lower prices.
  The wick should ideally sweep into the OB/FVG zone.
- **Morning star**: Three-candle reversal at a demand zone with increasing volume.

**Bearish patterns** (at supply zones / OBs / after BSL sweep):
- **Bearish engulfing**: Mirror of bullish engulfing at supply.
- **Shooting star / Inverted pin bar**: Long upper wick showing rejection of higher prices.
- **Evening star**: Three-candle reversal at supply with increasing volume.

**Continuation patterns** (during pullbacks in trends):
- **Inside bar**: Contraction candle within prior candle's range. At an OB during a pullback,
  an inside bar followed by a break in the trend direction is a low-risk entry.
- **Doji at support/resistance**: Indecision followed by directional break.

---

## 7. Multi-Timeframe Indicator Alignment

### Indicator Alignment Matrix

Before entering any trade, check indicator alignment across timeframes:

```
INDICATOR ALIGNMENT CHECK — [PAIR]

                    HTF         MTF         LTF
EMA Stack:          [Bull/Bear] [Bull/Bear] [Bull/Bear]
RSI Position:       [XX]        [XX]        [XX]
RSI Divergence:     [Y/N]       [Y/N]       [Y/N]
MACD Histogram:     [+/-]       [+/-]       [+/-]
Volume Trend:       [Inc/Dec]   [Inc/Dec]   [Inc/Dec]
VWAP Position:      [Above/Below] —         —

Alignment Score: [All aligned / Mostly aligned / Mixed / Conflicting]
```

### Decision Framework

| Structure Aligned? | Indicators Aligned? | Action |
|-------------------|--------------------:|--------|
| Yes | Yes | Full position, high confidence |
| Yes | Partially | Standard position, monitor indicators for confirmation |
| Yes | No | Reduced position or wait for indicators to catch up |
| No | Yes | No trade — indicators lag structure |
| No | No | No trade — wait for clarity |

Structure always takes priority. Never override a structural thesis purely because of
indicator readings — but use indicator misalignment as a reason to reduce size or wait.
