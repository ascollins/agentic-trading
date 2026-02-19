# Market Structure & Regime Detection Reference

## Table of Contents
1. Market Regime Classification
2. Swing Structure Mapping
3. Break of Structure (BOS) & Change of Character (CHoCH)
4. Multi-Timeframe Structure Alignment
5. Regime Transition Detection
6. Practical Application Templates

---

## 1. Market Regime Classification

Every analysis begins by classifying the current regime. There are three primary regimes,
each with distinct sub-types that inform strategy selection.

### Trending Regime
**Characteristics**: Sequential HH/HL (bullish) or LH/LL (bearish), price respecting
dynamic support/resistance (21/50 EMA), expanding volume on impulse legs.

Sub-types:
- **Impulse phase**: Strong directional moves with minimal pullbacks. Pullbacks are shallow
  (38.2% Fib or less). Strategy: trend continuation entries on pullbacks to OBs/FVGs.
- **Corrective phase**: Deeper pullbacks (50-78.6% Fib) within the trend. Price often fills
  FVGs and retests OBs. Strategy: wait for structure confirmation before re-entering.
- **Exhaustion phase**: Trend slowing — smaller impulse candles, increasing wicks, momentum
  divergence. Strategy: tighten stops, reduce position sizes, prepare for reversal.

### Ranging Regime
**Characteristics**: Price oscillating between defined support and resistance. Multiple
rejections at boundaries. Declining volume. Mean-reverting behaviour.

Sub-types:
- **Accumulation range**: After a downtrend, characterised by spring/upthrust patterns
  (Wyckoff). Smart money absorbing supply. Watch for the "spring" — a fake breakdown below
  range support that immediately reverses.
- **Distribution range**: After an uptrend, characterised by UTAD (upthrust after distribution).
  Smart money offloading to retail. Watch for the "upthrust" — a fake breakout above range
  resistance that immediately reverses.
- **Re-accumulation/re-distribution**: Mid-trend consolidation. The trend is pausing, not
  reversing. Key tell: the range forms above (re-accumulation) or below (re-distribution)
  the equilibrium of the prior impulse.

### Transitional Regime
**Characteristics**: A regime shift is occurring — trending to ranging or vice versa. Signals
include BOS/CHoCH on HTF, divergence between price and momentum, volume anomalies.

Transition patterns:
- **Trend → Range**: Watch for a failed impulse (unable to make new HH/LL) followed by price
  accepting both sides of the prior impulse candle. Typically coincides with HTF momentum
  divergence.
- **Range → Trend**: Watch for a definitive break of range support/resistance with volume
  expansion and a displacement candle (large body, small wicks). The breakout candle's body
  should close convincingly beyond the range boundary.
- **Trend reversal**: CHoCH on HTF followed by a confirmed BOS in the opposite direction.
  This is the highest-conviction transition signal.

---

## 2. Swing Structure Mapping

### Identifying Valid Swings

A swing point is valid when it creates a clear pivot that's respected by subsequent price action.
Not every wick is a swing — filter noise by:

- **HTF (Weekly/Daily)**: Only count swings that hold for 5+ candles on each side
- **MTF (4H/1H)**: Count swings that hold for 10+ candles on each side
- **LTF (15M/5M)**: Count swings that hold for 15+ candles on each side (used for execution only)

### Structure Types

**Internal structure**: The smaller swings within an impulse or corrective leg. Used for
fine-tuning entries but not for directional bias.

**External structure**: The major swing points that define the trend. These are the HH/HL
or LH/LL sequence that determines bullish or bearish bias.

### Mapping Process

1. Start on the highest relevant timeframe (Weekly for swing trades, Daily for intraday)
2. Mark the last 3-5 major swing highs and lows
3. Label them: HH, HL, LH, LL, or EQ (equal)
4. Draw horizontal lines at each swing point — these become key levels
5. Step down one timeframe and repeat
6. Note where HTF and LTF structure align or conflict

---

## 3. Break of Structure (BOS) & Change of Character (CHoCH)

### Break of Structure (BOS)
A BOS confirms trend continuation:
- **Bullish BOS**: Price breaks above a previous swing high (HH), confirming the uptrend
- **Bearish BOS**: Price breaks below a previous swing low (LL), confirming the downtrend

Validation criteria:
- The candle body must close beyond the swing level (wicks don't count for confirmation)
- Volume should expand on the BOS candle relative to the prior 5 candles
- The BOS should leave behind a clear order block (the last down-candle before a bullish BOS,
  or the last up-candle before a bearish BOS)

### Change of Character (CHoCH)
A CHoCH signals a potential trend reversal:
- **Bullish CHoCH**: In a downtrend (LH/LL sequence), price breaks above the most recent
  lower high. First signal that sellers are losing control.
- **Bearish CHoCH**: In an uptrend (HH/HL sequence), price breaks below the most recent
  higher low. First signal that buyers are losing control.

Important: A CHoCH is a warning, not a confirmed reversal. It means "the current trend
structure has been violated." A reversal is confirmed only when a CHoCH is followed by
a BOS in the new direction.

### CHoCH vs BOS Decision Tree

```
Price breaks a swing point
├── In the direction of the existing trend?
│   └── YES → This is a BOS (trend continuation confirmed)
│       └── Look for pullback entries to the new OB
└── Against the existing trend?
    └── YES → This is a CHoCH (potential reversal)
        ├── Wait for confirmation BOS in new direction
        ├── DO NOT enter on CHoCH alone
        └── Mark the CHoCH candle as a potential OB for future reference
```

---

## 4. Multi-Timeframe Structure Alignment

### The Fractal Principle
Market structure is fractal — the patterns on the 5M chart mirror those on the Weekly chart.
The key insight is that higher timeframe structure always dominates. A bullish LTF setup
inside a bearish HTF context has a lower probability of success.

### Alignment Scoring

Score the alignment across three timeframes (HTF, MTF, LTF) from 1-5:

| Score | Alignment | Interpretation |
|-------|-----------|----------------|
| 5 | All three TFs trending in the same direction with clear structure | Highest conviction setups. Trend-following entries. |
| 4 | HTF and MTF aligned, LTF pulling back (or forming a setup) | Standard high-probability entry zone. This is the bread and butter. |
| 3 | HTF clear, MTF transitional, LTF mixed | Reduced position size. Wait for MTF confirmation. |
| 2 | HTF and MTF conflicting | Stay flat or scalp only. No swing positions. |
| 1 | All TFs conflicting or ranging | No trade. Sit on hands. Capital preservation mode. |

### Recommended Timeframe Combinations

| Trading Style | HTF (Bias) | MTF (Setup) | LTF (Entry) |
|---------------|-----------|-------------|-------------|
| Position/Swing | Weekly | Daily | 4H |
| Swing | Daily | 4H | 1H |
| Intraday Swing | 4H | 1H | 15M |
| Scalp | 1H | 15M | 5M/1M |

---

## 5. Regime Transition Detection

### Early Warning Signals

These signals suggest a regime transition is approaching (in order of reliability):

1. **Momentum divergence on HTF**: RSI or MACD divergence on Daily/Weekly. This is the
   earliest and most reliable signal that the current regime is weakening.

2. **Volume anomaly**: Climactic volume (3x+ average) on an impulse leg that fails to
   produce proportional price movement. Indicates absorption.

3. **Time-based exhaustion**: The current impulse leg has been running for significantly
   longer than previous legs without a meaningful correction. Measured in candles.

4. **Structural degradation**: Impulse legs getting shorter, corrections getting deeper.
   The trend is losing momentum even if structure hasn't broken yet.

5. **Funding rate extremes** (crypto-specific): When funding is at extreme levels
   (>0.05% per 8h on Bybit), the trend is crowded and vulnerable to a squeeze.

6. **Open interest divergence**: OI rising while price is flat or falling (bearish divergence)
   or OI falling while price is rising (distribution).

### Confirmation Sequence

A regime transition is confirmed through this sequence:
1. Early warning signals appear (momentum divergence, volume anomaly)
2. CHoCH on the MTF
3. BOS in the new direction on the MTF
4. HTF structure adjusts (the MTF BOS creates a CHoCH on the HTF)

Trade the new regime only after step 3. Anticipating transitions before step 2 is gambling.

---

## 6. Practical Application Templates

### Template: HTF Context Assessment

```
MARKET STRUCTURE ASSESSMENT — [PAIR] — [DATE]

Weekly Structure:
  Last 3 swings: [HH/HL/HH] or [LH/LL/LH] etc.
  Current trend: [Bullish/Bearish/Ranging]
  Key levels: [List 2-3 critical weekly S/R]

Daily Structure:
  Aligned with Weekly?: [Yes/No/Transitional]
  Last swing: [Direction + level]
  Active BOS/CHoCH: [Details if any]
  
4H Structure (if relevant):
  Internal structure direction: [Bullish/Bearish]
  Key OBs/FVGs: [List]

Regime Classification: [Trending-Impulse / Trending-Corrective / etc.]
Alignment Score: [1-5]
Bias: [Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish]
```

### Template: Regime Transition Alert

```
REGIME TRANSITION ALERT — [PAIR]

Current Regime: [X]
Suspected New Regime: [Y]

Warning Signals Present:
  □ Momentum divergence ([TF])
  □ Volume anomaly
  □ Time exhaustion
  □ Structural degradation
  □ Funding rate extreme
  □ OI divergence

Transition Stage:
  □ Early warning (signals only)
  □ CHoCH confirmed on [TF]
  □ BOS in new direction on [TF]
  □ HTF structure adjusted

Action:
  [Reduce exposure / Tighten stops / Prepare reversal entries / No action yet]
```
