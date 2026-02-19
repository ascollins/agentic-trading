# Smart Money Concepts (SMC) & Institutional Order Flow Reference

## Table of Contents
1. Core SMC Framework
2. Order Blocks (OB)
3. Fair Value Gaps (FVG) & Imbalances
4. Liquidity Concepts
5. Inducement & Manipulation
6. Wyckoff Integration
7. SMC Confluence Scoring

---

## 1. Core SMC Framework

Smart Money Concepts model how institutional participants (market makers, hedge funds,
proprietary desks) operate in markets. The central thesis: institutions cannot fill large
orders at a single price. They must engineer liquidity, accumulate/distribute over time,
and use retail trader behaviour against them.

### The SMC Cycle

```
Accumulation → Manipulation → Distribution → Decline
     │              │              │             │
 Smart money    Stop hunts     Smart money    Retail left
 builds         trigger        offloads       holding
 position       retail exits   to retail      the bag
```

In crypto specifically, this cycle is amplified by:
- High leverage enabling cascading liquidations
- 24/7 markets with thinner liquidity during off-hours (Asian/European session overlaps)
- Funding rate mechanics creating predictable squeeze conditions
- Concentrated market maker activity on perpetual futures

### Key Principle: Follow the Imbalance

Price always seeks to fill imbalances (areas where buying and selling were not matched).
These imbalances are visible as FVGs on the chart and order blocks in the order flow.
Price gravitates toward these zones because institutional orders remain unfilled there.

---

## 2. Order Blocks (OB)

### Definition
An order block is the last opposing candle before a significant price move (displacement).
It represents the zone where institutional orders were placed.

### Bullish Order Block
- The last bearish (red/down) candle before a strong bullish displacement
- The institutional buying was absorbed by that selling candle
- When price returns to this zone, it's expected to find buyers again (the unfilled orders)

### Bearish Order Block
- The last bullish (green/up) candle before a strong bearish displacement
- The institutional selling was absorbed by that buying candle
- When price returns to this zone, it's expected to find sellers again

### Qualifying an Order Block

Not every candle before a move is a valid OB. Apply these filters:

1. **Displacement requirement**: The move away from the OB must be a displacement — a large
   body candle (or series) that's 2x+ the average candle range. Weak, grinding moves don't
   create valid OBs.

2. **Structural significance**: The OB should be associated with a BOS or a significant swing
   point. Random candles in the middle of a range are noise.

3. **Freshness**: An unmitigated (untouched) OB has higher probability than one that's been
   partially tapped. Once an OB has been fully swept through, it's invalidated.

4. **HTF alignment**: An LTF OB that sits inside an HTF OB or demand/supply zone has
   significantly higher probability.

### OB Refinement

The full OB zone (entire candle range) is often too wide for precise entries. Refine by:
- Using the body of the OB candle rather than the full wick range
- Stepping down one timeframe and looking for the specific candle within the OB
  that preceded the displacement
- Using the 50% level of the OB as the optimal entry (the "mitigation block")

### OB Invalidation
- Price closes through the OB zone with conviction (body close, not just a wick)
- The structural level the OB was protecting gets broken
- The OB has been tapped 3+ times (diminishing unfilled orders)

---

## 3. Fair Value Gaps (FVG) & Imbalances

### Definition
A Fair Value Gap is a three-candle pattern where the wicks of candle 1 and candle 3
do not overlap, leaving a gap (imbalance) at candle 2. This represents aggressive
one-sided order flow where the market moved too fast for proper two-sided filling.

### Bullish FVG
- Candle 1 high < Candle 3 low
- The gap between them is the FVG zone
- Price tends to return and fill (partially or fully) this gap before continuing higher

### Bearish FVG
- Candle 1 low > Candle 3 high
- Price tends to return and fill this gap before continuing lower

### FVG Classification

| Type | Description | Trading Approach |
|------|-------------|-----------------|
| **Consequent encroachment (CE)** | Price fills exactly 50% of the FVG | High-probability reaction point. Use as precision entry. |
| **Full fill** | Price completely fills the FVG | The imbalance is resolved. Look for the next setup. |
| **Rejection fill** | Price enters FVG but reverses before CE | Weak fill — the original directional pressure is very strong. |
| **Inverse FVG** | An FVG from the opposite direction that overlaps | Creates a battle zone. Wait for resolution before trading. |

### Using FVGs in Practice

FVGs are most powerful when they coincide with other confluences:
- An FVG inside an order block = very high probability reaction zone
- An FVG at a key Fibonacci level (especially 0.618 or 0.705-0.79 OTE) = strong entry
- An FVG aligned with the HTF bias direction = trade the fill as a pullback entry

### FVG on Different Timeframes
- **HTF FVGs (Daily/Weekly)**: These are significant structural imbalances. Price may take
  days or weeks to fill them. They act as magnets for price action.
- **MTF FVGs (4H/1H)**: Primary trading FVGs. Use for swing entry refinement.
- **LTF FVGs (15M/5M)**: Execution FVGs. Use for precise entry timing within a valid setup.

---

## 4. Liquidity Concepts

### What is Liquidity?
In SMC terms, liquidity = resting orders. Every stop loss is someone else's entry.
Institutions need liquidity to fill their orders. They will move price to where the
liquidity sits.

### Types of Liquidity

**Buy-side liquidity (BSL)**:
- Stop losses from short sellers sit above swing highs
- Buy stop orders from breakout traders sit above resistance
- This liquidity pool acts as a magnet for bullish moves
- Once swept, the buying pressure is absorbed and price often reverses

**Sell-side liquidity (SSL)**:
- Stop losses from long traders sit below swing lows
- Sell stop orders from breakdown traders sit below support
- This liquidity pool acts as a magnet for bearish moves
- Once swept, the selling pressure is absorbed and price often reverses

### Liquidity Engineering

Institutions engineer liquidity by:

1. **Equal highs/lows**: When price creates two or more touches at the same level, retail
   traders cluster their stops just beyond. This creates a dense liquidity pool that
   institutions will eventually sweep.

2. **Trendline liquidity**: Retail traders place stops below ascending trendlines and above
   descending trendlines. Institutions know this and will breach the trendline to trigger
   those stops before continuing in the original direction.

3. **Session liquidity**: Previous session (Asian, London, NY) highs and lows accumulate
   stops. The next session often sweeps the prior session's high or low before establishing
   its own direction.

4. **Round number liquidity**: Psychological levels ($50K, $60K, $100K on BTC) accumulate
   massive stop clusters. These are high-priority sweep targets.

### Liquidity Sweep vs Liquidity Grab

- **Sweep**: Price takes out the liquidity (breaks the level) with a wick but closes back
  inside. This is the classic "stop hunt" — a rapid move to trigger stops followed by
  immediate reversal. High-probability reversal signal.

- **Grab**: Price takes out the liquidity and holds beyond the level. This is a genuine
  breakout/breakdown. The liquidity was used to fuel continuation, not reversal.

How to tell the difference: Watch the displacement after the sweep. If price immediately
and aggressively moves away from the swept level (displacement candle), it's a sweep.
If price consolidates or grinds beyond the level, it's more likely a genuine grab.

---

## 5. Inducement & Manipulation

### Inducement
Inducement is a minor structural break designed to trap traders into the wrong side before
the real move. It's the market's way of building positions at better prices.

**How it works**:
1. Price is in a downtrend (LH/LL sequence)
2. A minor CHoCH appears — price breaks a recent LH slightly
3. Retail traders interpret this as a reversal and go long
4. Price immediately reverses, sweeping the new longs' stops
5. This creates the sell-side liquidity the institutions needed to continue lower

### Identifying Inducement vs Real CHoCH

| Factor | Inducement | Real CHoCH |
|--------|-----------|------------|
| Displacement | Weak, no follow-through | Strong displacement candle |
| Volume | Low/average on the break | Above-average, expanding |
| Timeframe | Occurs on LTF only | Visible on MTF |
| Location | Random within the trend | At a significant HTF level |
| Follow-through | Immediate reversal | Builds structure in new direction |

### Manipulation Patterns in Crypto

Crypto markets are particularly prone to manipulation due to fragmented liquidity and
leverage. Common patterns:

1. **Liquidation cascade engineering**: Price pushed to a level where a cluster of
   liquidations sit, triggering a cascade that fills institutional orders at favourable prices.

2. **Funding rate manipulation**: In perpetuals, when funding is highly positive, a quick
   dump can flip the rate and force longs to close, creating sell pressure that fills
   institutional buy orders.

3. **Thin-book manipulation**: During low-liquidity hours (typically 00:00-06:00 UTC),
   smaller capital can move price significantly to engineer setups for the next active session.

4. **News-based manipulation**: Major news events create volatility spikes. The initial
   move is often a liquidity grab (stop hunt) before the "real" move begins. Wait 15-30
   minutes after major news for the dust to settle.

---

## 6. Wyckoff Integration

### Wyckoff Phases Mapped to SMC

The Wyckoff method and SMC are complementary frameworks describing the same institutional
behaviour. Here's how they align:

| Wyckoff Phase | SMC Equivalent | What's Happening |
|--------------|----------------|-----------------|
| Preliminary Support (PS) | First demand zone reaction | Initial buying appears after a decline |
| Selling Climax (SC) | SSL sweep + displacement | Capitulation selling, institutions start buying |
| Automatic Rally (AR) | Bullish BOS on LTF | Short covering rally, defines range top |
| Secondary Test (ST) | OB retest | Test of supply at SC level, lower volume |
| Spring | Liquidity sweep below range | Final manipulation below range — the key entry signal |
| Sign of Strength (SOS) | Bullish BOS on MTF + displacement | Confirmed breakout with volume |
| Last Point of Support (LPS) | OB retest after BOS | Final pullback before markup phase |

### The Spring as an Entry Signal

The Spring is one of the highest-probability setups in both Wyckoff and SMC:
1. Price breaks below range support (sweeps SSL)
2. Volume spikes but price immediately reverses
3. The spring candle leaves a long lower wick
4. This is your entry zone — invalidation below the spring low
5. Target: opposite side of the range (AR level), then beyond

---

## 7. SMC Confluence Scoring

When evaluating a setup, score each confluence factor:

| Factor | Present? | Weight |
|--------|----------|--------|
| HTF bias aligned | □ | 3 |
| Unmitigated OB at entry | □ | 2 |
| FVG overlapping entry zone | □ | 2 |
| Liquidity swept before entry | □ | 3 |
| Momentum divergence confirmation | □ | 1 |
| Volume confirmation | □ | 1 |
| Funding rate supportive | □ | 1 |
| OI context supportive | □ | 1 |

**Scoring**:
- 10-14 points: A+ setup — full position size
- 7-9 points: B setup — standard position size
- 4-6 points: C setup — reduced size or skip
- Below 4: No trade

This is a guideline, not a rigid formula. Context matters — a single high-weight factor
(like a perfect HTF-aligned liquidity sweep into an unmitigated OB) can override a lower
total score. Use judgment.
