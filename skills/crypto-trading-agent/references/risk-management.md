# Risk Management & Position Sizing Reference

## Table of Contents
1. Core Risk Principles
2. Position Sizing Models
3. Leverage Framework for Crypto
4. Portfolio-Level Risk Management
5. Trade Management Rules
6. Drawdown Recovery Protocol

---

## 1. Core Risk Principles

### The Three Absolutes
1. **Never risk more than you can afford to lose on a single trade.** Default: 1-2% of account.
2. **Every trade must have a defined invalidation BEFORE entry.** No "mental stops."
3. **Risk-reward must be calculated BEFORE entry, not rationalised afterward.**

### Risk-Adjusted Thinking

The goal is not to maximise win rate — it's to maximise expected value (EV).

```
EV = (Win Rate × Average Win) - (Loss Rate × Average Loss)

Example:
  Win rate: 45%
  Average win: 3R
  Average loss: 1R
  EV = (0.45 × 3) - (0.55 × 1) = 1.35 - 0.55 = +0.80R per trade

This system loses more often than it wins but is highly profitable.
```

A system with 45% win rate and 3:1 R:R is superior to a system with 70% win rate
and 0.8:1 R:R. Always prioritise R:R over win rate.

---

## 2. Position Sizing Models

### Fixed Percentage Risk Model (Default)

The standard model. Risk a fixed percentage of account equity on each trade.

```
Position Size = (Account Balance × Risk%) / (Entry Price - Stop Loss Price)

Example:
  Account: $50,000
  Risk per trade: 1.5% = $750
  Entry: $95,000 (BTC long)
  Stop: $93,500 (below OB)
  Risk per unit: $1,500
  Position size: $750 / $1,500 = 0.5 BTC
  Notional: 0.5 × $95,000 = $47,500
  Leverage required: $47,500 / $50,000 = ~0.95x (under 1x — no leverage needed)
```

### Risk Percentage by Setup Quality

| Setup Grade | Risk % | Context |
|-------------|--------|---------|
| A+ (10-14 confluence) | 2.0% | Maximum conviction, all factors aligned |
| B (7-9 confluence) | 1.0-1.5% | Standard setup, good alignment |
| C (4-6 confluence) | 0.5% | Reduced conviction, proceed with caution |
| Scalp / Speculative | 0.25-0.5% | Quick in-and-out, no strong thesis |

### Kelly Criterion (Advanced)

For traders with sufficient data (100+ trade history):

```
Kelly % = (Win Rate / Average Loss Ratio) - ((1 - Win Rate) / Average Win Ratio)

Practical application: Use half-Kelly (divide result by 2) to account for
uncertainty in estimated probabilities.
```

### Volatility-Adjusted Sizing

Adjust position size based on current volatility (ATR):

```
Volatility Scalar = 14-day ATR Average / Current 14-day ATR

If average ATR is 2% and current ATR is 4%:
Scalar = 2% / 4% = 0.5 → halve position size

If average ATR is 2% and current ATR is 1%:
Scalar = 2% / 1% = 2.0 → cap at 1.5x normal size (don't double up on low vol,
it often precedes expansion)
```

---

## 3. Leverage Framework for Crypto

### Leverage Guidelines by Trade Type

| Trade Type | Max Leverage | Rationale |
|-----------|-------------|-----------|
| Position (days-weeks) | 3x | Wide stops needed; funding costs compound |
| Swing (hours-days) | 5-10x | Moderate stops; funding manageable |
| Intraday | 10-15x | Tight stops; no overnight funding |
| Scalp | 15-20x | Very tight stops; seconds-to-minutes hold |

### The Leverage Trap

Higher leverage doesn't mean higher risk IF position size is adjusted correctly.
The risk is the distance to your stop × position size, not the leverage multiplier.

```
SAME RISK, DIFFERENT LEVERAGE:

Account: $10,000, Risk: 1% = $100

Setup A (3x leverage):
  Entry: $95,000, Stop: $94,000 (1.05% away)
  Position: $100 / $1,000 = 0.1 BTC = $9,500 notional
  Leverage used: $9,500 / $10,000 = 0.95x (under 1x!)
  Liquidation price: Extremely far away — effectively unliquidatable

Setup B (10x leverage):
  Entry: $95,000, Stop: $94,700 (0.32% away)
  Position: $100 / $300 = 0.333 BTC = $31,667 notional
  Leverage used: $31,667 / $10,000 = 3.17x
  Liquidation: Much closer — but stop should execute well before

Both trades risk exactly $100. The difference is stop distance and position size.
```

### Liquidation Awareness

Always calculate and know your liquidation price. On Bybit:
- Isolated margin: Liquidation based on position margin only
- Cross margin: Liquidation based on entire account balance

**Rule**: Your stop loss must ALWAYS be hit before your liquidation price. If the
distance between your stop and liquidation is less than 2x the average spread + slippage
for that pair, you're too leveraged.

### Funding Rate Impact on Leverage

For perpetual futures on Bybit, funding is paid every 8 hours. Factor this into
swing/position trades:

```
Daily funding cost = Position Size × Funding Rate × 3 (three 8h periods)

Example:
  Position: $50,000 notional long
  Funding rate: 0.03% per 8h
  Daily cost: $50,000 × 0.0003 × 3 = $45/day
  Weekly cost: $315

If your expected gain on the trade is $500 over a week, $315 in funding eats 63%
of your profit. Either the position is too large or the setup needs better R:R.
```

---

## 4. Portfolio-Level Risk Management

### Correlation Awareness

In crypto, most altcoins are highly correlated with BTC. Having 5 altcoin longs is
essentially 5x exposure to BTC direction with added altcoin-specific risk.

**Portfolio correlation rules**:
- Maximum 3 simultaneous positions in the same direction (long or short)
- Treat correlated positions as a single combined risk
- Total portfolio risk across all open positions: maximum 5% of account
- If BTC is moving, all your alt positions will move with it — size accordingly

### Sector Diversification (When Taking Multiple Positions)

If running multiple positions, diversify across crypto sectors:
- Layer 1s (BTC, ETH, SOL)
- DeFi tokens
- Infrastructure/utility tokens
- Meme/momentum plays (smallest allocation)

### Cash Allocation

Never be 100% deployed. Maintain a cash reserve:
- Strong trending market (aligned across TFs): 30-50% cash minimum
- Uncertain/transitional market: 50-70% cash
- Bearish/choppy market: 70-90% cash

Cash is a position. Being in cash during uncertain conditions IS the trade.

---

## 5. Trade Management Rules

### Entry Execution

**Limit orders preferred**: Place limit orders at the refined entry zone (within the OB/FVG).
This ensures you get your planned price and avoids emotional market-order entries.

**Scaling in**: For high-conviction setups, consider:
- 50% at the top of the OB/FVG zone
- 30% at the CE (50% of the zone)
- 20% at the bottom of the zone
- Stop loss below the entire zone

### Stop Loss Management

**Hard rules**:
- Stop loss is set at entry and not moved further away. Ever.
- The only direction a stop moves is toward profit (trailing) or to break-even.

**Break-even rule**: Move stop to break-even (plus spread/fees) when price reaches 1R profit.
This makes the trade risk-free from that point forward.

**Trailing stop methods** (choose one per trade, based on setup type):
1. **Structure trail**: Move stop to below each new HL (longs) or above each new LH (shorts)
   as they form. Best for swing trades.
2. **EMA trail**: Trail stop below the 21 EMA on the execution timeframe. Best for trending
   trades with strong momentum.
3. **ATR trail**: Trail stop at entry minus 1.5-2x ATR. Best for volatile conditions.
4. **Fixed R trail**: Move stop to break-even at 1R, to 1R profit at 2R, etc.

### Take Profit Strategy

**Scaling out** (recommended for most setups):
- TP1 (33% of position): First liquidity target / opposing OB / Fib -0.272 extension
- TP2 (33% of position): Second liquidity target / Fib -0.618 extension
- TP3 (remaining 34%): Let it run with a trailing stop / Fib -1.0 extension

**Alternative: All-or-nothing** (for very high conviction / strong trend):
- Set a single TP at the full target
- Use a structure-based trailing stop
- Accept that some winners will turn into break-even trades

---

## 6. Drawdown Recovery Protocol

### Drawdown Tiers

| Drawdown | Response | Position Size Adjustment |
|----------|----------|------------------------|
| 0-5% | Normal trading | No change |
| 5-10% | Review recent trades for pattern errors | Reduce to 75% normal size |
| 10-15% | Pause new entries for 24h, full trade journal review | Reduce to 50% normal size |
| 15-20% | Stop trading for 48-72h, review entire strategy | Reduce to 25% normal size |
| 20%+ | Stop trading for 1 week minimum | Paper trade only until confidence returns |

### Recovery Mindset

After a drawdown, the instinct is to "make it back quickly" by increasing size. This is
the single most destructive behaviour in trading. The math of recovery:

```
After 20% drawdown: Need 25% gain to recover
After 30% drawdown: Need 43% gain to recover
After 50% drawdown: Need 100% gain to recover
```

Smaller position sizes during drawdowns mean slower recovery but dramatically reduce
the probability of catastrophic loss. The account survives to trade another day.

### Trade Journal Requirements

After every trade (win or loss), record:
1. Setup grade and confluence factors present
2. Entry, stop, and TP levels (planned vs actual)
3. What the market actually did vs expectation
4. Emotional state at entry and during the trade
5. Lessons / what would you do differently

Review the journal weekly. Look for patterns in losses — are they clustering around a
specific setup type, time of day, emotional state, or market condition?
