# Crypto Options — Greeks, Volatility & Strategies Reference

Covers options mechanics as applied to crypto markets. Grounded in CMT L1 options fundamentals
and L2/L3 volatility analysis. Primary venue: Deribit (dominant crypto options exchange).
Secondary: CME (BTC/ETH options), OKX, Bybit.

---

## 1. The Greeks — Core Definitions

### Delta (Δ)
- **Measures**: Rate of change of option price per $1 move in underlying
- **Range**: 0 to 1 for calls, 0 to -1 for puts
- ATM options ≈ 0.50 delta (calls) / -0.50 (puts)
- Deep ITM → delta approaches 1.0 (call) or -1.0 (put)
- Deep OTM → delta approaches 0
- **Trading use**: Delta tells you directional exposure. A 0.30 delta call gains ~$0.30 for every $1 BTC rises.
- **Portfolio delta**: Sum of all position deltas. Delta-neutral = no net directional bias.
- **Crypto nuance**: In crypto's high-vol environment, delta can shift rapidly — gamma risk is amplified.

### Gamma (Γ)
- **Measures**: Rate of change of delta per $1 move in underlying
- Highest for ATM options, near expiry
- **Trading use**: Gamma tells you how much your delta will change. High gamma = delta shifts fast = position becomes more directional quickly.
- **Long gamma**: Long options positions. Benefits from big moves in either direction. Cost = time decay (theta).
- **Short gamma**: Short options positions. Benefits from low volatility / range-bound markets. Risk = large sudden moves.
- **Crypto nuance**: Crypto's tendency for explosive moves makes short gamma positions particularly dangerous. Weekend gaps and 24/7 trading mean there's no "closing bell" protection.

### Theta (Θ)
- **Measures**: Time decay — loss of option value per day, all else equal
- Always negative for long options (you pay for time)
- Accelerates as expiry approaches (steepest decay in final 2 weeks)
- **Trading use**: Long options = paying theta (need the move to happen). Short options = collecting theta (need the move NOT to happen).
- **Crypto nuance**: Crypto options often have elevated IV, which means premium is fatter — but theta decay is also higher in absolute terms. Weeklies on Deribit have aggressive theta decay.

### Vega (ν)
- **Measures**: Change in option price per 1% change in implied volatility
- Highest for ATM, long-dated options
- **Trading use**: Long vega = benefit from IV increase. Short vega = benefit from IV decrease (volatility crush).
- **Crypto nuance**: Vega is arguably the most important Greek in crypto. IV swings are massive — crypto IV can move 20-30 points in a day during events. Buying options before a known catalyst (ETF decisions, halvings, Fed meetings) is essentially a vega bet.

### Rho (ρ)
- **Measures**: Sensitivity to interest rate changes
- Minimal impact in crypto (no standard risk-free rate in DeFi), generally ignored for trading purposes
- Slightly more relevant for CME-listed crypto options where USD financing rates apply

---

## 2. Implied Volatility in Crypto

### IV Basics (CMT L1-L2)
- IV is the market's forecast of future volatility, derived from option prices via pricing models
- **Historical/realised volatility (HV)**: What actually happened. Calculated from past price returns.
- **IV > HV**: Options are "expensive" — market expects more volatility than recently realised. Favours premium sellers.
- **IV < HV**: Options are "cheap" — market complacent. Favours premium buyers.
- **Volatility Risk Premium (VRP)**: IV typically exceeds HV (CMT L2). In crypto, VRP averages 10-20 IV points — sellers of crypto options are generally compensated for the risk.

### Crypto IV Levels — Context
- BTC IV 30-40%: Low volatility. Historically precedes explosive moves (squeeze setup).
- BTC IV 50-70%: Normal range for trending markets.
- BTC IV 80-100%: Elevated fear/uncertainty. Often near local bottoms if spiking from lower levels.
- BTC IV > 100%: Extreme. Usually event-driven (exchange collapse, regulatory shock, liquidation cascade).
- ETH typically trades 5-15% higher IV than BTC. Altcoin IV often 100%+ even in calm markets.

### DVOL (Deribit Volatility Index)
- Crypto's equivalent of VIX. Measures 30-day implied volatility of BTC options on Deribit.
- DVOL rising + price falling = fear increasing (put demand)
- DVOL falling + price rising = healthy rally, confidence increasing
- DVOL at extreme lows = complacency, potential for volatility expansion (like low VIX readings in equities)
- ETH DVOL also available — tracks ETH-specific volatility sentiment

### Term Structure
- **Contango** (normal): Longer-dated IV > shorter-dated IV. Market expects volatility to increase or remain stable over time.
- **Backwardation** (inverted): Short-dated IV > longer-dated IV. Near-term fear/uncertainty — event risk priced in.
- Shift from contango to backwardation = early warning signal (CMT L2 — same concept as VIX term structure).
- **Trading use**: In backwardation, calendar spreads (sell short-dated, buy long-dated) can capture term structure normalisation.

### Volatility Smile/Skew
- **Skew**: Difference between OTM put IV and OTM call IV at same delta (typically 25-delta)
- Negative skew (puts > calls): Market pricing in downside risk — fear
- Positive skew (calls > puts): Market pricing in upside — FOMO/euphoria (common in crypto bull runs)
- **Crypto-specific**: BTC often shows positive skew in bull markets (call buying from institutions wanting upside exposure). This is unusual vs. equities which almost always have negative skew.
- Skew reversal (from positive to negative) = sentiment shift worth noting

---

## 3. Key Volatility Formulas

### Expected Move from IV (CMT L2)
```
Daily Expected Move = Price × (IV / √365)
Weekly Expected Move = Price × (IV / √52)
Monthly Expected Move = Price × (IV / √12)
N-day Expected Move = Price × (IV × √(N/365))
```
Example: BTC at $100,000, IV = 60%
- Daily: $100,000 × (0.60 / √365) = ~$3,140 (±3.14%)
- Weekly: $100,000 × (0.60 / √52) = ~$8,321
- Monthly: $100,000 × (0.60 / √12) = ~$17,321

### IV from Expected Move (reverse)
```
Implied Volatility = (Expected Daily Move / Price) × √365
```

### Annualising Realised Volatility
```
HV = StdDev(daily log returns) × √365    [crypto uses 365, not 252]
```
Note: Crypto trades 365 days/year, so annualisation factor is √365, not √252 (equities).

---

## 4. Options Strategies for Crypto

### Directional Strategies

**Long Call / Long Put**
- Pure directional bet with defined risk (premium paid)
- Use when: High conviction on direction, expect the move within the option's timeframe
- Max loss: Premium paid. Max gain: Unlimited (call) / substantial (put).
- Crypto consideration: High IV means expensive premiums. Consider spreads to reduce cost.

**Bull/Bear Call/Put Spread (Vertical)**
- Buy one strike, sell another at a different strike (same expiry)
- Reduces cost but caps profit
- Use when: Moderately bullish/bearish, want to reduce premium outlay in high-IV environments
- Crypto consideration: Vertical spreads are often more capital-efficient in crypto's high-IV regime

### Volatility Strategies

**Long Straddle (ATM Call + ATM Put)**
- Profit from large moves in either direction
- Cost: Both premiums (expensive in crypto)
- Use when: Expecting a major move but unsure of direction (pre-halving, pre-ETF decision, pre-FOMC)
- Break-even: Strike ± total premium paid
- Crypto consideration: High IV = expensive straddles. Better to buy before the IV spike, not during.

**Long Strangle (OTM Call + OTM Put)**
- Cheaper version of straddle — wider break-even but lower cost
- Use when: Same thesis as straddle but want lower premium outlay
- Crypto consideration: 15-25 delta strangles are popular in crypto for event plays

**Short Straddle/Strangle (Premium Selling)**
- Collect premium, profit from range-bound / low volatility
- Risk: Unlimited on large moves
- Use when: IV is elevated and you expect a vol crush / mean reversion
- Crypto consideration: EXTREMELY risky in crypto given tail risk. Size conservatively. Always have a stop plan.

### Income / Yield Strategies

**Covered Call (Long spot + Short OTM call)**
- Generate yield on held crypto by selling upside
- Use when: Holding long-term, willing to cap upside for income
- Crypto consideration: Works well in sideways/mild bull markets. Risk is capping yourself during parabolic moves — use further OTM strikes (20-25 delta) to leave room.

**Cash-Secured Put (Short OTM put)**
- Collect premium with obligation to buy at strike if assigned
- Use when: Wanting to accumulate at lower prices while getting paid to wait
- Crypto consideration: Part of the "wheel strategy." Effective when you have a target buy zone from your TA. Sell puts at or near your technical support levels.

**The Wheel (Cash-Secured Put → Covered Call cycle)**
- Sell puts until assigned, then sell calls until called away, repeat
- Generates consistent premium income
- Best in: Range-bound to mild trending markets
- Risk: Large drawdown if underlying drops substantially below put strike
- Crypto consideration: Works on BTC/ETH where you're comfortable holding through drawdowns. Avoid on volatile altcoins where price can drop 50%+ and never recover.

### Hedging Strategies

**Protective Put (Long spot + Long put)**
- Insurance against downside while maintaining upside exposure
- Cost: Put premium reduces overall returns
- Use when: Holding significant spot position and want defined downside protection (e.g., ahead of macro events)

**Collar (Long spot + Long OTM put + Short OTM call)**
- Caps both downside and upside — put financed by call premium
- Zero-cost collar: Premium from call fully offsets put cost
- Use when: Wanting protection without net cost but willing to cap upside
- Crypto consideration: Effective for portfolio hedging of large BTC/ETH positions

**Put Spread Collar**
- Variant using put spread (long put + short further OTM put) instead of single put
- Reduces or eliminates cost while providing partial protection

---

## 5. Greeks-Based Position Management

### Delta Management
- Monitor portfolio delta. If net long delta exceeds comfort during uncertain period, reduce via:
  - Selling calls against positions
  - Buying puts
  - Reducing spot
- Delta-neutral strategies: Use for pure volatility plays without directional risk

### Gamma Awareness
- As options approach expiry, gamma increases dramatically (especially ATM)
- Near expiry, small price moves create large delta swings — position can whipsaw
- "Gamma risk" is amplified in crypto's 24/7 market — no overnight gap protection
- For short gamma positions: set hard stops or roll before final week

### Theta Harvesting
- In range-bound crypto markets, short premium strategies can be profitable
- Theta decay accelerates in final 7-14 days — this is when time decay income is richest but gamma risk is highest
- Balance theta collection against gamma risk by adjusting position size near expiry

### Vega Trading
- Before known events (halving, ETF decisions, FOMC, upgrades): IV tends to rise → long vega
- After events resolve: IV crushes → short vega (sell the news)
- Vega P&L = Vega × Change in IV. A 5-point IV drop on a position with $500 vega = $2,500 loss for long vega

---

## 6. Crypto-Specific Options Considerations

### 24/7 Markets
- No market close means no overnight gap risk — but also no overnight protection
- Theta ticks continuously, not just on market days
- Weekend moves can be significant — Friday expiry options have unique dynamics

### Settlement
- Deribit: Cash-settled in BTC or ETH (not USD). Profit/loss denominated in the underlying.
- This creates "quanto risk" — your PnL changes value as the underlying moves
- CME: Cash-settled in USD — cleaner for USD-denominated portfolios

### Liquidity
- BTC options: Good liquidity on Deribit for ATM and near-term. Wider spreads on far OTM/long-dated.
- ETH options: Decent but thinner than BTC
- Altcoin options: Very thin. Stick to BTC/ETH for options strategies.
- Use limit orders. Market orders in crypto options can be costly due to wide bid-ask spreads.

### Max Pain & Expiry Dynamics
- Large quarterly expiries (end of March, June, Sept, Dec) often see price gravitating toward max pain
- Weekly expiries on Fridays (Deribit) — smaller impact but still notable
- "Pin risk" near expiry: price may be pushed toward strikes with large OI

### Regulatory Considerations
- Deribit not available to US-based traders (offshore)
- CME options: US-regulated, institutional-grade
- Tax treatment of options varies by jurisdiction — consult relevant guidance
