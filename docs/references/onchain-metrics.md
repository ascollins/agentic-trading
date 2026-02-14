# On-Chain Metrics — Crypto Fundamental Analysis Reference

On-chain data is to digital assets what fundamental analysis is to equities (CMT L3).
These metrics provide insight into network health, investor behaviour, and supply dynamics
that are invisible on a price chart alone.

Primary data sources: Glassnode, CryptoQuant, IntoTheBlock, Santiment, Dune Analytics, DefiLlama

---

## 1. Valuation Metrics

### NVT Ratio (Network Value to Transactions)
- **Formula**: Market Cap / Daily Transaction Volume (USD)
- **Interpretation**: Crypto's equivalent of P/E ratio
- High NVT (>95): Network is overvalued relative to its transaction utility — price may be driven by speculation
- Low NVT (<65): Network is undervalued relative to economic throughput — potential accumulation zone
- **NVT Signal** (smoothed variant): Uses 90-day MA of transaction volume for less noise
- **Best for**: BTC, ETH — less reliable for low-throughput chains

### MVRV Ratio (Market Value to Realised Value)
- **Formula**: Market Cap / Realised Cap
- **Realised Cap**: Sum of all coins valued at the price they last moved on-chain (cost basis of all holders)
- MVRV > 3.5: Historically marks cycle tops — aggregate holders sitting on 250%+ unrealised profit, distribution likely
- MVRV between 1.0-2.0: Healthy mid-cycle range
- MVRV < 1.0: Market trading below aggregate cost basis — capitulation/max pain zone, historically excellent long-term entry
- **MVRV Z-Score**: Standardised version — above 7 = extreme overvaluation, below 0 = extreme undervaluation

### SOPR (Spent Output Profit Ratio)
- **Formula**: Realised Value of spent outputs / Value at creation of spent outputs
- SOPR > 1: Coins moving at a profit on average — holders are in profit
- SOPR < 1: Coins moving at a loss — holders selling at a loss (capitulation)
- SOPR = 1: Break-even — acts as support in bull markets (holders refuse to sell at loss), resistance in bear markets
- **aSOPR** (adjusted): Filters out outputs younger than 1 hour to remove noise from change addresses
- **Trading signal**: In confirmed uptrends, SOPR resets to 1.0 and bounces = buy-the-dip confirmation

### Puell Multiple
- **Formula**: Daily coin issuance (USD) / 365-day MA of daily issuance
- Measures miner revenue relative to historical average
- Puell > 4: Miners earning significantly above average — increased selling pressure likely
- Puell < 0.5: Miners in distress, potential capitulation — historically marks cycle bottoms
- **Best for**: BTC (direct mining economics), less applicable to PoS chains

### Thermocap Multiple
- **Formula**: Market Cap / Cumulative miner revenue (all-time)
- Measures speculative premium above total security spend
- Values > 32: Historically coincide with cycle peaks
- Values < 8: Historically coincide with accumulation phases

---

## 2. Supply Distribution Metrics

### Exchange Flows
- **Exchange Inflows**: Coins moving TO exchanges — potential selling pressure
  - Spike in inflows after a rally = profit-taking signal
  - Spike in inflows during fear = panic selling / capitulation
- **Exchange Outflows**: Coins moving FROM exchanges — accumulation signal
  - Sustained outflows = holders moving to cold storage (bullish long-term)
- **Net Flow**: Inflows minus outflows
  - Persistent negative net flow (outflows > inflows) = supply squeeze developing
  - Persistent positive net flow = distribution / selling pressure building
- **Whale Exchange Deposits**: Track deposits > 100 BTC or > 1000 ETH specifically — institutional-scale moves

### Supply Held by Long-Term vs. Short-Term Holders
- **Long-Term Holders (LTH)**: Coins held > 155 days
  - LTH supply increasing = accumulation phase (bullish)
  - LTH supply decreasing = distribution phase (bearish, especially near ATH)
- **Short-Term Holders (STH)**: Coins held < 155 days
  - STH cost basis acts as dynamic support/resistance
  - Price below STH realised price = STH underwater, potential panic zone
- **LTH-to-STH supply ratio**: Rising = accumulation, falling = distribution

### Coin Days Destroyed (CDD)
- **Concept**: When a coin that hasn't moved for 100 days is spent, it destroys 100 coin-days
- High CDD spike: Old coins moving — potential signal that long-term holders are distributing
- Sustained low CDD: Dormant supply, holders convicted — bullish for supply dynamics
- **Binary CDD**: 1 when CDD is above average, 0 when below — simpler trend signal

### Supply in Profit/Loss
- **% Supply in Profit**: When > 95%, market is euphoric — distribution likely
- **% Supply in Loss**: When > 50%, deep bear territory — capitulation may be near
- Convergence of profit/loss at ~50/50 often marks major turning points

---

## 3. Network Activity Metrics

### Active Addresses
- **Definition**: Unique addresses participating in transactions (daily)
- Rising active addresses + rising price = healthy trend (demand-driven)
- Rising price + flat/falling addresses = speculative rally (less sustainable)
- Falling price + rising addresses = potential accumulation
- **Caution**: Can be inflated by spam transactions or protocol-level activity

### New Addresses
- Measures network growth / new user adoption
- Sustained new address growth in uptrend = organic demand
- New address decline during rally = late-stage speculation

### Transaction Count & Volume
- Increasing transaction count = network utility growing
- Large transaction volume (>$100K) = institutional/whale activity
- **Adjusted transaction volume**: Filters out change addresses and internal transfers for cleaner signal

### Hash Rate (PoW) / Staking Rate (PoS)
- **Hash rate** (BTC): Rising = miner confidence, network security increasing. Hash rate new ATH while price consolidates = bullish divergence (miners investing despite price)
- **Staking rate** (ETH, SOL, etc.): Higher staking = more supply locked, reduced liquid supply. Validator growth = network confidence
- **Hash Ribbons**: When 30-day MA of hash rate crosses above 60-day MA after a miner capitulation = historically strong buy signal (BTC)

---

## 4. DeFi-Specific Metrics

### Total Value Locked (TVL)
- Aggregate capital deposited in DeFi protocols
- Rising TVL = capital flowing into ecosystem (bullish for ecosystem tokens)
- TVL/Market Cap ratio: Low ratio = potential value, high ratio = may be fully valued
- **Caution**: TVL can be inflated by recursive leverage / yield farming loops

### DEX Volume / CEX Volume Ratio
- Rising DEX share = decentralisation trend, potential regulatory hedge
- Spikes in DEX volume often precede or accompany high volatility events

### Stablecoin Supply & Flows
- **Total stablecoin market cap**: Growing = dry powder available to enter crypto
- **Stablecoin supply ratio (SSR)**: BTC market cap / total stablecoin market cap
  - Low SSR = high buying power relative to BTC — bullish
  - High SSR = low buying power — less fuel for rally
- **Stablecoin exchange reserves**: Growing = potential buying pressure, capital staged for entry
- **USDT/USDC dominance**: USDT dominance rising in risk-off, USDC in risk-on (institutional preference)

### Protocol Revenue
- Fees generated by protocol = real economic activity
- Revenue-based valuation (P/F ratio): Protocol market cap / annualised fees
- Sustainable fee generation > token emissions = deflationary pressure potential

---

## 5. Derivatives On-Chain Metrics

### Futures Open Interest (On-Chain)
- Rising OI + rising price = new longs, conviction (if not excessive)
- Rising OI + falling price = new shorts building
- OI at ATH with price stalling = potential liquidation cascade risk
- Sudden OI drop = mass liquidation event just occurred

### Funding Rate
- Positive funding > 0.01%/8h: Longs paying shorts — moderate bullish consensus
- Positive funding > 0.05%/8h: Crowded long — elevated squeeze risk
- Negative funding < -0.03%/8h: Crowded short — short squeeze likely
- Near-zero funding with trending price = healthiest trend signal

### Estimated Leverage Ratio
- **Formula**: Open Interest / Exchange Reserve
- Rising ELR = more leverage in system relative to available collateral — fragility increasing
- Extremely high ELR = system is overleveraged, liquidation cascade probability elevated

### Options Open Interest & Volume
- Put/Call ratio rising: Hedging demand increasing, potential fear
- Max pain price: Where most options expire worthless — price tends to gravitate toward this near expiry
- Large OI clusters at specific strikes = potential support/resistance magnets

---

## 6. Composite Indicators & Frameworks

### Bitcoin Cycle Indicators (Multi-Metric)
- **Pi Cycle Top**: 111-day MA crosses above 2x 350-day MA — historically marks within 3 days of cycle top
- **200-Week MA**: Long-term mean. Price touching 200W MA has historically been generational buy zone
- **Stock-to-Flow** (S2F): Controversial but watched. Based on scarcity (issuance relative to existing supply). Useful as one input, not standalone.
- **Rainbow Chart**: Log regression band — shows where price sits within historical range (speculative mania → fire sale)

### On-Chain Scoring Framework
When multiple on-chain metrics align, conviction increases:

| Signal | Bullish Alignment | Bearish Alignment |
|--------|-------------------|-------------------|
| MVRV | < 1.5 | > 3.0 |
| Exchange flows | Net outflows | Net inflows |
| LTH supply | Increasing | Decreasing |
| Active addresses | Rising | Falling |
| Funding rate | Neutral/slightly positive | Extreme positive |
| SOPR | Bouncing off 1.0 | Breaking below 1.0 |
| Stablecoin reserves | Growing on exchanges | Declining on exchanges |

Score 5+ aligned = high conviction. Score 3-4 = moderate. Below 3 = mixed/wait.

---

## 7. Chain-Specific Considerations

### Bitcoin
- Most developed on-chain analytics. MVRV, SOPR, hash rate, coin days destroyed all well-calibrated.
- Halving cycle is a unique supply shock — reduces issuance by 50% every ~4 years. Historically, price peaks 12-18 months post-halving.

### Ethereum
- Transition to PoS changed key metrics: validator count and staking rate replace hash rate
- EIP-1559 burn mechanism: Track daily ETH burned vs. issued. Net deflationary when burn > issuance
- Blob fees (L2 data posting) as new metric for L2 ecosystem health

### Solana
- TPS and fee revenue as network utility proxies
- Staking rate typically very high (>60%) — large portion of supply locked
- Validator economics and decentralisation metrics (Nakamoto coefficient)

### Layer 2s (ARB, OP, etc.)
- TVL as primary metric. Sequencer revenue for actual economic activity.
- Bridge flows: capital flowing to L2 = ecosystem growth signal
- User/address growth on L2 relative to L1

---

## Usage Notes for Trading

- On-chain metrics are **medium-to-long-term signals** (days to weeks). They don't time entries on 4H charts.
- Best used as **confluence with technical analysis**: TA gives timing, on-chain gives conviction and context.
- Multiple metrics aligning > any single metric in isolation.
- On-chain data has latency — most metrics update daily, some hourly. Not suitable for intraday decisions.
- Some metrics are BTC-native and don't translate cleanly to altcoins. Noted above where applicable.
