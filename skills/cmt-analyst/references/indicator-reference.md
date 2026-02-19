# Indicator Reference — CMT Level I Complete

Source chapters: 7 (MAs), 12 (Volume Indicators), 13 (Confirmation/Oscillators), 18 (Breadth), 29-30 (Volatility)

---

## 1. Moving Averages (Ch 7)

### Simple Moving Average (SMA)
- **Formula**: SMA = (P₁ + P₂ + ... + Pₙ) / n
- **Characteristic**: Equal weight to all observations. Smoothest. Most lag.
- **Common periods**: 10 (short-term), 20 (Bollinger default), 50 (medium-term trend), 100, 200 (long-term trend)
- **Signals**: Price crossing above SMA = bullish. Below = bearish. Short SMA crossing long SMA = trend change (golden cross/death cross).
- **Limitation**: Lagging indicator by definition. Whipsaws in sideways markets.

### Weighted Moving Average (WMA)
- **Formula**: WMA = Σ(Pᵢ × wᵢ) / Σ(wᵢ), where weights are linear: n, n-1, n-2, ..., 1
- **Characteristic**: More weight to recent prices than SMA but less than EMA. Middle ground.
- **Usage**: Less common in practice than SMA or EMA.

### Exponential Moving Average (EMA)
- **Formula**: EMA = (Price × k) + (Previous EMA × (1 - k)), where k = 2/(n+1)
- **Characteristic**: Exponential weighting — most responsive to recent prices. Least lag of the three.
- **Common periods**: 12 and 26 (MACD), 21 (short-term trend), 50, 200
- **Advantage**: Reacts faster to price changes. Better for timing.
- **Disadvantage**: More prone to false signals/whipsaws.

### Wilder Smoothing
- **Formula**: WS = ((Previous WS × (n-1)) + Current Value) / n
- **Characteristic**: Similar to EMA but with a different smoothing constant. Used in RSI, ATR, ADX calculations.
- **Note**: Wilder's 14-period is equivalent to a 27-period EMA in responsiveness.

### Response speed ranking: EMA > WMA > SMA (fastest to slowest)

### Moving Average Strategies
- **Single MA**: Price above/below MA determines bias. Works best in trending markets.
- **Dual MA crossover**: Short MA crossing long MA generates signals. Common: 50/200 (golden cross/death cross), 12/26 (MACD basis).
- **Triple MA**: Three MAs (e.g., 10/50/200) for short/medium/long-term alignment. All three aligned = strongest trend signal.
- **MA as dynamic S/R**: In uptrends, MA acts as support on pullbacks. In downtrends, acts as resistance on rallies.

### Moving Average Envelopes and Channels
- **Envelope**: Fixed percentage above/below MA (e.g., ±3%). Creates parallel channel.
- **Keltner Channel**: MA ± multiple of ATR. Adapts to volatility.
- **Bollinger Bands**: MA ± multiple of standard deviation. Expands/contracts with volatility.

---

## 2. Momentum Oscillators (Ch 13)

### RSI — Relative Strength Index
- **Creator**: J. Welles Wilder Jr.
- **Formula**: RSI = 100 - (100 / (1 + RS)), where RS = Avg Gain (n) / Avg Loss (n)
- **Default period**: 14
- **Range**: 0 to 100
- **Standard levels**: Overbought >70, Oversold <30

**Interpretation framework:**
- **Trending market**: RSI stays in the 40-80 range in uptrends, 20-60 in downtrends. Don't sell just because RSI is >70 in a strong uptrend.
- **Range-bound market**: RSI oscillates between overbought/oversold — traditional mean-reversion signals work.
- **Bullish divergence**: Price makes lower low, RSI makes higher low → weakening downside momentum.
- **Bearish divergence**: Price makes higher high, RSI makes lower high → weakening upside momentum.
- **Failure swing (bullish)**: RSI drops below 30, bounces, pulls back but stays above 30, then exceeds prior RSI high.
- **Failure swing (bearish)**: RSI rises above 70, dips, rallies but stays below 70, then drops below prior RSI low.
- **Hidden divergence**: In uptrend, price makes higher low but RSI makes lower low = trend continuation signal. Opposite for downtrends.

### MACD — Moving Average Convergence Divergence
- **Creator**: Gerald Appel
- **Components**:
  - MACD Line = 12-period EMA − 26-period EMA
  - Signal Line = 9-period EMA of MACD Line
  - Histogram = MACD Line − Signal Line
- **Signals**:
  - MACD crosses above signal line = bullish
  - MACD crosses below signal line = bearish
  - Histogram expanding = momentum accelerating
  - Histogram contracting = momentum decelerating
  - Zero line cross: MACD crossing above zero = bullish trend. Below = bearish.
- **Divergence**: Price makes new extreme but MACD doesn't = momentum weakening.
- **Strengths**: Combines trend and momentum. Clear visual signals.
- **Weakness**: Lagging (based on MAs). Can whipsaw in ranges.

### Stochastic Oscillator
- **Creator**: George Lane
- **Formula**: %K = 100 × ((Close - Lowest Low) / (Highest High - Lowest Low)) over n periods
- **%D** = 3-period SMA of %K (slow stochastic)
- **Default period**: 14 for %K, 3 for %D smoothing
- **Range**: 0 to 100
- **Standard levels**: Overbought >80, Oversold <20

**Interpretation:**
- Best suited for RANGE-BOUND markets. Less reliable in strong trends.
- Buy signal: %K crosses above %D below 20 (oversold crossover)
- Sell signal: %K crosses below %D above 80 (overbought crossover)
- Divergence: Same principles as RSI divergence
- In strong trends, stochastic can stay overbought/oversold for extended periods — don't fade the trend

---

## 3. Directional Movement System (Ch 7)

### DMI / ADX
- **Creator**: J. Welles Wilder Jr.
- **Components**:
  - +DI (Positive Directional Indicator): Measures upward price movement
  - -DI (Negative Directional Indicator): Measures downward price movement
  - ADX (Average Directional Index): Measures trend STRENGTH (not direction)
- **Default period**: 14

**Interpretation:**
- +DI > -DI: Bullish pressure dominant
- -DI > +DI: Bearish pressure dominant
- +DI crossing above -DI: Buy signal
- -DI crossing above +DI: Sell signal
- ADX > 25: Market is trending (trade trend-following strategies)
- ADX < 20: Market is ranging (trade mean-reversion strategies)
- ADX rising: Trend strengthening regardless of direction
- ADX falling: Trend weakening, potential transition to range
- ADX turning up from low levels: New trend developing — watch +DI/-DI for direction

**Application**: Use ADX to determine WHICH strategy to apply (trending vs. ranging), then use +DI/-DI or other tools for direction.

---

## 4. Bollinger Bands (Ch 7)

- **Creator**: John Bollinger
- **Formula**:
  - Middle Band = 20-period SMA
  - Upper Band = Middle + (2 × standard deviation of price over 20 periods)
  - Lower Band = Middle - (2 × standard deviation)
- **Default**: 20 periods, 2 standard deviations

### Key Measures
- **BandWidth**: (Upper - Lower) / Middle. Measures relative volatility.
  - Narrow BandWidth = low volatility = "squeeze" → expect expansion
  - Wide BandWidth = high volatility → may precede contraction
- **%b**: (Price - Lower Band) / (Upper - Lower Band). Where price sits within the bands.
  - %b > 1: Price above upper band
  - %b < 0: Price below lower band
  - %b = 0.5: Price at middle band

### Interpretation
- **Squeeze**: Bands narrow to historically low BandWidth. Volatility contraction precedes expansion. Direction of expansion breakout = signal. This is the MOST powerful Bollinger Band signal.
- **Walking the bands**: In strong trends, price hugs the upper (uptrend) or lower (downtrend) band. This is NOT overbought/oversold — it's trend strength.
- **Mean reversion**: In ranges, move from one band toward the other. Price touching upper band → expect reversion toward middle. Touching lower → expect reversion up.
- **W-bottoms and M-tops**: Double bottom (W) where second low holds above lower band = bullish. Double top (M) where second high fails at upper band = bearish.
- Bollinger Bands adapt to volatility — they expand when volatility rises, contract when it falls. This makes them more versatile than fixed envelopes.

---

## 5. Volume Indicators (Ch 12)

### On-Balance Volume (OBV)
- **Creator**: Joe Granville
- **Formula**: If close > prior close: OBV = Prior OBV + Volume. If close < prior close: OBV = Prior OBV - Volume.
- **Interpretation**: Cumulative indicator. Direction of OBV matters, not absolute level.
  - Rising OBV + rising price: Confirmed uptrend
  - Rising OBV + flat price: Accumulation — bullish divergence
  - Falling OBV + rising price: Distribution — bearish divergence
  - Falling OBV + flat price: Distribution in progress

### Accumulation/Distribution Line (A/D)
- **Creator**: Marc Chaikin
- **Formula**: Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low). Money Flow Volume = MFM × Volume. A/D = Cumulative MFV.
- **Interpretation**: Weights volume by where the close falls within the day's range. Close near high = buying pressure. Close near low = selling pressure.
- **Advantage over OBV**: Accounts for close position within range, not just up/down day.

### Money Flow Index (MFI)
- **Formula**: Essentially a volume-weighted RSI
- Typical Price = (High + Low + Close) / 3. Money Flow = TP × Volume. Positive/negative MF based on TP vs. prior TP. MFI = 100 - (100 / (1 + Money Ratio)).
- **Range**: 0 to 100
- **Levels**: Overbought >80, Oversold <20
- **Use**: Like RSI but incorporating volume. Divergence between MFI and price is particularly powerful.

### Chaikin Money Flow (CMF)
- **Period**: Typically 20 or 21
- **Formula**: Sum of Money Flow Volume / Sum of Volume over n periods
- **Range**: -1 to +1
- **Interpretation**: Positive = buying pressure dominant. Negative = selling pressure. Crossing zero line = shift in control.

### Volume Rate of Change
- **Formula**: ((Current Volume - Volume n periods ago) / Volume n periods ago) × 100
- **Use**: Identifies volume surges relative to historical norms. Spike in volume ROC on breakout = confirmation.

### VWAP (Volume Weighted Average Price)
- **Formula**: Cumulative(Price × Volume) / Cumulative(Volume), reset each session
- **Use**: Institutional benchmark. Price above VWAP = buyers in control. Below = sellers. Institutional traders use VWAP to assess execution quality.

---

## 6. Market Breadth Indicators (Ch 18)

### Advance/Decline Line
- **Formula**: Cumulative daily (Advancing issues - Declining issues)
- **Interpretation**: 
  - A/D line rising with price index = healthy trend (broad participation)
  - A/D line diverging from price (index making new high, A/D not) = major bearish warning. This is one of the most important divergence signals in technical analysis.
  - A/D line leading price higher = bullish confirmation

### McClellan Oscillator
- **Formula**: 19-day EMA of (Advances - Declines) minus 39-day EMA of (Advances - Declines)
- **Interpretation**: 
  - Oscillates around zero
  - Positive = bullish breadth momentum
  - Negative = bearish breadth momentum
  - Extreme readings (>+100 or <-100) = overbought/oversold breadth

### McClellan Summation Index
- **Formula**: Cumulative McClellan Oscillator
- **Interpretation**: Long-term breadth trend. Rising = bullish trend in breadth. Falling = bearish.

### New Highs vs. New Lows
- **Interpretation**: 
  - Expanding new highs + rising index = healthy trend
  - Shrinking new highs + rising index = weakening trend (fewer stocks participating)
  - New lows expanding = bearish pressure broadening
  - New highs minus new lows: Positive = bullish breadth. Persistently negative = bear market.

### Percentage of Stocks Above Moving Average
- Common measures: % above 50 MA, % above 200 MA
- **Interpretation**: 
  - >70% above 200 MA: Broadly bullish market
  - <30% above 200 MA: Broadly bearish market
  - Divergence: Index at new high but % above declining = participation narrowing (danger signal)

---

## 7. Volatility Indicators (Chs 29-30)

### True Range / Average True Range (ATR)
- **True Range** = Maximum of: (High - Low), |High - Prior Close|, |Low - Prior Close|
- **ATR** = Wilder smoothing (14-period default) of True Range
- **Use**: Measures volatility. Used for stop placement (1.5-2x ATR), position sizing, and Keltner Channels.
- Rising ATR = volatility expanding. Falling ATR = volatility contracting.

### Historical Volatility (HV)
- **Formula**: Standard deviation of log returns × √(annualisation factor)
  - Equities: × √252
  - Crypto: × √365
- **Use**: What volatility actually WAS over the measurement period. Backward-looking.

### Implied Volatility (IV)
- **Source**: Derived from option prices via pricing models (Black-Scholes or similar)
- **Use**: What the market EXPECTS volatility to be. Forward-looking.
- **IV > HV**: Options are relatively expensive — market expects increased volatility.
- **IV < HV**: Options are relatively cheap — market expects calm.

### VIX (CBOE Volatility Index)
- **Measures**: 30-day expected volatility of S&P 500, derived from options prices.
- **Range context**: 
  - VIX < 15: Low fear / complacency
  - VIX 15-25: Normal range
  - VIX 25-35: Elevated anxiety
  - VIX > 35: High fear / potential panic

**Formulas:**
- 30-day expected move: VIX% / √12
- 1-day expected move: VIX% / √252 (equities)
- Example: VIX = 20 → ±5.77% expected 30-day SPX move

**VIX behaviour:**
- Typically inversely correlated with SPX
- VIX spikes = fear events (sharp selloffs)
- VIX can stay elevated during prolonged uncertainty
- Mean-reverting: Extreme VIX readings tend to revert — this supports contrarian strategies
- VIX as sentiment gauge: extreme high = contrarian bullish, extreme low = contrarian bearish

### Keltner Channels
- **Formula**: Middle = EMA (typically 20). Upper/Lower = EMA ± multiple of ATR (typically 1.5x or 2x).
- **Difference from Bollinger**: Uses ATR instead of standard deviation. Smoother, less reactive to individual price spikes.
- **Bollinger-Keltner Squeeze**: When Bollinger Bands narrow inside Keltner Channels, volatility is extremely compressed → high-probability breakout setup imminent.

---

## 8. Relative Strength (Ch 44)

### Relative Strength Ratio
- **Formula**: Price of asset / Price of benchmark (e.g., stock / S&P 500)
- **Interpretation**:
  - RS line rising: Asset outperforming benchmark
  - RS line falling: Asset underperforming
  - RS line making new high: Strong relative performance
  - RS divergence: Price flat but RS rising = underlying strength

### Uses
- **Sector rotation**: Identify which sectors are leading/lagging
- **Stock selection**: Within a sector, select stocks with strongest relative performance
- **Market regime**: If most sectors' RS is declining vs. market, breadth is narrowing
- **Momentum persistence**: RS tends to persist — outperformers continue outperforming (academic evidence)

### Limitations
- Backward-looking — measures what HAS happened, not what WILL happen
- Can fail at major turning points (leaders become laggards at reversals)
- Doesn't account for risk — a high-RS asset may carry more risk
- Best used as one input alongside trend, pattern, and indicator analysis
