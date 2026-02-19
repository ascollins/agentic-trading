{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 ---\
id: pairs_trading_parameter_optimizer\
name: Pairs Trading Parameter Optimizer (Cointegration + Degradation Analysis)\
version: 1.0.0\
type: skill\
domain: trading_systems\
tags:\
  - crypto\
  - pairs-trading\
  - cointegration\
  - z-score\
  - walk-forward\
  - overfitting-diagnostics\
  - genetic-algorithms\
  - nested-cross-validation\
owner: you\
status: draft\
references:\
  - "Palazzi (2025), Trading Games: Beating Passive Strategies in the Bullish Crypto Market, Journal of Futures Markets"\
---\
\
# Pairs Trading Parameter Optimizer (Cointegration + Degradation Analysis)\
\
## 1) What this skill does\
Builds, tunes, and validates a **cointegration-based pairs trading strategy** using **Z-score spread signals**, with a strong focus on **overfitting detection** via **parameter degradation analysis** across sequential out-of-sample windows.\
\
Outputs:\
- cointegrated pair set + hedge ratios\
- optimized parameters (lookback, thresholds, costs, vol filter, trailing stop)\
- walk-forward + degradation diagnostics report\
- deployable signal engine + risk overlays\
\
## 2) When to use\
Use when:\
- You trade **spreads** between cointegrated assets (often crypto spot/perps, equities, futures)\
- Your main degrees of freedom are **lookback window** and **Z-score thresholds**\
- You need robust, explainable strategies with explicit overfitting checks\
\
Do NOT use when:\
- You need microstructure/HFT execution (latency/queue/LOB) \'97 use the MARL HFT skill instead.\
\
## 3) Inputs\
\
### Required\
- price series for candidate universe: `P[t, asset]` (prefer log-prices)\
- transaction cost model (at minimum per-trade cost `c` or fee + slippage approximation)\
- train/eval configuration:\
  - walk-forward windows (IS length, OOS length, step size)\
  - number of sequential OOS windows to track\
- parameter search space:\
  - lookback range `L`\
  - thresholds `ThresholdLong`, `ThresholdShort` (or symmetric `\'b1\uc0\u952 `)\
\
### Optional (recommended)\
- volatility estimation window `Lv` and multiplier `k`\
- trailing stop-loss parameters (volatility-scaled)\
- optimization methods:\
  - grid search\
  - nested cross-validation\
  - genetic algorithm\
- regime labeling method (bull/bear) and regime buckets\
\
## 4) Core methodology\
\
### 4.1 Find cointegrated pairs\
1. For each candidate pair (A, B), estimate cointegrating relationship on IS window:\
   - `log(PA) = \uc0\u945  + \u946  * log(PB) + \u949 `\
2. Define spread:\
   - `S_t = log(PA_t) - (\uc0\u945  + \u946 *log(PB_t))`\
3. Retain pairs passing cointegration tests (and minimum liquidity filters, if available).\
\
### 4.2 Z-score signal generation\
Compute rolling mean and stdev of spread over lookback `L`:\
- `Z_t = (S_t - mean(S_\{t-L:t\})) / std(S_\{t-L:t\})`\
\
Signal:\
- `Signal_t = +1` if `Z_t <= ThresholdLong`\
- `Signal_t = -1` if `Z_t >= ThresholdShort`\
- `Signal_t = 0` otherwise\
\
### 4.3 PnL / returns with costs (baseline)\
Use spread changes with transaction costs penalizing position flips:\
- `\uc0\u916 S_t = S_t - S_\{t-1\}`\
- `R_t = Signal_\{t-1\} * \uc0\u916 S_t - c * |\u916 Signal_t|`\
(where `\uc0\u916 Signal_t = Signal_t - Signal_\{t-1\}`)\
\
### 4.4 Volatility filter (risk gating)\
Estimate rolling volatility `\uc0\u963 _t` over window `Lv` and average volatility `\u963 \u772 `.\
Define:\
- `VolFilter_t = 1` if `\uc0\u963 _t <= k * \u963 \u772 `, else `0`\
Apply:\
- `SignalFiltered_t = Signal_t * VolFilter_t`\
\
### 4.5 Dynamic trailing stop-loss (volatility-aware)\
For each open position:\
- track post-entry spread extreme (max for long, min for short)\
- close position when spread reverses beyond a volatility-scaled stop threshold\
(Implementation detail: stop should adapt using volatility estimates so it widens/narrows with conditions.)\
\
## 5) Optimization & validation\
\
### 5.1 Walk-forward validation (required)\
Run walk-forward:\
- Calibrate parameters on IS window\
- Evaluate on OOS window\
- Slide forward and repeat\
\
Report distribution of OOS results, not just one split.\
\
### 5.2 Parameter degradation analysis (required)\
Track how calibrated parameters hold up over **multiple consecutive OOS windows**:\
- For each calibration, compute performance across sequential forward windows\
- Score robustness by:\
  - mean OOS performance\
  - variance across windows\
  - degradation slope (performance decay vs window index)\
\
Goal: identify parameter *ranges* that generalize, not single-point optima.\
\
### 5.3 Optimization techniques (options)\
- Grid search (baseline)\
- Nested cross-validation (strong overfitting control)\
- Genetic algorithm (explores wider search spaces; must be validated with degradation framework)\
\
## 6) Evaluation outputs (minimum)\
Trading metrics:\
- total return, annualized return, Sharpe, max drawdown, volatility\
- hit rate / win rate, profit factor, average trade return, turnover\
\
Robustness metrics:\
- OOS mean/median Sharpe\
- OOS worst-window drawdown\
- parameter sensitivity curves (performance vs lookback, thresholds)\
- degradation report (sequential OOS)\
\
Risk overlay metrics:\
- time in \'93no-trade\'94 due to vol filter\
- stop-loss trigger frequency and effect on drawdown tails\
\
## 7) Quality gates (fail-fast)\
Fail the run if:\
- No walk-forward evaluation configured\
- Degradation analysis not produced (single split only)\
- Optimization selects a parameter that performs only in-sample but collapses in sequential OOS windows\
- Transaction cost model missing (even coarse)\
\
Warn if:\
- performance is extremely sensitive to tiny changes in lookback/thresholds\
- strategy is dominated by a small number of trades (fragile)\
\
## 8) Implementation notes (recommended repo layout)\
- `pairs/cointegration.py` (tests + \uc0\u946  estimation)\
- `signals/zscore.py`\
- `risk/vol_filter.py`\
- `risk/trailing_stop.py`\
- `opt/grid_search.py`\
- `opt/nested_cv.py`\
- `opt/genetic.py`\
- `eval/walk_forward.py`\
- `eval/degradation.py`\
- `reports/diagnostics.md` (auto-generated)\
\
## 9) Orchestrator contract (invocation template)\
**Goal:** Build robust cointegration pairs strategy with degradation-based validation.\
\
**Instruction template:**\
- Find cointegrated pairs on each IS window; compute spread and Z-score signals.\
- Optimize lookback and thresholds using chosen optimizer.\
- Evaluate using walk-forward; generate parameter degradation analysis across sequential OOS windows.\
- Add volatility filter (k, Lv) and trailing stop-loss; quantify drawdown reduction and robustness impact.\
- Output deployable signal engine + full report.}