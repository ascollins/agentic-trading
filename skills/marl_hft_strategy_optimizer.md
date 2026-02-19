{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .SFNS-Semibold;}
{\colortbl;\red255\green255\blue255;\red14\green14\blue14;}
{\*\expandedcolortbl;;\cssrgb\c6700\c6700\c6700;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs34 \cf2 diff --git a/skills/marl_hft_strategy_optimizer.md b/skills/marl_hft_strategy_optimizer.md\
index 0000000..1111111 100644\
--- a/skills/marl_hft_strategy_optimizer.md\
+++ b/skills/marl_hft_strategy_optimizer.md\
@@\
 ## 8) Evaluation (must-have)\
@@\
 ### 8.2 Market-quality / microstructure impact metrics\
 Also report:\
 - effective spread\
 - realized spread\
 - price impact\
 - order-to-trade ratio\
\
+### 8.4 Robustness & Overfitting Harness (sequential OOS degradation)\
+Purpose: detect fragile policies that look strong in-sample but decay across forward windows.\
+\
+Required tests:\
+- Walk-forward evaluation (multiple forward windows, not one split)\
+- Sequential OOS degradation tracking:\
+  - freeze a trained policy from each IS window\
+  - evaluate across multiple consecutive OOS windows\
+  - report mean/variance and \'93degradation slope\'94 of key metrics (Sharpe, MDD, impact, OTR)\
+- Split sensitivity:\
+  - run multiple IS/OOS ratios (e.g., 60/40, 70/30, 80/20, 90/10) and compare stability\
+- Hyperparameter sensitivity curves:\
+  - performance vs decision frequency, impact penalty \uc0\u955 2, entropy coefficient, inventory penalty\
+\
+Pass criteria:\
+- OOS stability within acceptable variance bands\
+- No catastrophic degradation in later OOS windows (define thresholds in evaluation_config)\
+\
+Warn criteria:\
+- large dispersion across seeds or across adjacent OOS windows\
+- strong IS edge but weak/negative median OOS edge\
+\
+### 8.5 Optional regime & volatility gating wrapper\
+Add a safety wrapper that can temporarily suspend trading when:\
+- realized micro-volatility spikes above a calibrated threshold; or\
+- spread/impact metrics exceed safety bands\
+Measure: time-in-no-trade, drawdown tail reduction, and recovery time.\
@@\
 ## 10) Quality gates (fail-fast rules)\
 \
 Fail the run if any of these are true:\
 - No transaction cost + impact model implemented (reward becomes meaningless in HFT)\
 - No out-of-sample / walk-forward evaluation\
+- No sequential OOS degradation report (single split only)\
 - No risk limits or kill-switch logic in the environment wrapper\
 - OTR ratio is high and uncontrolled (requires throttling & cancel policy)}