import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, ComposedChart,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
  ResponsiveContainer, Cell
} from 'recharts';

// ═══════════════════════════════════════════════════════════════════
// DATA GENERATION
// ═══════════════════════════════════════════════════════════════════

const STRATEGY_COLORS = {
  bb_squeeze: '#a855f7',
  trend_following: '#3b82f6',
  multi_tf_ma: '#06b6d4',
  stochastic_macd: '#f59e0b',
  supply_demand: '#ef4444',
};

const STRATEGY_NAMES = {
  bb_squeeze: 'BB Squeeze',
  trend_following: 'Trend Following',
  multi_tf_ma: 'Multi TF MA',
  stochastic_macd: 'Stochastic MACD',
  supply_demand: 'Supply/Demand',
};

const STRATEGY_DESCRIPTIONS = {
  bb_squeeze: 'Bollinger Band squeeze breakout with Keltner confirmation',
  trend_following: 'Triple MA alignment (21/50/200) with ADX filter',
  multi_tf_ma: 'Elder triple screen — weekly trend, daily pullback, 4H entry',
  stochastic_macd: 'Stochastic crossover confirmed by MACD histogram',
  supply_demand: 'Supply/demand zone identification with order block confirmation',
};

const STRATEGY_PARAMS = {
  bb_squeeze: { bb_period: 20, bb_std: 2.0, kc_period: 20, kc_atr: 1.5, rsi_filter: 50 },
  trend_following: { fast_ma: 21, mid_ma: 50, slow_ma: 200, adx_threshold: 25, atr_stop: 2.0 },
  multi_tf_ma: { weekly_ma: 50, daily_rsi: 14, rsi_oversold: 35, entry_tf: '4h', atr_stop: 1.5 },
  stochastic_macd: { stoch_k: 14, stoch_d: 3, macd_fast: 12, macd_slow: 26, macd_signal: 9 },
  supply_demand: { zone_lookback: 50, min_zone_touches: 2, ob_period: 10, rsi_confirm: true },
};

// Seeded pseudo-random for reproducibility
function seededRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function generateBTCPrices() {
  const rand = seededRandom(42);
  const prices = [];
  let price = 42500;
  const startDate = new Date(2024, 0, 1);

  for (let day = 0; day < 31; day++) {
    for (let hour = 0; hour < 24; hour++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + day);
      date.setHours(hour);

      // BTC Jan 2024: ~42k start, rally to ~49k mid-month, settle ~46k
      const trend = day < 15
        ? price * 0.003 * (rand() - 0.35)
        : price * 0.003 * (rand() - 0.55);
      const vol = price * 0.002 * (rand() - 0.5);
      price = Math.max(41000, Math.min(50000, price + trend + vol));

      prices.push({
        date: date.toISOString(),
        dateStr: `${date.getMonth() + 1}/${date.getDate()}`,
        dayOfMonth: day + 1,
        hour,
        dayOfWeek: date.getDay(),
        price: Math.round(price * 100) / 100,
      });
    }
  }
  return prices;
}

const BTC_PRICES = generateBTCPrices();

function generateTrades(strategyKey, btcPrices) {
  const rand = seededRandom(
    { bb_squeeze: 101, trend_following: 202, multi_tf_ma: 303, stochastic_macd: 404, supply_demand: 505 }[strategyKey]
  );

  const configs = {
    bb_squeeze: { count: 19, winRate: 0.368, avgWinR: 2.8, avgLossR: 0.87 },
    trend_following: { count: 23, winRate: 0.435, avgWinR: 2.14, avgLossR: 1.0 },
    multi_tf_ma: { count: 25, winRate: 0.400, avgWinR: 1.38, avgLossR: 1.0 },
    stochastic_macd: { count: 15, winRate: 0.533, avgWinR: 1.04, avgLossR: 1.0 },
    supply_demand: { count: 39, winRate: 0.154, avgWinR: 1.85, avgLossR: 1.0 },
  };

  const cfg = configs[strategyKey];
  const wins = Math.round(cfg.count * cfg.winRate);
  const losses = cfg.count - wins;

  // Build win/loss sequence
  const outcomes = [];
  let w = wins, l = losses;
  for (let i = 0; i < cfg.count; i++) {
    if (w === 0) { outcomes.push(false); l--; }
    else if (l === 0) { outcomes.push(true); w--; }
    else if (rand() < cfg.winRate) { outcomes.push(true); w--; }
    else { outcomes.push(false); l--; }
  }

  const trades = [];
  const hoursPerTrade = Math.floor((31 * 24) / (cfg.count + 2));
  let priceIdx = Math.floor(rand() * 24);

  for (let i = 0; i < cfg.count; i++) {
    const entryIdx = Math.min(priceIdx, btcPrices.length - 48);
    const entryPrice = btcPrices[entryIdx].price;
    const isWin = outcomes[i];
    const side = rand() > 0.5 ? 'LONG' : 'SHORT';

    const riskPerUnit = entryPrice * 0.015 * (0.5 + rand());
    const stopDistance = riskPerUnit;
    const stopPrice = side === 'LONG'
      ? Math.round((entryPrice - stopDistance) * 100) / 100
      : Math.round((entryPrice + stopDistance) * 100) / 100;

    let rMultiple, exitPrice;
    if (isWin) {
      rMultiple = cfg.avgWinR * (0.5 + rand());
      exitPrice = side === 'LONG'
        ? Math.round((entryPrice + rMultiple * stopDistance) * 100) / 100
        : Math.round((entryPrice - rMultiple * stopDistance) * 100) / 100;
    } else {
      rMultiple = -(cfg.avgLossR * (0.3 + rand() * 0.7));
      exitPrice = side === 'LONG'
        ? Math.round((entryPrice + rMultiple * stopDistance) * 100) / 100
        : Math.round((entryPrice - rMultiple * stopDistance) * 100) / 100;
    }

    const targetDistance = stopDistance * (2 + rand() * 2);
    const targetPrice = side === 'LONG'
      ? Math.round((entryPrice + targetDistance) * 100) / 100
      : Math.round((entryPrice - targetDistance) * 100) / 100;

    const durationHours = Math.max(1, Math.floor(4 + rand() * 36));
    const exitIdx = Math.min(entryIdx + durationHours, btcPrices.length - 1);

    const pnlDollars = ((exitPrice - entryPrice) * (side === 'LONG' ? 1 : -1)) / entryPrice * 10000;
    const pnlPercent = ((exitPrice - entryPrice) * (side === 'LONG' ? 1 : -1)) / entryPrice * 100;
    const confluence = Math.floor(3 + rand() * 9);
    const exitReasons = isWin
      ? ['Target 1 hit', 'Target 2 hit', 'Trailing stop', 'Time exit (profit)']
      : ['Stopped out', 'Stopped out', 'Time exit (loss)', 'Manual close'];
    const exitReason = exitReasons[Math.floor(rand() * exitReasons.length)];

    trades.push({
      id: i + 1,
      date: btcPrices[entryIdx].date,
      dateStr: `Jan ${btcPrices[entryIdx].dayOfMonth}`,
      dayOfMonth: btcPrices[entryIdx].dayOfMonth,
      hour: btcPrices[entryIdx].hour,
      dayOfWeek: btcPrices[entryIdx].dayOfWeek,
      side,
      entry: entryPrice,
      exit: exitPrice,
      stop: stopPrice,
      target: targetPrice,
      rrPlan: Math.round((targetDistance / stopDistance) * 100) / 100,
      rAchieved: Math.round(rMultiple * 100) / 100,
      pnlDollars: Math.round(pnlDollars * 100) / 100,
      pnlPercent: Math.round(pnlPercent * 10000) / 10000,
      duration: `${durationHours}h`,
      durationHours,
      confluence: `${confluence}/11`,
      confluenceNum: confluence,
      exitReason,
      isWin,
    });

    priceIdx = exitIdx + Math.floor(rand() * hoursPerTrade * 0.5) + Math.floor(hoursPerTrade * 0.5);
    if (priceIdx >= btcPrices.length - 48) break;
  }

  return trades;
}

function generateEquityCurve(trades, totalDays = 31) {
  const curve = [];
  let cumPnl = 0;
  let peak = 0;
  let tradeIdx = 0;

  for (let day = 1; day <= totalDays; day++) {
    let dailyPnl = 0;
    while (tradeIdx < trades.length && trades[tradeIdx].dayOfMonth <= day) {
      dailyPnl += trades[tradeIdx].pnlPercent;
      tradeIdx++;
    }
    cumPnl += dailyPnl;
    peak = Math.max(peak, cumPnl);
    const dd = cumPnl - peak;

    curve.push({
      day,
      dateStr: `Jan ${day}`,
      cumPnl: Math.round(cumPnl * 100) / 100,
      dailyPnl: Math.round(dailyPnl * 100) / 100,
      drawdown: Math.round(dd * 100) / 100,
    });
  }
  return curve;
}

function generateBuyHoldCurve(btcPrices, totalDays = 31) {
  const startPrice = btcPrices[0].price;
  const curve = [];
  for (let day = 1; day <= totalDays; day++) {
    const idx = Math.min((day - 1) * 24 + 12, btcPrices.length - 1);
    const pnl = ((btcPrices[idx].price - startPrice) / startPrice) * 100;
    curve.push({
      day,
      dateStr: `Jan ${day}`,
      buyHold: Math.round(pnl * 100) / 100,
    });
  }
  return curve;
}

function generateOptimizationGrid(strategyKey) {
  const rand = seededRandom(
    { bb_squeeze: 1001, trend_following: 1002, multi_tf_ma: 1003, stochastic_macd: 1004, supply_demand: 1005 }[strategyKey]
  );

  const grids = {
    bb_squeeze: {
      param1: { name: 'BB Period', values: [10, 14, 18, 20, 24, 30] },
      param2: { name: 'BB Std Dev', values: [1.5, 1.8, 2.0, 2.2, 2.5, 3.0] },
      optX: 3, optY: 2, baseSharpe: 1.84, basePnl: 3.38,
    },
    trend_following: {
      param1: { name: 'Fast MA', values: [10, 15, 21, 25, 30, 40] },
      param2: { name: 'ADX Threshold', values: [15, 20, 25, 30, 35, 40] },
      optX: 2, optY: 2, baseSharpe: 1.42, basePnl: 2.37,
    },
    multi_tf_ma: {
      param1: { name: 'Weekly MA', values: [20, 30, 40, 50, 60, 80] },
      param2: { name: 'RSI Oversold', values: [25, 30, 35, 40, 45, 50] },
      optX: 3, optY: 2, baseSharpe: 0.61, basePnl: 0.63,
    },
    stochastic_macd: {
      param1: { name: 'Stoch K', values: [5, 8, 11, 14, 18, 21] },
      param2: { name: 'MACD Fast', values: [6, 8, 10, 12, 15, 18] },
      optX: 3, optY: 3, baseSharpe: 0.22, basePnl: 0.14,
    },
    supply_demand: {
      param1: { name: 'Zone Lookback', values: [20, 30, 40, 50, 60, 80] },
      param2: { name: 'OB Period', values: [5, 8, 10, 14, 18, 24] },
      optX: 3, optY: 2, baseSharpe: -1.23, basePnl: -3.57,
    },
  };

  const g = grids[strategyKey];
  const cells = [];

  for (let y = 0; y < 6; y++) {
    for (let x = 0; x < 6; x++) {
      const distX = Math.abs(x - g.optX);
      const distY = Math.abs(y - g.optY);
      const dist = Math.sqrt(distX * distX + distY * distY);

      // Robust strategies have broader plateaus
      const falloff = strategyKey === 'supply_demand' ? 3.0
        : strategyKey === 'stochastic_macd' ? 2.0
        : strategyKey === 'multi_tf_ma' ? 1.5
        : 0.8;

      const sharpe = g.baseSharpe - dist * falloff * (0.3 + rand() * 0.3) + (rand() - 0.5) * 0.2;
      const pnl = g.basePnl - dist * falloff * (0.5 + rand() * 0.4) + (rand() - 0.5) * 0.3;
      const calmar = sharpe * (1.2 + rand() * 0.6);
      const winRate = Math.max(10, Math.min(65, 35 + sharpe * 8 + (rand() - 0.5) * 10));
      const profitFactor = Math.max(0.1, 1 + sharpe * 0.4 + (rand() - 0.5) * 0.3);
      const maxDD = Math.min(-0.2, -1 - dist * 0.8 * (rand() + 0.5));

      cells.push({
        x, y,
        param1: g.param1.values[x],
        param2: g.param2.values[y],
        sharpe: Math.round(sharpe * 100) / 100,
        pnl: Math.round(pnl * 100) / 100,
        calmar: Math.round(calmar * 100) / 100,
        winRate: Math.round(winRate * 10) / 10,
        profitFactor: Math.round(profitFactor * 100) / 100,
        maxDD: Math.round(maxDD * 100) / 100,
        isOptimal: x === g.optX && y === g.optY,
      });
    }
  }

  return { ...g, cells };
}

function generateConvergenceData(strategyKey) {
  const rand = seededRandom(
    { bb_squeeze: 2001, trend_following: 2002, multi_tf_ma: 2003, stochastic_macd: 2004, supply_demand: 2005 }[strategyKey]
  );

  const targets = {
    bb_squeeze: { sharpe: 1.84, pnl: 3.38, maxDD: -1.08, winRate: 36.8 },
    trend_following: { sharpe: 1.42, pnl: 2.37, maxDD: -1.08, winRate: 43.5 },
    multi_tf_ma: { sharpe: 0.61, pnl: 0.63, maxDD: -0.77, winRate: 40.0 },
    stochastic_macd: { sharpe: 0.22, pnl: 0.14, maxDD: -0.50, winRate: 53.3 },
    supply_demand: { sharpe: -1.23, pnl: -3.57, maxDD: -4.21, winRate: 15.4 },
  };

  const t = targets[strategyKey];
  const iterations = 36;
  const data = [];
  let bestSharpe = -5, bestPnl = -10, bestDD = -20, bestWR = 0;

  for (let i = 1; i <= iterations; i++) {
    const progress = i / iterations;
    const currentSharpe = t.sharpe * progress * (0.3 + rand() * 0.7) + (rand() - 0.5) * 0.5;
    const currentPnl = t.pnl * progress * (0.3 + rand() * 0.7) + (rand() - 0.5) * 1.0;
    const currentDD = -Math.abs(t.maxDD) * (1 + (1 - progress) * rand() * 2);
    const currentWR = t.winRate * (0.5 + progress * 0.5) + (rand() - 0.5) * 10;

    bestSharpe = Math.max(bestSharpe, currentSharpe);
    bestPnl = Math.max(bestPnl, currentPnl);
    bestDD = Math.max(bestDD, currentDD);
    bestWR = Math.max(bestWR, currentWR);

    data.push({
      iteration: i,
      currentSharpe: Math.round(currentSharpe * 100) / 100,
      bestSharpe: Math.round(bestSharpe * 100) / 100,
      currentPnl: Math.round(currentPnl * 100) / 100,
      bestPnl: Math.round(bestPnl * 100) / 100,
      currentDD: Math.round(currentDD * 100) / 100,
      bestDD: Math.round(bestDD * 100) / 100,
      currentWR: Math.round(currentWR * 10) / 10,
      bestWR: Math.round(bestWR * 10) / 10,
    });
  }

  return data;
}

function computeRobustness(strategyKey, grid) {
  const profitableCells = grid.cells.filter(c => c.pnl > 0);
  const breadth = Math.round((profitableCells.length / grid.cells.length) * 100);

  const sharpes = profitableCells.map(c => c.sharpe);
  const meanSharpe = sharpes.length > 0 ? sharpes.reduce((a, b) => a + b, 0) / sharpes.length : 0;
  const consistency = sharpes.length > 1
    ? Math.round(Math.sqrt(sharpes.reduce((s, v) => s + (v - meanSharpe) ** 2, 0) / (sharpes.length - 1)) * 100) / 100
    : 0;

  const validationGaps = {
    bb_squeeze: -12, trend_following: -18, multi_tf_ma: -35, stochastic_macd: -42, supply_demand: -68,
  };
  const validationGap = validationGaps[strategyKey];

  const opt = grid.cells.find(c => c.isOptimal);
  const neighbors = grid.cells.filter(c =>
    Math.abs(c.x - grid.optX) <= 1 && Math.abs(c.y - grid.optY) <= 1 && !c.isOptimal
  );
  const neighborSharpes = neighbors.map(c => c.sharpe);
  const sensitivity = neighborSharpes.length > 0
    ? Math.round(Math.abs(opt.sharpe - (neighborSharpes.reduce((a, b) => a + b, 0) / neighborSharpes.length)) * 100) / 100
    : 0;

  // Score: breadth (40%), consistency inv (20%), validation gap inv (20%), sensitivity inv (20%)
  const breadthScore = Math.min(100, breadth * 1.4);
  const consistencyScore = Math.max(0, 100 - consistency * 40);
  const validationScore = Math.max(0, 100 + validationGap * 1.2);
  const sensitivityScore = Math.max(0, 100 - sensitivity * 50);
  const overall = Math.round((breadthScore * 0.4 + consistencyScore * 0.2 + validationScore * 0.2 + sensitivityScore * 0.2));

  let verdict, verdictColor, description;
  if (overall >= 70) {
    verdict = 'ROBUST'; verdictColor = '#22c55e';
    description = 'Broad plateau, low parameter sensitivity, small OOS gap';
  } else if (overall >= 50) {
    verdict = 'ACCEPTABLE'; verdictColor = '#f59e0b';
    description = 'Moderate sensitivity, some optimization risk';
  } else if (overall >= 30) {
    verdict = 'FRAGILE'; verdictColor = '#ef4444';
    description = 'Narrow peak, high sensitivity, significant OOS degradation';
  } else {
    verdict = 'CURVE-FIT'; verdictColor = '#ef4444';
    description = 'Isolated peak, extreme param sensitivity, massive OOS gap';
  }

  return { breadth, consistency, validationGap, sensitivity, overall, verdict, verdictColor, description };
}

function generateRegimeData(strategyKey, trades) {
  const rand = seededRandom(
    { bb_squeeze: 3001, trend_following: 3002, multi_tf_ma: 3003, stochastic_macd: 3004, supply_demand: 3005 }[strategyKey]
  );

  // Assign regimes to trades based on day
  const volRegimes = { low: [], med: [], high: [] };
  const trendRegimes = { withTrend: [], counterTrend: [], range: [] };

  trades.forEach(t => {
    const r1 = rand();
    if (r1 < 0.35) volRegimes.low.push(t);
    else if (r1 < 0.75) volRegimes.med.push(t);
    else volRegimes.high.push(t);

    const r2 = rand();
    if (r2 < 0.55) trendRegimes.withTrend.push(t);
    else if (r2 < 0.8) trendRegimes.counterTrend.push(t);
    else trendRegimes.range.push(t);
  });

  const computeRegime = (regTrades) => {
    if (regTrades.length === 0) return { trades: 0, winRate: 0, avgR: 0, pnl: 0 };
    const w = regTrades.filter(t => t.isWin).length;
    const avgR = regTrades.reduce((s, t) => s + t.rAchieved, 0) / regTrades.length;
    const pnl = regTrades.reduce((s, t) => s + t.pnlPercent, 0);
    return {
      trades: regTrades.length,
      winRate: Math.round((w / regTrades.length) * 1000) / 10,
      avgR: Math.round(avgR * 100) / 100,
      pnl: Math.round(pnl * 100) / 100,
    };
  };

  return {
    volatility: {
      low: computeRegime(volRegimes.low),
      med: computeRegime(volRegimes.med),
      high: computeRegime(volRegimes.high),
    },
    trend: {
      withTrend: computeRegime(trendRegimes.withTrend),
      counterTrend: computeRegime(trendRegimes.counterTrend),
      range: computeRegime(trendRegimes.range),
    },
  };
}

function generateDayHourHeatmap(trades) {
  const grid = {};
  for (let d = 0; d < 7; d++) {
    for (let h = 0; h < 24; h += 4) {
      grid[`${d}-${h}`] = { pnl: 0, count: 0 };
    }
  }
  trades.forEach(t => {
    const hBucket = Math.floor(t.hour / 4) * 4;
    const key = `${t.dayOfWeek}-${hBucket}`;
    if (grid[key]) {
      grid[key].pnl += t.pnlPercent;
      grid[key].count++;
    }
  });
  return grid;
}

// Build all data
const STRATEGY_KEYS = ['bb_squeeze', 'trend_following', 'multi_tf_ma', 'stochastic_macd', 'supply_demand'];

const ALL_TRADES = {};
const ALL_EQUITY = {};
const ALL_OPT_GRIDS = {};
const ALL_CONVERGENCE = {};
const ALL_ROBUSTNESS = {};
const ALL_REGIMES = {};
const ALL_DAY_HOUR = {};

STRATEGY_KEYS.forEach(key => {
  ALL_TRADES[key] = generateTrades(key, BTC_PRICES);
  ALL_EQUITY[key] = generateEquityCurve(ALL_TRADES[key]);
  ALL_OPT_GRIDS[key] = generateOptimizationGrid(key);
  ALL_CONVERGENCE[key] = generateConvergenceData(key);
  ALL_ROBUSTNESS[key] = computeRobustness(key, ALL_OPT_GRIDS[key]);
  ALL_REGIMES[key] = generateRegimeData(key, ALL_TRADES[key]);
  ALL_DAY_HOUR[key] = generateDayHourHeatmap(ALL_TRADES[key]);
});

const BUY_HOLD = generateBuyHoldCurve(BTC_PRICES);

// Strategy stats
const STRATEGY_STATS = {
  bb_squeeze: { sharpe: 1.84, sortino: 2.41, calmar: 3.12, maxDD: -1.08, netPnL: 3.38 },
  trend_following: { sharpe: 1.42, sortino: 1.87, calmar: 2.19, maxDD: -1.08, netPnL: 2.37 },
  multi_tf_ma: { sharpe: 0.61, sortino: 0.78, calmar: 0.82, maxDD: -0.77, netPnL: 0.63 },
  stochastic_macd: { sharpe: 0.22, sortino: 0.31, calmar: 0.28, maxDD: -0.50, netPnL: 0.14 },
  supply_demand: { sharpe: -1.23, sortino: -0.98, calmar: -1.45, maxDD: -4.21, netPnL: -3.57 },
};

function computeLeaderboard() {
  return STRATEGY_KEYS.map(key => {
    const trades = ALL_TRADES[key];
    const stats = STRATEGY_STATS[key];
    const wins = trades.filter(t => t.isWin);
    const losses = trades.filter(t => !t.isWin);
    const winRate = trades.length > 0 ? (wins.length / trades.length) * 100 : 0;
    const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + Math.abs(t.rAchieved), 0) / wins.length : 0;
    const avgLoss = losses.length > 0 ? losses.reduce((s, t) => s + Math.abs(t.rAchieved), 0) / losses.length : 0;
    const avgWL = avgLoss > 0 ? avgWin / avgLoss : avgWin;
    const grossWin = wins.reduce((s, t) => s + Math.abs(t.pnlDollars), 0);
    const grossLoss = losses.reduce((s, t) => s + Math.abs(t.pnlDollars), 0);
    const pf = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? 99 : 0;
    const avgR = trades.reduce((s, t) => s + t.rAchieved, 0) / trades.length;
    const expectancy = trades.reduce((s, t) => s + t.pnlDollars, 0) / trades.length;

    let status;
    if (stats.sharpe > 0.8 && stats.calmar > 1.0 && stats.netPnL > 0 && pf > 1.3) {
      status = 'PASS';
    } else if (stats.netPnL > 0) {
      status = 'WEAK';
    } else {
      status = 'FAIL';
    }

    return {
      key,
      name: STRATEGY_NAMES[key],
      trades: trades.length,
      winRate: Math.round(winRate * 10) / 10,
      avgWL: Math.round(avgWL * 100) / 100,
      pf: Math.round(pf * 100) / 100,
      netPnL: stats.netPnL,
      sharpe: stats.sharpe,
      sortino: stats.sortino,
      calmar: stats.calmar,
      maxDD: stats.maxDD,
      avgR: Math.round(avgR * 100) / 100,
      expectancy: Math.round(expectancy),
      status,
      color: STRATEGY_COLORS[key],
    };
  });
}

const LEADERBOARD = computeLeaderboard();

const TOTAL_TRADES = LEADERBOARD.reduce((s, r) => s + r.trades, 0);

// ═══════════════════════════════════════════════════════════════════
// UTILITY COMPONENTS
// ═══════════════════════════════════════════════════════════════════

function formatPnl(val, decimals = 2) {
  const sign = val >= 0 ? '+' : '';
  return `${sign}${val.toFixed(decimals)}%`;
}

function formatNum(val, decimals = 2) {
  return val.toFixed(decimals);
}

function pnlColor(val) {
  if (val > 0.001) return '#22c55e';
  if (val < -0.001) return '#ef4444';
  return '#e2e2e8';
}

function thresholdColor(metric, value) {
  const thresholds = {
    netPnL: { green: 2, amber: 0 },
    sharpe: { green: 1.5, amber: 0.8 },
    sortino: { green: 2.0, amber: 1.0 },
    calmar: { green: 2.0, amber: 1.0 },
    maxDD: { green: -5, amber: -10 },
    winRate: { green: 45, amber: 35 },
  };
  const t = thresholds[metric];
  if (!t) return '#22c55e';
  if (metric === 'maxDD') {
    if (value > t.green) return '#22c55e';
    if (value > t.amber) return '#f59e0b';
    return '#ef4444';
  }
  if (value > t.green) return '#22c55e';
  if (value > t.amber) return '#f59e0b';
  return '#ef4444';
}

function statusBadge(status) {
  if (status === 'PASS') return <span style={{ color: '#22c55e' }}>✅ PASS</span>;
  if (status === 'WEAK') return <span style={{ color: '#f59e0b' }}>⚠️ WEAK</span>;
  return <span style={{ color: '#ef4444' }}>❌ FAIL</span>;
}

function heatmapCellColor(value, min, max) {
  const range = max - min || 1;
  const ratio = (value - min) / range;
  if (ratio < 0.3) return '#7f1d1d';
  if (ratio < 0.45) return '#1a1a2a';
  if (ratio < 0.6) return '#14532d';
  if (ratio < 0.8) return '#166534';
  return '#22c55e';
}

function heatmapTextColor(bgColor) {
  return '#e2e2e8';
}

// ═══════════════════════════════════════════════════════════════════
// TOOLTIP COMPONENTS
// ═══════════════════════════════════════════════════════════════════

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div style={{
      background: '#12121a', border: '1px solid #2e2e4e', borderRadius: 4,
      padding: '8px 12px', fontSize: 12, fontFamily: "'JetBrains Mono', monospace",
    }}>
      <div style={{ color: '#8888a0', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || '#e2e2e8', display: 'flex', justifyContent: 'space-between', gap: 16 }}>
          <span>{p.name}</span>
          <span>{typeof p.value === 'number' ? p.value.toFixed(2) + '%' : p.value}</span>
        </div>
      ))}
    </div>
  );
};

const DrawdownTooltip = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div style={{
      background: '#12121a', border: '1px solid #2e2e4e', borderRadius: 4,
      padding: '8px 12px', fontSize: 12, fontFamily: "'JetBrains Mono', monospace",
    }}>
      <div style={{ color: '#8888a0', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: '#ef4444' }}>
          Drawdown: {p.value?.toFixed(2)}%
        </div>
      ))}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════
// SECTION COMPONENTS
// ═══════════════════════════════════════════════════════════════════

// ── HEADER ──
function Header() {
  const champion = LEADERBOARD[0];
  const stats = STRATEGY_STATS.bb_squeeze;

  const kpis = [
    { label: 'Net P&L', value: formatPnl(stats.netPnL), metric: 'netPnL', raw: stats.netPnL },
    { label: 'Sharpe', value: formatNum(stats.sharpe), metric: 'sharpe', raw: stats.sharpe },
    { label: 'Sortino', value: formatNum(stats.sortino), metric: 'sortino', raw: stats.sortino },
    { label: 'Calmar', value: formatNum(stats.calmar), metric: 'calmar', raw: stats.calmar },
    { label: 'Max DD', value: formatPnl(stats.maxDD), metric: 'maxDD', raw: stats.maxDD },
    { label: 'Win Rate', value: `${champion.winRate}%`, metric: 'winRate', raw: champion.winRate },
  ];

  return (
    <div style={{
      position: 'sticky', top: 0, zIndex: 50, background: '#0a0a0f',
      borderBottom: '1px solid #1e1e2e',
    }}>
      {/* Main header row */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 20px', fontSize: 13,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: '#e2e2e8', fontFamily: "'Inter', sans-serif" }}>
            CMT Backtest Lab
          </span>
          <span style={{
            background: '#1e1e2e', padding: '3px 10px', borderRadius: 4,
            color: '#f59e0b', fontFamily: "'JetBrains Mono', monospace", fontSize: 12,
          }}>
            BTC/USDT
          </span>
          <span style={{ color: '#8888a0' }}>Jan 1 — Jan 31, 2024</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#8888a0', fontSize: 12 }}>
          <span>{STRATEGY_KEYS.length} strategies tested</span>
          <span style={{ color: '#555570' }}>·</span>
          <span>{TOTAL_TRADES} total trades</span>
          <span style={{ color: '#555570' }}>·</span>
          <span>Best: <span style={{ color: '#a855f7' }}>bb_squeeze</span> <span style={{ color: '#22c55e' }}>(+3.38%)</span></span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {['Export CSV', 'Export JSON', 'Share Report'].map(label => (
            <button key={label} style={{
              background: '#1e1e2e', border: '1px solid #2e2e4e', borderRadius: 4,
              color: '#8888a0', padding: '4px 10px', fontSize: 11, cursor: 'pointer',
              fontFamily: "'Inter', sans-serif",
            }}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* KPI ribbon */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)',
        borderTop: '1px solid #1e1e2e', background: '#0d0d14',
      }}>
        {kpis.map((kpi, i) => (
          <div key={i} style={{
            padding: '8px 16px', textAlign: 'center',
            borderRight: i < 5 ? '1px solid #1e1e2e' : 'none',
          }}>
            <div style={{ fontSize: 10, color: '#555570', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 2, fontFamily: "'Inter', sans-serif" }}>
              {kpi.label}
            </div>
            <div style={{
              fontSize: 16, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
              color: pnlColor(kpi.raw),
            }}>
              {kpi.value}
            </div>
            <div style={{
              width: 6, height: 6, borderRadius: '50%',
              background: thresholdColor(kpi.metric, kpi.raw),
              margin: '4px auto 0',
            }} />
          </div>
        ))}
      </div>
    </div>
  );
}

// ── SECTION 1: LEADERBOARD ──
function StrategyLeaderboard({ sortKey, setSortKey, sortAsc, setSortAsc, hoveredStrategy, setHoveredStrategy, onSelectStrategy }) {
  const sorted = useMemo(() => {
    const arr = [...LEADERBOARD];
    arr.sort((a, b) => sortAsc ? a[sortKey] - b[sortKey] : b[sortKey] - a[sortKey]);
    return arr;
  }, [sortKey, sortAsc]);

  const columns = [
    { key: 'rank', label: '#', width: 36 },
    { key: 'name', label: 'Strategy', width: 140 },
    { key: 'trades', label: 'Trades', width: 60 },
    { key: 'winRate', label: 'Win%', width: 60 },
    { key: 'avgWL', label: 'Avg W/L', width: 70 },
    { key: 'pf', label: 'PF', width: 55 },
    { key: 'netPnL', label: 'Net P&L', width: 75 },
    { key: 'sharpe', label: 'Sharpe', width: 65 },
    { key: 'sortino', label: 'Sortino', width: 65 },
    { key: 'calmar', label: 'Calmar', width: 65 },
    { key: 'maxDD', label: 'Max DD', width: 70 },
    { key: 'avgR', label: 'Avg R', width: 60 },
    { key: 'expectancy', label: 'Expectancy', width: 90 },
    { key: 'status', label: 'Status', width: 80 },
  ];

  const handleSort = (key) => {
    if (key === 'rank' || key === 'name' || key === 'status') return;
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(false); }
  };

  // PnL contribution bar
  const totalAbsPnl = sorted.reduce((s, r) => s + Math.abs(r.netPnL), 0);

  return (
    <div style={{ padding: '16px 20px' }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e2e8', marginBottom: 10, fontFamily: "'Inter', sans-serif" }}>
        Strategy Leaderboard
        <span style={{ fontSize: 11, color: '#555570', marginLeft: 8, fontWeight: 400 }}>
          Sorted by {sortKey} {sortAsc ? '↑' : '↓'}
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }}>
          <thead>
            <tr>
              {columns.map(col => (
                <th key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{
                    padding: '6px 8px', textAlign: col.key === 'name' ? 'left' : 'right',
                    color: sortKey === col.key ? '#3b82f6' : '#555570',
                    borderBottom: '1px solid #1e1e2e', cursor: 'pointer',
                    fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5,
                    fontFamily: "'Inter', sans-serif", fontWeight: 500,
                    minWidth: col.width,
                  }}>
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row, idx) => (
              <tr key={row.key}
                onMouseEnter={() => setHoveredStrategy(row.key)}
                onMouseLeave={() => setHoveredStrategy(null)}
                onClick={() => onSelectStrategy(row.key)}
                style={{
                  cursor: 'pointer',
                  background: hoveredStrategy === row.key ? '#1a1a28' : 'transparent',
                  borderLeft: idx === 0 ? '3px solid #eab308' : '3px solid transparent',
                  transition: 'background 0.15s',
                }}>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: '#555570' }}>{idx + 1}</td>
                <td style={{ padding: '8px 8px', textAlign: 'left' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: row.color, marginRight: 6 }} />
                  <span style={{ color: '#e2e2e8', fontFamily: "'Inter', sans-serif", fontSize: 12 }}>{row.name}</span>
                </td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: '#8888a0' }}>{row.trades}</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: pnlColor(row.winRate - 40) }}>{row.winRate}%</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: '#8888a0' }}>{row.avgWL.toFixed(2)}</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: pnlColor(row.pf - 1) }}>{row.pf.toFixed(2)}</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: pnlColor(row.netPnL), fontWeight: 600 }}>{formatPnl(row.netPnL)}</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: pnlColor(row.sharpe) }}>{row.sharpe.toFixed(2)}</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: pnlColor(row.sortino) }}>{row.sortino.toFixed(2)}</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: pnlColor(row.calmar) }}>{row.calmar.toFixed(2)}</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: pnlColor(row.maxDD + 5) }}>{formatPnl(row.maxDD)}</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: pnlColor(row.avgR) }}>{row.avgR > 0 ? '+' : ''}{row.avgR.toFixed(2)}R</td>
                <td style={{ padding: '8px 8px', textAlign: 'right', color: pnlColor(row.expectancy) }}>
                  {row.expectancy >= 0 ? '+' : ''}${row.expectancy}/trade
                </td>
                <td style={{ padding: '8px 8px', textAlign: 'right' }}>{statusBadge(row.status)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* P&L Contribution Bar */}
      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: 10, color: '#555570', marginBottom: 4, fontFamily: "'Inter', sans-serif", textTransform: 'uppercase', letterSpacing: 0.5 }}>
          Strategy P&L Contribution
        </div>
        <div style={{ display: 'flex', height: 24, borderRadius: 4, overflow: 'hidden', border: '1px solid #1e1e2e' }}>
          {sorted.map(row => {
            const width = Math.max(2, (Math.abs(row.netPnL) / totalAbsPnl) * 100);
            return (
              <div key={row.key} style={{
                width: `${width}%`, background: row.color + '80',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 9, color: '#e2e2e8', overflow: 'hidden', whiteSpace: 'nowrap',
                borderRight: '1px solid #0a0a0f',
              }}>
                {width > 10 && <span>{row.name} {formatPnl(row.netPnL)}</span>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── SECTION 2: EQUITY & DRAWDOWN ──
function EquityDrawdownCharts({ hoveredStrategy, visibleStrategies }) {
  const equityData = useMemo(() => {
    const combined = [];
    for (let day = 1; day <= 31; day++) {
      const point = { day, dateStr: `Jan ${day}` };
      STRATEGY_KEYS.forEach(key => {
        const eq = ALL_EQUITY[key];
        const dp = eq.find(e => e.day === day);
        point[key] = dp ? dp.cumPnl : 0;
      });
      const bh = BUY_HOLD.find(b => b.day === day);
      point.buyHold = bh ? bh.buyHold : 0;
      combined.push(point);
    }
    return combined;
  }, []);

  const drawdownData = useMemo(() => {
    return ALL_EQUITY.bb_squeeze.map(e => ({
      day: e.day,
      dateStr: e.dateStr,
      drawdown: e.drawdown,
    }));
  }, []);

  const maxDDPoint = useMemo(() => {
    let min = 0, minDay = 1;
    drawdownData.forEach(d => {
      if (d.drawdown < min) { min = d.drawdown; minDay = d.day; }
    });
    return { day: minDay, value: min };
  }, [drawdownData]);

  return (
    <div style={{ padding: '0 20px 16px' }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e2e8', marginBottom: 10, fontFamily: "'Inter', sans-serif" }}>
        Equity Curves & Drawdown
      </div>

      {/* Equity Curves */}
      <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: '12px 8px 4px' }}>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={equityData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2a" />
            <XAxis dataKey="dateStr" tick={{ fill: '#555570', fontSize: 10 }} interval={2} />
            <YAxis tick={{ fill: '#555570', fontSize: 10 }} tickFormatter={v => `${v}%`} />
            <Tooltip content={<ChartTooltip />} />
            <ReferenceLine y={0} stroke="#333" strokeDasharray="4 4" />

            {/* Buy & Hold benchmark */}
            <Line type="monotone" dataKey="buyHold" name="Buy & Hold"
              stroke="#06b6d4" strokeDasharray="6 3" strokeWidth={1.5} dot={false}
              hide={!visibleStrategies.includes('buyHold')} />

            {/* Strategy lines */}
            {STRATEGY_KEYS.map(key => (
              <Line key={key} type="monotone" dataKey={key}
                name={STRATEGY_NAMES[key]}
                stroke={STRATEGY_COLORS[key]}
                strokeWidth={key === 'bb_squeeze' ? 2.5 : hoveredStrategy === key ? 2 : 1.5}
                strokeOpacity={hoveredStrategy && hoveredStrategy !== key ? 0.3 : 1}
                dot={false}
                hide={!visibleStrategies.includes(key)} />
            ))}

            <Legend
              wrapperStyle={{ fontSize: 11, fontFamily: "'Inter', sans-serif", paddingTop: 8 }}
              iconType="line" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown chart */}
      <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: '12px 8px 4px', marginTop: 8 }}>
        <div style={{ fontSize: 11, color: '#555570', marginBottom: 4, paddingLeft: 8, fontFamily: "'Inter', sans-serif" }}>
          Champion Drawdown (BB Squeeze)
        </div>
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart data={drawdownData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2a" />
            <XAxis dataKey="dateStr" tick={{ fill: '#555570', fontSize: 10 }} interval={2} />
            <YAxis tick={{ fill: '#555570', fontSize: 10 }} tickFormatter={v => `${v}%`} domain={['auto', 0]} />
            <Tooltip content={<DrawdownTooltip />} />
            <ReferenceLine y={-5} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: '-5%', fill: '#f59e0b', fontSize: 10, position: 'right' }} />
            <ReferenceLine y={-10} stroke="#ef4444" strokeDasharray="4 4" label={{ value: '-10%', fill: '#ef4444', fontSize: 10, position: 'right' }} />
            <Area type="monotone" dataKey="drawdown" fill="#7f1d1d" fillOpacity={0.4} stroke="#ef4444" strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ── SECTION 3: OPTIMIZATION JOURNEY ──
function OptimizationJourney({ selectedOptStrategy, setSelectedOptStrategy, selectedHeatmapMetric, setSelectedHeatmapMetric }) {
  const grid = ALL_OPT_GRIDS[selectedOptStrategy];
  const convergence = ALL_CONVERGENCE[selectedOptStrategy];
  const cells = grid.cells;

  const metricKey = selectedHeatmapMetric;
  const values = cells.map(c => c[metricKey]);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);

  // Robustness scorecards sorted by overall score
  const robustnessSorted = useMemo(() => {
    return STRATEGY_KEYS
      .map(key => ({ key, ...ALL_ROBUSTNESS[key] }))
      .sort((a, b) => b.overall - a.overall);
  }, []);

  return (
    <div style={{ padding: '0 20px 16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: '#e2e2e8', fontFamily: "'Inter', sans-serif" }}>
          Optimization Journey
        </span>
        <select value={selectedOptStrategy} onChange={e => setSelectedOptStrategy(e.target.value)}
          style={{
            background: '#1e1e2e', border: '1px solid #2e2e4e', borderRadius: 4,
            color: '#e2e2e8', padding: '3px 8px', fontSize: 11, fontFamily: "'Inter', sans-serif",
          }}>
          {STRATEGY_KEYS.map(k => (
            <option key={k} value={k}>{STRATEGY_NAMES[k]}</option>
          ))}
        </select>
        <select value={selectedHeatmapMetric} onChange={e => setSelectedHeatmapMetric(e.target.value)}
          style={{
            background: '#1e1e2e', border: '1px solid #2e2e4e', borderRadius: 4,
            color: '#e2e2e8', padding: '3px 8px', fontSize: 11, fontFamily: "'Inter', sans-serif",
          }}>
          {[
            { key: 'sharpe', label: 'Sharpe' },
            { key: 'pnl', label: 'Net P&L' },
            { key: 'calmar', label: 'Calmar' },
            { key: 'winRate', label: 'Win Rate' },
            { key: 'profitFactor', label: 'Profit Factor' },
          ].map(m => (
            <option key={m.key} value={m.key}>{m.label}</option>
          ))}
        </select>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1.5fr 1.5fr', gap: 12 }}>
        {/* 3A — Heatmap */}
        <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12 }}>
          <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 8, fontFamily: "'Inter', sans-serif" }}>
            Parameter Heatmap — {grid.param2.name} vs {grid.param1.name}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Y-axis labels + cells */}
            {[5, 4, 3, 2, 1, 0].map(y => (
              <div key={y} style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <div style={{
                  width: 40, textAlign: 'right', fontSize: 10, color: '#555570',
                  fontFamily: "'JetBrains Mono', monospace", paddingRight: 4,
                }}>
                  {grid.param2.values[y]}
                </div>
                {[0, 1, 2, 3, 4, 5].map(x => {
                  const cell = cells.find(c => c.x === x && c.y === y);
                  const val = cell[metricKey];
                  const bg = heatmapCellColor(val, minVal, maxVal);
                  return (
                    <div key={x}
                      title={`${grid.param1.name}: ${cell.param1}, ${grid.param2.name}: ${cell.param2}\nSharpe: ${cell.sharpe}, P&L: ${cell.pnl}%, Calmar: ${cell.calmar}\nWin%: ${cell.winRate}, PF: ${cell.profitFactor}`}
                      style={{
                        flex: 1, minWidth: 54, height: 36, background: bg,
                        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                        borderRadius: 2, cursor: 'pointer', position: 'relative',
                        border: cell.isOptimal ? '2px solid #eab308' : '1px solid #1e1e2e',
                        transition: 'border 0.15s',
                      }}>
                      <span style={{
                        fontSize: 11, fontWeight: cell.isOptimal ? 700 : 400,
                        fontFamily: "'JetBrains Mono', monospace",
                        color: heatmapTextColor(bg),
                      }}>
                        {cell.isOptimal ? '★' : ''}{metricKey === 'pnl' ? formatPnl(val) : val.toFixed(2)}
                      </span>
                    </div>
                  );
                })}
              </div>
            ))}
            {/* X-axis labels */}
            <div style={{ display: 'flex', gap: 2, marginLeft: 44 }}>
              {grid.param1.values.map((v, i) => (
                <div key={i} style={{
                  flex: 1, minWidth: 54, textAlign: 'center', fontSize: 10,
                  color: '#555570', fontFamily: "'JetBrains Mono', monospace",
                }}>
                  {v}
                </div>
              ))}
            </div>
            <div style={{ textAlign: 'center', fontSize: 10, color: '#555570', marginTop: 2, fontFamily: "'Inter', sans-serif" }}>
              {grid.param1.name} →
            </div>
          </div>

          {/* Robustness label */}
          {(() => {
            const rob = ALL_ROBUSTNESS[selectedOptStrategy];
            return (
              <div style={{
                marginTop: 8, padding: '4px 8px', borderRadius: 4, textAlign: 'center',
                fontSize: 11, fontWeight: 600, fontFamily: "'Inter', sans-serif",
                background: rob.overall >= 70 ? '#166534' : rob.overall >= 40 ? '#78350f' : '#7f1d1d',
                color: rob.verdictColor,
              }}>
                {rob.overall >= 70 ? '✅ ROBUST' : rob.overall >= 40 ? '⚠️ FRAGILE' : '❌ CURVE-FIT'}
                {' — '}{rob.description}
              </div>
            );
          })()}
        </div>

        {/* 3B — Convergence */}
        <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12 }}>
          <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 8, fontFamily: "'Inter', sans-serif" }}>
            Metric Convergence
          </div>
          {[
            { key: 'Sharpe', best: 'bestSharpe', current: 'currentSharpe', color: '#3b82f6' },
            { key: 'Net P&L', best: 'bestPnl', current: 'currentPnl', color: '#22c55e' },
            { key: 'Max DD', best: 'bestDD', current: 'currentDD', color: '#ef4444' },
            { key: 'Win Rate', best: 'bestWR', current: 'currentWR', color: '#f59e0b' },
          ].map(metric => (
            <div key={metric.key} style={{ marginBottom: 4 }}>
              <div style={{ fontSize: 9, color: '#555570', marginBottom: 1, fontFamily: "'Inter', sans-serif" }}>
                {metric.key}: <span style={{ color: metric.color, fontFamily: "'JetBrains Mono', monospace" }}>
                  {convergence[convergence.length - 1][metric.best]}{metric.key === 'Win Rate' ? '%' : metric.key === 'Net P&L' ? '%' : ''}
                </span>
              </div>
              <ResponsiveContainer width="100%" height={50}>
                <ComposedChart data={convergence} margin={{ top: 2, right: 4, left: -20, bottom: 0 }}>
                  <Area type="monotone" dataKey={metric.current} fill={metric.color} fillOpacity={0.08} stroke="none" />
                  <Line type="stepAfter" dataKey={metric.best} stroke={metric.color} strokeWidth={1.5} dot={false} />
                  <XAxis dataKey="iteration" hide />
                  <YAxis hide />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>

        {/* 3C — Robustness Scorecards */}
        <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12, overflowY: 'auto', maxHeight: 440 }}>
          <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 8, fontFamily: "'Inter', sans-serif" }}>
            Robustness Scorecards
          </div>
          {robustnessSorted.map(r => (
            <div key={r.key} style={{
              marginBottom: 8, padding: 8, borderRadius: 4,
              border: `1px solid ${STRATEGY_COLORS[r.key]}30`,
              background: '#0d0d14',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 12, color: STRATEGY_COLORS[r.key], fontWeight: 600, fontFamily: "'Inter', sans-serif" }}>
                  {STRATEGY_NAMES[r.key]}
                </span>
                <span style={{
                  fontSize: 11, fontWeight: 700, color: r.verdictColor,
                  fontFamily: "'JetBrains Mono', monospace",
                }}>
                  {r.overall}/100
                </span>
              </div>
              {/* Progress bar */}
              <div style={{ height: 4, background: '#1e1e2e', borderRadius: 2, marginBottom: 6 }}>
                <div style={{
                  height: '100%', borderRadius: 2, width: `${r.overall}%`,
                  background: r.verdictColor,
                }} />
              </div>
              <div style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: '#8888a0', lineHeight: 1.6 }}>
                <div>Breadth: <span style={{ color: r.breadth > 50 ? '#22c55e' : '#ef4444' }}>{r.breadth}% profitable</span></div>
                <div>Consistency: <span style={{ color: r.consistency < 0.8 ? '#22c55e' : '#ef4444' }}>σ = {r.consistency}</span></div>
                <div>Validation: <span style={{ color: Math.abs(r.validationGap) < 25 ? '#22c55e' : '#ef4444' }}>{r.validationGap}% gap</span></div>
                <div>Sensitivity: <span style={{ color: r.sensitivity < 0.5 ? '#22c55e' : '#ef4444' }}>±{r.sensitivity} Sharpe</span></div>
              </div>
              <div style={{
                marginTop: 4, fontSize: 10, color: r.verdictColor, fontWeight: 600,
                fontFamily: "'Inter', sans-serif",
              }}>
                {r.verdict === 'ROBUST' ? '✅' : r.verdict === 'ACCEPTABLE' ? '⚠️' : '❌'} {r.verdict}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── SECTION 4: STRATEGY DEEP DIVE ──
function StrategyDeepDive({ strategyKey }) {
  const [activeTab, setActiveTab] = useState('trades');
  const trades = ALL_TRADES[strategyKey];
  const equity = ALL_EQUITY[strategyKey];

  const tabs = [
    { key: 'trades', label: 'Trade Log' },
    { key: 'rdist', label: 'R-Distribution' },
    { key: 'monthly', label: 'Monthly Returns' },
    { key: 'streaks', label: 'Win/Loss Streaks' },
  ];

  return (
    <div style={{ padding: '0 20px 16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: STRATEGY_COLORS[strategyKey], fontFamily: "'Inter', sans-serif" }}>
          {STRATEGY_NAMES[strategyKey]} Deep Dive
        </span>
        <span style={{ fontSize: 11, color: '#555570', fontFamily: "'Inter', sans-serif" }}>
          {STRATEGY_DESCRIPTIONS[strategyKey]}
        </span>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 10 }}>
        {tabs.map(tab => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key)}
            style={{
              background: activeTab === tab.key ? '#1e1e2e' : 'transparent',
              border: `1px solid ${activeTab === tab.key ? '#2e2e4e' : '#1e1e2e'}`,
              borderRadius: 4, padding: '4px 12px', fontSize: 11, cursor: 'pointer',
              color: activeTab === tab.key ? '#e2e2e8' : '#555570',
              fontFamily: "'Inter', sans-serif",
            }}>
            {tab.label}
          </button>
        ))}
      </div>

      <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12 }}>
        {activeTab === 'trades' && <TradeLog trades={trades} />}
        {activeTab === 'rdist' && <RDistribution trades={trades} strategyKey={strategyKey} />}
        {activeTab === 'monthly' && <MonthlyReturns equity={equity} />}
        {activeTab === 'streaks' && <WinLossStreaks trades={trades} />}
      </div>
    </div>
  );
}

function TradeLog({ trades }) {
  const [sortCol, setSortCol] = useState(null);
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = useMemo(() => {
    if (!sortCol) return trades;
    const arr = [...trades];
    arr.sort((a, b) => {
      const va = a[sortCol], vb = b[sortCol];
      if (typeof va === 'number') return sortAsc ? va - vb : vb - va;
      return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });
    return arr;
  }, [trades, sortCol, sortAsc]);

  const handleSort = (col) => {
    if (sortCol === col) setSortAsc(!sortAsc);
    else { setSortCol(col); setSortAsc(true); }
  };

  const cols = [
    { key: 'id', label: '#' },
    { key: 'dateStr', label: 'Date' },
    { key: 'side', label: 'Side' },
    { key: 'entry', label: 'Entry' },
    { key: 'exit', label: 'Exit' },
    { key: 'stop', label: 'Stop' },
    { key: 'target', label: 'Target' },
    { key: 'rrPlan', label: 'R:R Plan' },
    { key: 'rAchieved', label: 'R Achieved' },
    { key: 'pnlDollars', label: 'P&L' },
    { key: 'duration', label: 'Duration' },
    { key: 'confluence', label: 'Confluence' },
    { key: 'exitReason', label: 'Exit Reason' },
  ];

  return (
    <div style={{ overflowX: 'auto', maxHeight: 360, overflowY: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
        <thead>
          <tr>
            {cols.map(col => (
              <th key={col.key} onClick={() => handleSort(col.key)}
                style={{
                  padding: '4px 6px', textAlign: 'right', color: '#555570', cursor: 'pointer',
                  borderBottom: '1px solid #1e1e2e', fontSize: 9, textTransform: 'uppercase',
                  fontFamily: "'Inter', sans-serif", position: 'sticky', top: 0, background: '#12121a',
                }}>
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map(t => (
            <tr key={t.id} style={{ borderBottom: '1px solid #1a1a2a' }}>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#555570' }}>{t.id}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#8888a0' }}>{t.dateStr}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: t.side === 'LONG' ? '#22c55e' : '#ef4444' }}>{t.side}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#e2e2e8' }}>{t.entry.toFixed(0)}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#e2e2e8' }}>{t.exit.toFixed(0)}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#ef4444' }}>{t.stop.toFixed(0)}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#22c55e' }}>{t.target.toFixed(0)}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#8888a0' }}>{t.rrPlan.toFixed(2)}:1</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: pnlColor(t.rAchieved), fontWeight: 600 }}>
                {t.rAchieved > 0 ? '+' : ''}{t.rAchieved.toFixed(2)}R
              </td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: pnlColor(t.pnlDollars), fontWeight: 600 }}>
                {t.pnlDollars > 0 ? '+' : ''}${t.pnlDollars.toFixed(0)}
              </td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#8888a0' }}>{t.duration}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#8888a0' }}>{t.confluence}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: t.isWin ? '#22c55e' : '#ef4444', fontFamily: "'Inter', sans-serif", fontSize: 10 }}>
                {t.exitReason}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RDistribution({ trades, strategyKey }) {
  const buckets = useMemo(() => {
    const b = {};
    for (let r = -3; r <= 5; r += 0.5) b[r] = 0;
    trades.forEach(t => {
      let bucket = Math.round(t.rAchieved * 2) / 2;
      bucket = Math.max(-3, Math.min(5, bucket));
      if (b[bucket] !== undefined) b[bucket]++;
    });
    return Object.entries(b).map(([r, count]) => ({
      r: parseFloat(r),
      label: `${parseFloat(r) >= 0 ? '+' : ''}${parseFloat(r).toFixed(1)}R`,
      count,
      fill: parseFloat(r) >= 0 ? '#22c55e' : '#ef4444',
    }));
  }, [trades]);

  const meanR = trades.reduce((s, t) => s + t.rAchieved, 0) / trades.length;
  const medianR = [...trades].sort((a, b) => a.rAchieved - b.rAchieved)[Math.floor(trades.length / 2)]?.rAchieved || 0;
  const n = trades.length;
  const mu = meanR;
  const skewNum = trades.reduce((s, t) => s + Math.pow(t.rAchieved - mu, 3), 0) / n;
  const sigma3 = Math.pow(trades.reduce((s, t) => s + Math.pow(t.rAchieved - mu, 2), 0) / n, 1.5);
  const skewness = sigma3 > 0 ? skewNum / sigma3 : 0;

  return (
    <div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={buckets} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2a" />
          <XAxis dataKey="label" tick={{ fill: '#555570', fontSize: 9 }} />
          <YAxis tick={{ fill: '#555570', fontSize: 10 }} allowDecimals={false} />
          <ReferenceLine x="+0.0R" stroke="#555570" strokeDasharray="4 4" />
          <Bar dataKey="count" radius={[2, 2, 0, 0]}>
            {buckets.map((b, i) => (
              <Cell key={i} fill={b.fill} fillOpacity={0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{
        display: 'flex', gap: 20, justifyContent: 'center', marginTop: 8,
        fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: '#8888a0',
      }}>
        <span>Mean R: <span style={{ color: pnlColor(meanR) }}>{meanR > 0 ? '+' : ''}{meanR.toFixed(2)}</span></span>
        <span>Median R: <span style={{ color: pnlColor(medianR) }}>{medianR > 0 ? '+' : ''}{medianR.toFixed(2)}</span></span>
        <span>Skew: <span style={{ color: skewness > 0.5 ? '#22c55e' : skewness < -0.5 ? '#ef4444' : '#8888a0' }}>
          {skewness > 0 ? '+' : ''}{skewness.toFixed(2)} ({skewness > 0.3 ? 'right' : skewness < -0.3 ? 'left' : 'symmetric'})
        </span></span>
      </div>
      <div style={{
        marginTop: 8, padding: '6px 12px', background: '#0d0d14', borderRadius: 4,
        fontSize: 10, color: '#555570', fontFamily: "'Inter', sans-serif", textAlign: 'center',
      }}>
        {skewness > 0.5 ? 'Classic trend-following profile — frequent small losses offset by occasional large wins.'
          : skewness < -0.5 ? 'Left-skewed profile — frequent small wins with occasional large losses. Risk of tail events.'
          : 'Symmetric distribution — balanced win/loss sizes.'}
      </div>
    </div>
  );
}

function MonthlyReturns({ equity }) {
  // Build weekly grid for January
  const weeks = [[], [], [], [], []];
  equity.forEach(e => {
    const date = new Date(2024, 0, e.day);
    const dayOfWeek = date.getDay();
    const weekNum = Math.floor((e.day + (new Date(2024, 0, 1).getDay()) - 1) / 7);
    if (weekNum < 5) {
      weeks[weekNum][dayOfWeek] = e.dailyPnl;
    }
  });

  const dayLabels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  return (
    <div>
      <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 8, fontFamily: "'Inter', sans-serif" }}>
        Daily Returns — January 2024
      </div>
      <table style={{ borderCollapse: 'collapse', fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
        <thead>
          <tr>
            <th style={{ padding: '4px 8px', color: '#555570', fontSize: 9, fontFamily: "'Inter', sans-serif" }}></th>
            {dayLabels.map(d => (
              <th key={d} style={{
                padding: '4px 12px', color: '#555570', fontSize: 9, textAlign: 'center',
                fontFamily: "'Inter', sans-serif",
              }}>{d}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {weeks.map((week, wi) => (
            <tr key={wi}>
              <td style={{ padding: '4px 8px', color: '#555570', fontSize: 9, fontFamily: "'Inter', sans-serif" }}>
                W{wi + 1}
              </td>
              {dayLabels.map((_, di) => {
                const val = week[di];
                if (val === undefined) {
                  return <td key={di} style={{ padding: '4px 12px', textAlign: 'center', color: '#333' }}>—</td>;
                }
                const absVal = Math.abs(val);
                const bg = val > 0.3 ? '#166534' : val > 0.05 ? '#14532d60' : val < -0.3 ? '#7f1d1d' : val < -0.05 ? '#7f1d1d60' : 'transparent';
                return (
                  <td key={di} style={{
                    padding: '4px 12px', textAlign: 'center', background: bg,
                    color: pnlColor(val), borderRadius: 2,
                    border: '1px solid #1a1a2a',
                  }}>
                    {formatPnl(val)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WinLossStreaks({ trades }) {
  // Build streaks
  const streaks = [];
  let current = { isWin: trades[0]?.isWin, count: 0, startIdx: 0 };
  trades.forEach((t, i) => {
    if (t.isWin === current.isWin) {
      current.count++;
    } else {
      streaks.push({ ...current });
      current = { isWin: t.isWin, count: 1, startIdx: i };
    }
  });
  streaks.push({ ...current });

  const longestWin = Math.max(...streaks.filter(s => s.isWin).map(s => s.count), 0);
  const longestLoss = Math.max(...streaks.filter(s => !s.isWin).map(s => s.count), 0);

  // Runs test approximation
  const n1 = trades.filter(t => t.isWin).length;
  const n2 = trades.filter(t => !t.isWin).length;
  const runs = streaks.length;
  const expectedRuns = 1 + (2 * n1 * n2) / (n1 + n2);
  const pValue = Math.min(0.99, Math.max(0.01, 0.5 - Math.abs(runs - expectedRuns) / (runs + 1) * 0.5));

  return (
    <div>
      <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 8, fontFamily: "'Inter', sans-serif" }}>
        Win/Loss Streak Analysis
      </div>

      {/* Visual streak bar */}
      <div style={{ display: 'flex', gap: 2, marginBottom: 12, flexWrap: 'wrap' }}>
        {streaks.map((s, i) => (
          <div key={i} style={{ display: 'flex', gap: 1 }}>
            {Array.from({ length: s.count }).map((_, j) => (
              <div key={j} style={{
                width: 14, height: 20, borderRadius: 2,
                background: s.isWin ? '#22c55e' : '#ef4444',
                opacity: 0.6 + (s.count > 2 ? 0.2 : 0),
              }} />
            ))}
          </div>
        ))}
      </div>

      {/* Sequence text */}
      <div style={{
        fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#8888a0',
        marginBottom: 12, letterSpacing: 1,
      }}>
        {trades.map((t, i) => (
          <span key={i} style={{ color: t.isWin ? '#22c55e' : '#ef4444' }}>
            {t.isWin ? 'W' : 'L'}{' '}
          </span>
        ))}
      </div>

      {/* Stats */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12,
        fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
      }}>
        <div style={{ background: '#0d0d14', padding: 8, borderRadius: 4, textAlign: 'center' }}>
          <div style={{ color: '#555570', fontSize: 9, marginBottom: 2, fontFamily: "'Inter', sans-serif" }}>Longest Win</div>
          <div style={{ color: '#22c55e', fontSize: 18, fontWeight: 700 }}>{longestWin}</div>
        </div>
        <div style={{ background: '#0d0d14', padding: 8, borderRadius: 4, textAlign: 'center' }}>
          <div style={{ color: '#555570', fontSize: 9, marginBottom: 2, fontFamily: "'Inter', sans-serif" }}>Longest Loss</div>
          <div style={{ color: '#ef4444', fontSize: 18, fontWeight: 700 }}>{longestLoss}</div>
        </div>
        <div style={{ background: '#0d0d14', padding: 8, borderRadius: 4, textAlign: 'center' }}>
          <div style={{ color: '#555570', fontSize: 9, marginBottom: 2, fontFamily: "'Inter', sans-serif" }}>Runs Test p-value</div>
          <div style={{ color: pValue > 0.05 ? '#22c55e' : '#ef4444', fontSize: 18, fontWeight: 700 }}>{pValue.toFixed(2)}</div>
          <div style={{ color: '#555570', fontSize: 9, fontFamily: "'Inter', sans-serif" }}>
            {pValue > 0.05 ? 'random — no clustering' : 'non-random clustering'}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── SECTION 5: RISK ANALYSIS ──
function RiskAnalysis() {
  const champion = 'bb_squeeze';
  const equity = ALL_EQUITY[champion];
  const trades = ALL_TRADES[champion];

  // Drawdown analysis
  const drawdowns = [];
  let ddStart = null, ddDepth = 0;
  equity.forEach((e, i) => {
    if (e.drawdown < -0.01) {
      if (!ddStart) ddStart = i;
      ddDepth = Math.min(ddDepth, e.drawdown);
    } else if (ddStart !== null) {
      drawdowns.push({ start: ddStart, end: i, depth: ddDepth, duration: i - ddStart });
      ddStart = null; ddDepth = 0;
    }
  });
  if (ddStart !== null) drawdowns.push({ start: ddStart, end: equity.length - 1, depth: ddDepth, duration: equity.length - 1 - ddStart });

  const maxDD = Math.min(...equity.map(e => e.drawdown));
  const avgDD = drawdowns.length > 0 ? drawdowns.reduce((s, d) => s + d.depth, 0) / drawdowns.length : 0;
  const avgDDDuration = drawdowns.length > 0 ? drawdowns.reduce((s, d) => s + d.duration, 0) / drawdowns.length : 0;

  // Position sizing simulation
  const sizingData = useMemo(() => {
    const data = [];
    let cum1 = 0, cum2 = 0, cumK = 0;
    let peak1 = 0, peak2 = 0, peakK = 0;
    let maxDD1 = 0, maxDD2 = 0, maxDDK = 0;

    equity.forEach(e => {
      cum1 += e.dailyPnl * 0.5;
      cum2 += e.dailyPnl;
      cumK += e.dailyPnl * 1.5;
      peak1 = Math.max(peak1, cum1); peak2 = Math.max(peak2, cum2); peakK = Math.max(peakK, cumK);
      maxDD1 = Math.min(maxDD1, cum1 - peak1);
      maxDD2 = Math.min(maxDD2, cum2 - peak2);
      maxDDK = Math.min(maxDDK, cumK - peakK);

      data.push({
        day: e.day,
        dateStr: e.dateStr,
        conservative: Math.round(cum1 * 100) / 100,
        standard: Math.round(cum2 * 100) / 100,
        aggressive: Math.round(cumK * 100) / 100,
      });
    });

    return {
      data,
      stats: {
        conservative: { ret: data[data.length - 1]?.conservative || 0, maxDD: Math.round(maxDD1 * 100) / 100 },
        standard: { ret: data[data.length - 1]?.standard || 0, maxDD: Math.round(maxDD2 * 100) / 100 },
        aggressive: { ret: data[data.length - 1]?.aggressive || 0, maxDD: Math.round(maxDDK * 100) / 100 },
      },
    };
  }, [equity]);

  // Risk of Ruin
  const winRate = trades.filter(t => t.isWin).length / trades.length;
  const wins = trades.filter(t => t.isWin);
  const losses = trades.filter(t => !t.isWin);
  const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + Math.abs(t.rAchieved), 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((s, t) => s + Math.abs(t.rAchieved), 0) / losses.length : 0;
  const wlRatio = avgLoss > 0 ? avgWin / avgLoss : 99;

  // Simplified RoR calculation
  const a = avgLoss > 0 ? (1 - winRate) / (winRate * wlRatio) : 0;
  const fraction = 0.02;
  const riskOfRuin = Math.max(0, Math.min(100, Math.pow(a, 1 / fraction) * 100 * 0.01));

  const rorAngle = Math.min(180, riskOfRuin * 1.8);

  return (
    <div style={{ padding: '0 20px 16px' }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e2e8', marginBottom: 10, fontFamily: "'Inter', sans-serif" }}>
        Risk Analysis <span style={{ fontSize: 11, color: '#555570', fontWeight: 400 }}>— BB Squeeze (Champion)</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr 1fr', gap: 12 }}>
        {/* 5A — Drawdown Analysis */}
        <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12 }}>
          <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 8, fontFamily: "'Inter', sans-serif" }}>
            Drawdown Analysis
          </div>
          {[
            { label: 'Max Drawdown', value: `${maxDD.toFixed(2)}%`, color: maxDD > -5 ? '#22c55e' : '#ef4444' },
            { label: 'Avg DD Depth', value: `${avgDD.toFixed(2)}%`, color: '#f59e0b' },
            { label: 'Avg DD Duration', value: `${avgDDDuration.toFixed(1)} days`, color: '#8888a0' },
            { label: 'DD Events > 1%', value: `${drawdowns.filter(d => d.depth < -1).length}`, color: '#8888a0' },
            { label: 'Current DD', value: `${equity[equity.length - 1]?.drawdown.toFixed(2) || 0}%`, color: '#22c55e' },
          ].map((item, i) => (
            <div key={i} style={{
              display: 'flex', justifyContent: 'space-between', padding: '4px 0',
              borderBottom: i < 4 ? '1px solid #1a1a2a' : 'none',
            }}>
              <span style={{ fontSize: 11, color: '#555570', fontFamily: "'Inter', sans-serif" }}>{item.label}</span>
              <span style={{ fontSize: 11, color: item.color, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                {item.value}
              </span>
            </div>
          ))}

          {/* Mini histogram of DD depths */}
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 9, color: '#555570', marginBottom: 4, fontFamily: "'Inter', sans-serif" }}>DD Depth Distribution</div>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 40 }}>
              {[0, -0.5, -1, -1.5, -2, -3, -4].map((threshold, i) => {
                const count = drawdowns.filter(d => d.depth >= threshold - 0.5 && d.depth < threshold).length;
                return (
                  <div key={i} style={{
                    flex: 1, height: Math.max(2, count * 15),
                    background: threshold > -2 ? '#166534' : threshold > -3 ? '#78350f' : '#7f1d1d',
                    borderRadius: '2px 2px 0 0',
                  }} />
                );
              })}
            </div>
          </div>
        </div>

        {/* 5B — Position Sizing */}
        <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12 }}>
          <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 8, fontFamily: "'Inter', sans-serif" }}>
            Position Sizing Simulation
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={sizingData.data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2a" />
              <XAxis dataKey="dateStr" tick={{ fill: '#555570', fontSize: 9 }} interval={4} />
              <YAxis tick={{ fill: '#555570', fontSize: 9 }} tickFormatter={v => `${v}%`} />
              <Tooltip content={<ChartTooltip />} />
              <Line type="monotone" dataKey="conservative" name="1% Risk" stroke="#06b6d4" strokeWidth={1} dot={false} />
              <Line type="monotone" dataKey="standard" name="2% Risk" stroke="#22c55e" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="aggressive" name="Half-Kelly" stroke="#f59e0b" strokeWidth={1} dot={false} />
              <Legend wrapperStyle={{ fontSize: 10, fontFamily: "'Inter', sans-serif" }} />
            </LineChart>
          </ResponsiveContainer>

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10, fontFamily: "'JetBrains Mono', monospace", marginTop: 8 }}>
            <thead>
              <tr>
                {['Sizing', 'Return', 'Max DD', 'Calmar', 'RoR'].map(h => (
                  <th key={h} style={{ padding: '3px 6px', color: '#555570', textAlign: 'right', borderBottom: '1px solid #1e1e2e', fontSize: 9, fontFamily: "'Inter', sans-serif" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { name: '1% fixed', ...sizingData.stats.conservative, calmar: 3.15, ror: '0.001%' },
                { name: '2% fixed', ...sizingData.stats.standard, calmar: 3.15, ror: '0.012%' },
                { name: 'Half-Kelly', ...sizingData.stats.aggressive, calmar: 2.70, ror: '0.089%' },
              ].map(row => (
                <tr key={row.name}>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: '#8888a0', fontFamily: "'Inter', sans-serif" }}>{row.name}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: '#22c55e' }}>{formatPnl(row.ret)}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: '#ef4444' }}>{row.maxDD.toFixed(2)}%</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: '#8888a0' }}>{row.calmar.toFixed(2)}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: '#22c55e' }}>{row.ror}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* 5C — Risk of Ruin */}
        <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 8, fontFamily: "'Inter', sans-serif" }}>
            Risk of Ruin
          </div>

          {/* Gauge */}
          <div style={{ position: 'relative', width: 140, height: 80, margin: '0 auto 12px' }}>
            <svg viewBox="0 0 140 80" style={{ width: '100%', height: '100%' }}>
              {/* Background arc */}
              <path d="M 10 75 A 60 60 0 0 1 130 75" fill="none" stroke="#1e1e2e" strokeWidth="10" strokeLinecap="round" />
              {/* Green zone (0-1%) */}
              <path d="M 10 75 A 60 60 0 0 1 50 20" fill="none" stroke="#166534" strokeWidth="10" strokeLinecap="round" />
              {/* Amber zone (1-3%) */}
              <path d="M 50 20 A 60 60 0 0 1 90 20" fill="none" stroke="#78350f" strokeWidth="10" strokeLinecap="round" />
              {/* Red zone (3%+) */}
              <path d="M 90 20 A 60 60 0 0 1 130 75" fill="none" stroke="#7f1d1d" strokeWidth="10" strokeLinecap="round" />
              {/* Needle */}
              <line x1="70" y1="75" x2={70 + 50 * Math.cos(Math.PI - (rorAngle * Math.PI / 180))} y2={75 - 50 * Math.sin(Math.PI - (rorAngle * Math.PI / 180))}
                stroke="#e2e2e8" strokeWidth="2" />
              <circle cx="70" cy="75" r="4" fill="#e2e2e8" />
            </svg>
          </div>

          <div style={{ fontSize: 24, fontWeight: 700, color: '#22c55e', fontFamily: "'JetBrains Mono', monospace" }}>
            {riskOfRuin.toFixed(3)}%
          </div>
          <div style={{ fontSize: 9, color: '#555570', marginTop: 4, fontFamily: "'Inter', sans-serif" }}>
            Risk of Ruin (20% drawdown threshold)
          </div>

          <div style={{
            marginTop: 12, textAlign: 'left', fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: '#8888a0',
            lineHeight: 1.8,
          }}>
            <div>Win Rate: <span style={{ color: '#e2e2e8' }}>{(winRate * 100).toFixed(1)}%</span></div>
            <div>Avg W/L: <span style={{ color: '#e2e2e8' }}>{wlRatio.toFixed(2)}</span></div>
            <div>Fraction: <span style={{ color: '#e2e2e8' }}>2.0%</span></div>
          </div>

          <div style={{
            marginTop: 8, padding: '4px 8px', background: '#0d0d14', borderRadius: 4,
            fontSize: 9, color: '#555570', fontFamily: "'Inter', sans-serif",
          }}>
            If win rate drops to {Math.max(15, (winRate * 100 - 10)).toFixed(0)}%,
            RoR increases to {(riskOfRuin * 8).toFixed(2)}%
          </div>
        </div>
      </div>
    </div>
  );
}

// ── SECTION 6: REGIME BREAKDOWN ──
function RegimeBreakdown({ selectedStrategy }) {
  const key = selectedStrategy || 'bb_squeeze';
  const regime = ALL_REGIMES[key];
  const dayHour = ALL_DAY_HOUR[key];

  const RegimeTable = ({ title, rows }) => (
    <div>
      <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 6, fontFamily: "'Inter', sans-serif" }}>
        {title}
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
        <thead>
          <tr>
            {['Regime', 'Trades', 'Win%', 'Avg R', 'P&L'].map(h => (
              <th key={h} style={{
                padding: '4px 6px', textAlign: 'right', color: '#555570', borderBottom: '1px solid #1e1e2e',
                fontSize: 9, fontFamily: "'Inter', sans-serif",
              }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.label}>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#8888a0', fontFamily: "'Inter', sans-serif", fontSize: 10 }}>{r.label}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: '#8888a0' }}>{r.trades}</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: pnlColor(r.winRate - 40) }}>{r.winRate.toFixed(1)}%</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: pnlColor(r.avgR) }}>{r.avgR > 0 ? '+' : ''}{r.avgR.toFixed(2)}R</td>
              <td style={{ padding: '4px 6px', textAlign: 'right', color: pnlColor(r.pnl), fontWeight: 600 }}>{formatPnl(r.pnl)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  // Day/Hour heatmap
  const dayLabels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const hourLabels = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24'];
  const allPnls = Object.values(dayHour).map(v => v.pnl).filter(p => p !== 0);
  const maxAbsPnl = Math.max(0.01, ...allPnls.map(p => Math.abs(p)));

  return (
    <div style={{ padding: '0 20px 16px' }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e2e8', marginBottom: 10, fontFamily: "'Inter', sans-serif" }}>
        Regime Breakdown <span style={{ fontSize: 11, color: STRATEGY_COLORS[key], fontWeight: 400 }}>— {STRATEGY_NAMES[key]}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        {/* 6A — Volatility */}
        <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12 }}>
          <RegimeTable title="Performance by Volatility Regime" rows={[
            { label: 'Low Vol (ADX<20)', ...regime.volatility.low },
            { label: 'Med Vol (20-35)', ...regime.volatility.med },
            { label: 'High Vol (ADX>35)', ...regime.volatility.high },
          ]} />
        </div>

        {/* 6B — Trend */}
        <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12 }}>
          <RegimeTable title="Performance by Trend Direction" rows={[
            { label: 'With Trend', ...regime.trend.withTrend },
            { label: 'Counter-Trend', ...regime.trend.counterTrend },
            { label: 'Range/No Trend', ...regime.trend.range },
          ]} />
        </div>

        {/* 6C — Day/Hour Heatmap */}
        <div style={{ background: '#12121a', borderRadius: 6, border: '1px solid #1e1e2e', padding: 12 }}>
          <div style={{ fontSize: 11, color: '#8888a0', marginBottom: 6, fontFamily: "'Inter', sans-serif" }}>
            P&L by Day/Hour
          </div>
          <table style={{ borderCollapse: 'collapse', fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }}>
            <thead>
              <tr>
                <th style={{ padding: '2px 4px' }}></th>
                {hourLabels.map(h => (
                  <th key={h} style={{ padding: '2px 4px', color: '#555570', fontSize: 8, fontFamily: "'Inter', sans-serif" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dayLabels.map((day, di) => (
                <tr key={di}>
                  <td style={{ padding: '2px 4px', color: '#555570', fontSize: 8, fontFamily: "'Inter', sans-serif" }}>{day}</td>
                  {[0, 4, 8, 12, 16, 20].map(h => {
                    const cell = dayHour[`${di}-${h}`];
                    if (!cell || cell.count === 0) {
                      return <td key={h} style={{ padding: '2px 6px', textAlign: 'center', color: '#333' }}>—</td>;
                    }
                    const ratio = cell.pnl / maxAbsPnl;
                    const bg = ratio > 0.3 ? '#166534' : ratio > 0.05 ? '#14532d60' : ratio < -0.3 ? '#7f1d1d' : ratio < -0.05 ? '#7f1d1d60' : 'transparent';
                    return (
                      <td key={h} style={{
                        padding: '2px 6px', textAlign: 'center', background: bg,
                        color: pnlColor(cell.pnl), border: '1px solid #1a1a2a',
                      }}>
                        {cell.pnl > 0 ? '+' : ''}{cell.pnl.toFixed(1)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// MAIN DASHBOARD
// ═══════════════════════════════════════════════════════════════════

export default function CMTBacktestDashboard() {
  const [sortKey, setSortKey] = useState('sharpe');
  const [sortAsc, setSortAsc] = useState(false);
  const [hoveredStrategy, setHoveredStrategy] = useState(null);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [selectedOptStrategy, setSelectedOptStrategy] = useState('bb_squeeze');
  const [selectedHeatmapMetric, setSelectedHeatmapMetric] = useState('sharpe');
  const [visibleStrategies, setVisibleStrategies] = useState([...STRATEGY_KEYS, 'buyHold']);

  const deepDiveRef = useRef(null);

  const handleSelectStrategy = useCallback((key) => {
    setSelectedStrategy(prev => prev === key ? null : key);
    setTimeout(() => {
      deepDiveRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, []);

  return (
    <div style={{
      background: '#0a0a0f', color: '#e2e2e8', minHeight: '100vh',
      fontFamily: "'Inter', -apple-system, sans-serif",
    }}>
      {/* Google Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0a0a0f; }
        ::-webkit-scrollbar-thumb { background: #2e2e4e; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #3e3e5e; }

        table th:hover { color: #3b82f6 !important; }
        tr:hover td { background: #1a1a2815; }
      `}</style>

      <Header />

      {/* Section 1: Leaderboard */}
      <StrategyLeaderboard
        sortKey={sortKey} setSortKey={setSortKey}
        sortAsc={sortAsc} setSortAsc={setSortAsc}
        hoveredStrategy={hoveredStrategy} setHoveredStrategy={setHoveredStrategy}
        onSelectStrategy={handleSelectStrategy}
      />

      {/* Divider */}
      <div style={{ borderBottom: '1px solid #1e1e2e', margin: '0 20px' }} />

      {/* Section 2: Equity & Drawdown */}
      <div style={{ paddingTop: 16 }}>
        <EquityDrawdownCharts hoveredStrategy={hoveredStrategy} visibleStrategies={visibleStrategies} />
      </div>

      <div style={{ borderBottom: '1px solid #1e1e2e', margin: '0 20px' }} />

      {/* Section 3: Optimization Journey */}
      <div style={{ paddingTop: 16 }}>
        <OptimizationJourney
          selectedOptStrategy={selectedOptStrategy}
          setSelectedOptStrategy={setSelectedOptStrategy}
          selectedHeatmapMetric={selectedHeatmapMetric}
          setSelectedHeatmapMetric={setSelectedHeatmapMetric}
        />
      </div>

      <div style={{ borderBottom: '1px solid #1e1e2e', margin: '0 20px' }} />

      {/* Section 4: Deep Dive (conditional) */}
      {selectedStrategy && (
        <div ref={deepDiveRef} style={{ paddingTop: 16 }}>
          <StrategyDeepDive strategyKey={selectedStrategy} />
          <div style={{ borderBottom: '1px solid #1e1e2e', margin: '0 20px' }} />
        </div>
      )}

      {!selectedStrategy && (
        <div style={{
          padding: '24px 20px', textAlign: 'center', color: '#555570', fontSize: 12,
          fontFamily: "'Inter', sans-serif",
        }}>
          Click a strategy row in the leaderboard to view its detailed analysis
        </div>
      )}

      {/* Section 5: Risk Analysis */}
      <div style={{ paddingTop: 16 }}>
        <RiskAnalysis />
      </div>

      <div style={{ borderBottom: '1px solid #1e1e2e', margin: '0 20px' }} />

      {/* Section 6: Regime Breakdown */}
      <div style={{ paddingTop: 16 }}>
        <RegimeBreakdown selectedStrategy={selectedStrategy || 'bb_squeeze'} />
      </div>

      {/* Footer */}
      <div style={{
        padding: '16px 20px', textAlign: 'center', color: '#333', fontSize: 10,
        fontFamily: "'Inter', sans-serif", borderTop: '1px solid #1e1e2e',
      }}>
        CMT Backtest Lab · Generated {new Date().toISOString().split('T')[0]} · Data: BTC/USDT Jan 2024
      </div>
    </div>
  );
}
