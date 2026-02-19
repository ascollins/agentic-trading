"""Reconciliation layer — journal analytics and exchange-state reconciliation.

Provides trade journaling, performance analytics, exchange reconciliation,
and the unified :class:`ReconciliationManager` facade.

Prior to this move the journal code lived under ``agentic_trading.journal``
and the reconciliation loop under ``agentic_trading.execution.reconciliation``.
Both old locations now contain thin re-export shims for backward compatibility
(removed in PR 16).
"""

from agentic_trading.reconciliation.journal import (
    CoinFlipBaseline,
    ConfidenceCalibrator,
    CorrelationMatrix,
    Grade,
    MetricGrade,
    Mistake,
    MistakeDetector,
    MonteCarloProjector,
    OvertradingDetector,
    PortfolioQualityReport,
    QualityReport,
    QualityScorecard,
    RollingTracker,
    SessionAnalyser,
    StrategyType,
    TradeExporter,
    TradeJournal,
    TradeOutcome,
    TradePhase,
    TradeRecord,
    TradeReplayer,
)
from agentic_trading.reconciliation.loop import ReconciliationLoop
from agentic_trading.reconciliation.manager import ReconciliationManager

__all__ = [
    # Journal core
    "TradeRecord",
    "TradePhase",
    "TradeOutcome",
    "TradeJournal",
    # Analytics
    "RollingTracker",
    "ConfidenceCalibrator",
    "MonteCarloProjector",
    "OvertradingDetector",
    "CoinFlipBaseline",
    # Behavioral
    "MistakeDetector",
    "Mistake",
    "SessionAnalyser",
    "CorrelationMatrix",
    "TradeReplayer",
    # Export
    "TradeExporter",
    # Quality
    "QualityScorecard",
    "QualityReport",
    "PortfolioQualityReport",
    "MetricGrade",
    "Grade",
    "StrategyType",
    # Reconciliation
    "ReconciliationLoop",
    "ReconciliationManager",
]
