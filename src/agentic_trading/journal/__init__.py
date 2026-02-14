"""Trade Journal & Analytics — Edgewonk-inspired self-measurement.

Aggregates raw execution events into consolidated trade records,
computes per-trade and rolling performance metrics, and feeds
governance components (health tracker, drift detector, maturity manager).

Key components
--------------
TradeRecord      Consolidated lifecycle record for one trade
TradeJournal     Event subscriber that builds and closes trade records
RollingTracker   Sliding-window performance metrics
ConfidenceCalibrator  Signal quality vs outcome measurement
MonteCarloProjector   Equity projection with probability of ruin
OvertradingDetector   Abnormal signal frequency detection
CoinFlipBaseline      Statistical edge verification
"""

from .record import TradeRecord, TradePhase, TradeOutcome
from .journal import TradeJournal
from .rolling_tracker import RollingTracker
from .confidence import ConfidenceCalibrator
from .monte_carlo import MonteCarloProjector
from .overtrading import OvertradingDetector
from .coin_flip import CoinFlipBaseline

__all__ = [
    "TradeRecord",
    "TradePhase",
    "TradeOutcome",
    "TradeJournal",
    "RollingTracker",
    "ConfidenceCalibrator",
    "MonteCarloProjector",
    "OvertradingDetector",
    "CoinFlipBaseline",
]
