"""Trade Journal & Analytics — Edgewonk-inspired self-measurement.

Aggregates raw execution events into consolidated trade records,
computes per-trade and rolling performance metrics, and feeds
governance components (health tracker, drift detector, maturity manager).

Key components
--------------
**Tier 1 & 2 — Core Journal & Advanced Analytics**

TradeRecord      Consolidated lifecycle record for one trade
TradeJournal     Event subscriber that builds and closes trade records
RollingTracker   Sliding-window performance metrics
ConfidenceCalibrator  Signal quality vs outcome measurement
MonteCarloProjector   Equity projection with probability of ruin
OvertradingDetector   Abnormal signal frequency detection
CoinFlipBaseline      Statistical edge verification

**Tier 3 — Behavioral & Pattern Analysis**

MistakeDetector       Automated mistake detection and classification
SessionAnalyser       Time-of-day and session performance analysis
CorrelationMatrix     Cross-strategy and cross-asset correlation
TradeReplayer         Structured mark-to-market replay for visualization

**Tier 4 — Persistence & Export**

TradeExporter         CSV/JSON trade export and periodic reports
"""

from .record import TradeRecord, TradePhase, TradeOutcome
from .journal import TradeJournal
from .rolling_tracker import RollingTracker
from .confidence import ConfidenceCalibrator
from .monte_carlo import MonteCarloProjector
from .overtrading import OvertradingDetector
from .coin_flip import CoinFlipBaseline
from .mistakes import MistakeDetector, Mistake
from .session_analysis import SessionAnalyser
from .correlation import CorrelationMatrix
from .replay import TradeReplayer
from .export import TradeExporter

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
    "MistakeDetector",
    "Mistake",
    "SessionAnalyser",
    "CorrelationMatrix",
    "TradeReplayer",
    "TradeExporter",
]
