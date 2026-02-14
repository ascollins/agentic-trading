"""Avatar Narration module — live plain-English commentary on platform decisions.

Two output channels:
  1. **Text Stream** — Grafana homepage panel via HTTP JSON endpoint
  2. **Avatar Video** — Tavus.io avatar reads narration via /avatar/watch

All narration is grounded in DecisionExplanation schemas — the avatar
never invents reasons beyond what the platform actually computed.
"""

from .schema import DecisionExplanation, NarrationItem
from .service import NarrationService, Verbosity
from .store import NarrationStore
from .tavus import TavusAdapter, TavusAdapterHttp, MockTavusAdapter

__all__ = [
    "DecisionExplanation",
    "NarrationItem",
    "NarrationService",
    "Verbosity",
    "NarrationStore",
    "TavusAdapter",
    "TavusAdapterHttp",
    "MockTavusAdapter",
]
