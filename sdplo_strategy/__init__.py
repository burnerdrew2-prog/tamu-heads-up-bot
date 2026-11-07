"""Decision-making half of a heads-up Short Deck 5-card PLO bot.

Depends only on the standard library and numpy.  The core (evaluator, equity,
engine) is injected; ``adapter.py`` holds all coupling to it.
"""
from .bot import StrategyBot, fallback_action, legalize
from .opponent import OpponentModel, Priors
from .ranges import RangeTracker
from .settings import Settings
from .types import Action, ActionRecord, HandResult, Observation

__all__ = ["StrategyBot", "fallback_action", "legalize", "OpponentModel", "Priors",
           "RangeTracker", "Settings", "Action", "ActionRecord", "HandResult", "Observation"]
