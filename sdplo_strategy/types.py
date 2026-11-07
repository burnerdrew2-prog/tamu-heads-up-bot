"""Internal data types.  The adapter converts the core's objects into these.

Conventions (the adapter is responsible for meeting them):
  * Cards are ints in the encoding of ``cards.py``.
  * ``actor`` is 0 for us (hero) and 1 for the opponent.
  * Chip amounts are numbers in the core's units (ints if the core uses ints).
  * ``pot`` is ALL chips in the middle, including both players' bets on the
    current street (and the blinds).
  * A raise (or an opening bet) carries ``amount`` = raise-to, the player's
    total bet on this street after the action.  Blind posts are NOT actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"
KINDS = (FOLD, CHECK, CALL, RAISE)
HERO, OPP = 0, 1


@dataclass(frozen=True)
class ActionRecord:
    actor: int          # HERO or OPP
    street: int         # 0 preflop, 1 flop, 2 turn, 3 river
    kind: str           # fold / check / call / raise ("bet" is a raise with nothing to call)
    amount: float = 0   # raise-to for raises, ignored otherwise


@dataclass(frozen=True)
class Observation:
    hole: tuple
    board: tuple
    pot: float
    my_stack: float          # chips behind (not counting what is already bet)
    opp_stack: float
    to_call: float
    min_raise_to: Optional[float]   # None when raising is not allowed
    max_raise_to: Optional[float]
    street: int
    history: tuple = ()      # tuple[ActionRecord] for this hand, in order
    hand_number: int = 0
    small_blind: float = 1
    big_blind: float = 2
    hero_is_button: Optional[bool] = None   # button = small blind heads-up
    my_street_bet: Optional[float] = None   # chips we have in on this street
    opp_street_bet: Optional[float] = None
    # Discrete action space (official rules): the only legal raise-to totals,
    # e.g. (half-pot, pot).  None = any amount in [min_raise_to, max_raise_to].
    raise_options: Optional[tuple] = None

    @property
    def can_raise(self) -> bool:
        return (self.min_raise_to is not None and self.max_raise_to is not None
                and self.max_raise_to >= self.min_raise_to and self.my_stack > self.to_call)


@dataclass(frozen=True)
class Action:
    kind: str
    amount: float = 0   # raise-to for RAISE

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"bad action kind {self.kind!r}")


@dataclass
class HandResult:
    hand_number: int
    history: tuple                       # full ActionRecord sequence
    board: tuple = ()
    hero_hole: tuple = ()
    opp_hole: Optional[tuple] = None     # shown at showdown, else None
    hero_delta: Optional[float] = None   # chips won (+) / lost (-)
    hero_is_button: Optional[bool] = None
    small_blind: float = 1
    big_blind: float = 2
    extra: dict = field(default_factory=dict)
