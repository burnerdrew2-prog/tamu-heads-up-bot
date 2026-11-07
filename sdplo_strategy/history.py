"""Walk a hand's action history and label every decision with its context.

Heads-up: the button posts the small blind, acts first preflop and last after
the flop.  Contexts:

  preflop   sb_open     button's first action, facing only the big blind
            bb_option   big blind after a limp (can check)
            vs_raise    facing the first raise (big blind facing the open)
            limp_vs_raise  the button facing the big blind's raise after limping
            vs_reraise  facing a re-raise (3-bet or more)
  postflop  lead        first to act on the street (no bet yet)
            checked_to  the other player checked
            facing_bet  facing the street's first bet
            facing_raise facing a raise on the street
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .types import CALL, CHECK, FOLD, HERO, OPP, RAISE, ActionRecord

SIZE_EDGES = (0.45, 0.8)  # bet / pot-before-bet: small <= .45 < medium <= .8 < large
N_BUCKETS = 3


def size_bucket(frac: float) -> int:
    if frac <= SIZE_EDGES[0]:
        return 0
    if frac <= SIZE_EDGES[1]:
        return 1
    return 2


@dataclass(frozen=True)
class DecisionPoint:
    index: int
    actor: int
    street: int
    context: str
    to_call: float
    pot: float          # all chips in the middle before this action (incl. the bet faced)
    frac: float         # bet faced / pot before that bet (0 when nothing to call)
    kind: str
    amount: float
    size_frac: float    # for raises: increment over a call / pot after the call
    my_bet: float       # actor's street bet before acting
    opp_bet: float

    @property
    def bucket(self) -> int:
        return size_bucket(self.frac)

    @property
    def aggressive(self) -> bool:
        return self.kind == RAISE


@dataclass
class WalkState:
    street: int
    pot: float
    bets: list
    raises: int
    folded: Optional[int]


def walk(history: Sequence[ActionRecord], hero_is_button: bool, sb: float, bb: float):
    """Return (points, final_state)."""
    button = HERO if hero_is_button else OPP
    big = 1 - button
    bets = [0.0, 0.0]
    bets[button], bets[big] = float(sb), float(bb)
    pot = float(sb) + float(bb)
    street = 0
    raises = 0
    acted: set = set()
    folded = None
    points = []
    for i, rec in enumerate(history):
        if rec.street != street:
            street = rec.street
            bets = [0.0, 0.0]
            raises = 0
            acted = set()
        a = rec.actor
        o = 1 - a
        to_call = max(0.0, bets[o] - bets[a])
        if street == 0:
            if raises == 0 and to_call > 0:
                ctx = "sb_open"
            elif raises == 0:
                ctx = "bb_option"
            elif raises == 1:
                # facing the first raise: after limping (already in voluntarily) or as an open
                ctx = "limp_vs_raise" if a in acted else "vs_raise"
            else:
                ctx = "vs_reraise"
        else:
            if to_call > 0:
                ctx = "facing_bet" if raises <= 1 else "facing_raise"
            elif not acted:
                ctx = "lead"
            else:
                ctx = "checked_to"
        before_bet = pot - to_call
        frac = to_call / before_bet if to_call > 0 and before_bet > 0 else 0.0
        size_frac = 0.0
        kind = rec.kind
        if kind == RAISE:
            amount = float(rec.amount)
            after_call = pot + to_call
            size_frac = (amount - bets[o]) / after_call if after_call > 0 else 0.0
        else:
            amount = 0.0
        points.append(DecisionPoint(i, a, street, ctx, to_call, pot, frac, kind, amount,
                                    size_frac, bets[a], bets[o]))
        if kind == CALL:
            bets[a] += to_call
            pot += to_call
        elif kind == RAISE:
            put = amount - bets[a]
            bets[a] = amount
            pot += put
            raises += 1
        elif kind == FOLD:
            folded = a
        acted.add(a)
    return points, WalkState(street, pot, bets, raises, folded)


def infer_button(history: Sequence[ActionRecord]) -> Optional[bool]:
    """Guess whether hero is the button from who acted first preflop."""
    for rec in history:
        if rec.street == 0:
            return rec.actor == HERO
    return None


def opponent_line(points) -> str:
    """Summarise the opponent's postflop line for showdown bookkeeping."""
    opp_post = [p for p in points if p.actor == OPP and p.street > 0]
    if not opp_post:
        return "no_postflop"
    if any(p.kind == RAISE and p.street == 3 for p in opp_post):
        return "river_aggressive"
    if any(p.kind == RAISE for p in opp_post):
        return "aggressive"
    return "passive"


__all__ = ["DecisionPoint", "walk", "size_bucket", "infer_button", "opponent_line",
           "N_BUCKETS", "CHECK"]
