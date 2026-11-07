"""Expected-value decisions.

Every option is valued in chips won from now on, counting chips already in
the pot as up for grabs (fold = 0):

  check        rho * eq * pot
  call         rho * eq * (pot + call) - call
  raise to X   P(fold) * pot
               + P(call)  * (rho_c * eq_vs_calling_range * pot_called - our_chips)
               + P(raise) * max(-our_chips, value of calling a pot re-raise
                                 with eq_vs_reraising_range)

eq comes from one Monte Carlo pass against the weighted range; the calling and
re-raising ranges are the same range reweighted by the threshold model's
P(call | hand) and P(raise | hand) for that bet size, and the opponent model's
fold / call / raise frequencies for that size bucket are the P(...) above.
rho is equity realisation: 1 on the river and when all-in, less when out of
position with deep stacks; preflop it moves with the hand's preflop score and
stack depth.

On top of the EV choice, weak hands that would check (or give up) bet as a
bluff with probability tied to the bet's pot odds (bluff:value =
call / (pot + our bet)), times our estimated value-betting frequency, times
how much more than default this opponent folds.  Folding is never an option
when checking is free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .opponent import AGGR_I, FOLD_I, PASSIVE_I, OpponentModel
from .types import CALL, CHECK, FOLD, RAISE, Action


@dataclass
class Spot:
    hole: tuple
    board: tuple
    street: int
    pot: float
    to_call: float
    my_bet: float
    opp_bet: float
    my_stack: float
    opp_stack: float
    min_raise_to: Optional[float]
    max_raise_to: Optional[float]
    big_blind: float
    in_position: bool
    raises_this_street: int = 0
    integer_chips: bool = True
    preflop_pct: float = 0.5     # our hand's preflop percentile (preflop only)
    opp_range_strength: float = 0.5   # posterior mean preflop percentile of their range (preflop only)
    raise_options: Optional[tuple] = None   # discrete raise-to totals, if the rules fix them

    @property
    def can_raise(self) -> bool:
        return (self.min_raise_to is not None and self.max_raise_to is not None
                and self.max_raise_to >= self.min_raise_to and self.my_stack > self.to_call)


@dataclass
class Option:
    action: Action
    ev: float
    info: dict = field(default_factory=dict)


def raise_sizes(spot: Spot, fracs) -> List[float]:
    """Raise-to amounts for each pot fraction, inside the pot-limit bounds."""
    if not spot.can_raise:
        return []
    if spot.raise_options is not None:
        # discrete action space: exactly the totals the rules allow (deduped, ascending)
        return sorted({float(x) for x in spot.raise_options if x is not None and x > spot.opp_bet})
    after_call = spot.pot + spot.to_call
    out = []
    for f in fracs:
        x = spot.opp_bet + f * after_call
        x = min(max(x, spot.min_raise_to), spot.max_raise_to)
        # never commit more than our stack allows
        x = min(x, spot.my_bet + spot.my_stack)
        if spot.integer_chips:
            x = float(int(math.floor(x + 0.5)))
            if x < spot.min_raise_to:
                x = float(math.ceil(spot.min_raise_to))
            if x > spot.max_raise_to:
                x = float(math.floor(spot.max_raise_to))
        if x > spot.opp_bet and x not in out:
            out.append(x)
    return out


def realization(spot: Spot, settings, spr: float, street: Optional[int] = None) -> float:
    street = spot.street if street is None else street
    if street >= 3 or spr <= 0:
        return 1.0
    base = settings.realization_ip if spot.in_position else settings.realization_oop
    rho = 1.0 - (1.0 - base) * spr / (spr + 1.0)
    if street == 0:
        depth = min(1.0, (min(spot.my_stack, spot.opp_stack) / max(spot.big_blind, 1e-9)) / 100.0)
        rho *= 1.0 + settings.preflop_score_realization * (spot.preflop_pct - 0.5) * depth
        # against a range that has shown strength (3-bets, 4-bets) we realise less:
        # it holds the hands that win the big postflop pots
        rho *= 1.0 - settings.preflop_range_realization * max(0.0, spot.opp_range_strength - 0.5)
    return float(min(rho, 1.1))


def opponent_context(spot: Spot, split_4bet: bool = False) -> str:
    """What the opponent faces after we raise."""
    if spot.street == 0:
        if spot.raises_this_street == 0:
            # preflop with nothing to call = we are the big blind after their limp
            return "limp_vs_raise" if spot.to_call <= 0 else "vs_raise"
        if spot.raises_this_street >= 2 and split_4bet:
            return "vs_4bet"
        return "vs_reraise"
    return "facing_bet" if spot.to_call == 0 and spot.raises_this_street == 0 else "facing_raise"


def strength_rz(spot, settings, spr: float, eq: float) -> float:
    """Postflop, before the river, with chips behind: strong hands realise more
    than their equity (they get paid on later streets), weak ones less (they fold
    to later bets).  Factor 1 + k * (eq - 0.5), scaled by min(spr, 1); k =
    ``postflop_eq_realization`` (0 = off).  Preflop has its own slope."""
    k = settings.postflop_eq_realization
    if k == 0 or spot.street == 0 or spot.street >= 3 or spr <= 0:
        return 1.0
    return float(np.clip(1.0 + k * (eq - 0.5) * min(1.0, spr), settings.rz_min, settings.rz_max))


def ev_check(spot, settings, eq) -> float:
    spr = min(spot.my_stack, spot.opp_stack) / max(spot.pot, 1e-9)
    return realization(spot, settings, spr) * strength_rz(spot, settings, spr, eq) * eq * spot.pot


def ev_call(spot, settings, eq) -> float:
    c = min(spot.to_call, spot.my_stack)
    returned = spot.to_call - c
    pot = spot.pot - returned + c
    left = min(spot.my_stack - c, spot.opp_stack)
    spr = left / max(pot, 1e-9)
    rz = strength_rz(spot, settings, spr, eq) if settings.rz_on_call else 1.0
    return realization(spot, settings, spr) * rz * eq * pot - c


def ev_raise(spot, settings, x, p_fold, p_call, p_raise, eq_call, eq_reraise) -> tuple:
    a = x - spot.my_bet            # our chips going in now
    d = x - spot.opp_bet           # opponent's cost to call
    my_left = spot.my_stack - a
    opp_left = spot.opp_stack - d
    pot_c = spot.pot + a + d
    spr_c = max(0.0, min(my_left, opp_left)) / max(pot_c, 1e-9)
    rho_c = realization(spot, settings, spr_c)
    if spot.street > 0 and spr_c > 0:
        # called by a range that chose to continue, with streets and chips still to
        # play: out of position we realise less of the bigger pot (knob, 1 = off)
        k = settings.called_realization_oop if not spot.in_position else settings.called_realization_ip
        rho_c *= 1.0 - (1.0 - k) * min(1.0, spr_c)
    ev_called = rho_c * strength_rz(spot, settings, spr_c, eq_call) * eq_call * pot_c - a
    r_inc = max(0.0, min(pot_c, my_left, opp_left))
    if r_inc <= 0:
        # nobody can raise again (one of us is all-in): the hands that would have
        # re-raised just call, so that share is a call by the STRONG part of the
        # range, not the middle (using eq_call here overrated every all-in shove)
        ev_rr = realization(spot, settings, spr_c) * eq_reraise * pot_c - a
    else:
        pot_rr = pot_c + 2 * r_inc
        spr_rr = max(0.0, min(my_left, opp_left) - r_inc) / pot_rr
        ev_rr_call = (realization(spot, settings, spr_rr) * strength_rz(spot, settings, spr_rr, eq_reraise)
                      * eq_reraise * pot_rr - (a + r_inc))
        ev_rr = max(-a, ev_rr_call)
    s = p_fold + p_call + p_raise
    p_fold, p_call, p_raise = p_fold / s, p_call / s, p_raise / s
    ev = p_fold * spot.pot + p_call * ev_called + p_raise * ev_rr
    return ev, {"ev_called": ev_called, "ev_reraised": ev_rr, "a": a, "d": d}


def strength_adjust(freqs, delta: float, beta: float) -> np.ndarray:
    """Condition the opponent's fold frequency on how strong their range is NOW.

    The learned frequencies average over all the spots where we bet; ``delta`` is
    the posterior mean absolute strength of their range now minus its average at
    those spots.  A range that showed strength (raised the flop) folds far less
    than average to a turn bet, one that checked twice folds more; most student
    bots fold on absolute strength.  Fold odds move by exp(-beta * delta); the
    mass moves between fold and call.
    """
    f = np.asarray(freqs, dtype=float).copy()
    p = float(np.clip(f[FOLD_I], 1e-4, 1 - 1e-4))
    z = math.log(p / (1 - p)) - beta * float(delta)
    new = 1.0 / (1.0 + math.exp(-z))
    new = min(new, f[FOLD_I] + f[PASSIVE_I])
    f[PASSIVE_I] -= new - f[FOLD_I]
    f[FOLD_I] = new
    return f


def commitment_adjust(freqs, price: float, ref: float) -> np.ndarray:
    """Shrink the opponent's fold frequency when calling is cheap for them.

    ``price`` = their call / the pot if they call (the equity a call needs).  The
    learned frequencies pool all bet sizes and stack depths, but folding a third
    of the time when a call needs 10% equity (a small shove into a big pot, or a
    re-raise after they bet) is far off for any sane bot, and it is exactly where
    overestimated fold equity is most expensive.  Below ``ref`` the fold share is
    scaled by price / ref and the removed mass goes to calling.
    """
    f = np.asarray(freqs, dtype=float).copy()
    if ref <= 0 or price >= ref:
        return f
    k = max(0.0, price / ref)
    moved = f[FOLD_I] * (1.0 - k)
    f[FOLD_I] -= moved
    f[PASSIVE_I] += moved
    return f


def weighted_equities(weights, outcomes, reweights, prior_factors, prior_strength: float = 8.0):
    """Equity vs the range and vs each reweighted sub-range from per-hand expected
    outcomes (no sampling noise; thin sub-ranges shrink toward factor * base by
    their effective particle count, as the Monte Carlo path does)."""
    w = np.asarray(weights, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    tw = float(w.sum())
    base = float((w * o).sum() / tw) if tw > 0 else 0.5
    subs = []
    for q, fac in zip(reweights, prior_factors):
        wq = w * np.asarray(q, dtype=float)
        s = float(wq.sum())
        prior = base * float(fac)
        if s <= 1e-12:
            subs.append(prior)
            continue
        est = float((wq * o).sum() / s)
        n_eff = s * s / max(float((wq * wq).sum()), 1e-12)
        subs.append((n_eff * est + prior_strength * prior) / (n_eff + prior_strength))
    return base, subs


class DecisionEngine:
    def __init__(self, settings, equity_engine, model: OpponentModel, bands=None):
        self.s = settings
        self.equity = equity_engine
        self.model = model
        self.bands = bands      # preflop BandTable (exact preflop equity vs a range), optional
        # online fold calibration: predicted (before scaling) vs actual folds to our postflop bets
        self.fold_pred = 0.0
        self.fold_act = 0.0

    def fold_scale_now(self) -> float:
        """Postflop fold-share multiplier: the fixed ``fold_scale``, or with
        ``fold_calibrate`` that value as a prior updated by how often this opponent
        actually folded vs how often the model said it would."""
        s0 = self.s.fold_scale
        if not self.s.fold_calibrate:
            return s0
        k = self.s.fold_calibrate_strength          # pseudo-bets at the prior scale
        m = 0.35                                     # a typical predicted fold share
        c = (self.fold_act + k * m * s0) / (self.fold_pred + k * m)
        return float(min(self.s.fold_calibrate_max, max(self.s.fold_calibrate_min, c)))

    def record_fold_outcome(self, predicted_raw: float, folded: bool) -> None:
        self.fold_pred += float(predicted_raw)
        self.fold_act += 1.0 if folded else 0.0

    # -------------------------------------------------------------- inputs
    def size_inputs(self, spot: Spot, rng_range, x: float) -> dict:
        ctx = opponent_context(spot, self.s.split_preflop_4bet)
        a = x - spot.my_bet
        d = x - spot.opp_bet
        pot_after = spot.pot + a
        frac = d / max(pot_after - d, 1e-9)
        freqs = self.model.freqs(ctx, spot.street, frac)
        if (self.s.strength_fold_beta > 0 and spot.street >= self.s.strength_fold_from_street
                and hasattr(rng_range, "abs_strength_mean")):
            freqs = strength_adjust(freqs, rng_range.abs_strength_mean() - self.model.bet_strength(spot.street),
                                    self.s.strength_fold_beta)
        freqs = commitment_adjust(freqs, d / max(pot_after + d, 1e-9), self.s.commit_fold_ref)
        p_fold_raw = float(freqs[FOLD_I])
        scale = self.fold_scale_now() if spot.street >= self.s.fold_scale_from_street else 1.0
        if scale != 1.0:
            freqs = np.asarray(freqs, dtype=float).copy()
            moved = min(freqs[FOLD_I] * (1.0 - scale), freqs[PASSIVE_I] + freqs[FOLD_I]) if scale < 1.0 \
                else -min(freqs[FOLD_I] * (scale - 1.0), freqs[PASSIVE_I])
            freqs[FOLD_I] -= moved
            freqs[PASSIVE_I] += moved
        bluff = self.model.bluff_share(ctx, split=True) if self.s.split_bluff_decide else self.model.bluff_share(ctx)
        params = {"bluff_share": bluff * (0.5 if spot.street == 0 else 1.0),
                  "informativeness": self.model.informativeness(spot.street),
                  "call_informativeness": (max(self.model.informativeness(spot.street),
                                               self.model.call_informativeness())
                                           if (spot.street > 0 and self.s.split_informativeness) else None)}
        probs = rng_range.action_probs(freqs, **params)
        return {"ctx": ctx, "frac": frac, "freqs": freqs, "p_fold_raw": p_fold_raw,
                "q_call": probs[PASSIVE_I], "q_raise": probs[AGGR_I]}

    # -------------------------------------------------------------- decide
    def evaluate(self, spot: Spot, rng_range, n_samples: int, rng: np.random.Generator,
                 deadline: Optional[float] = None):
        if spot.street == 0 and self.s.preflop_range_realization and hasattr(rng_range, "abs_strength_mean"):
            spot.opp_range_strength = rng_range.abs_strength_mean()
        sizes = raise_sizes(spot, self.s.raise_fracs)
        if sizes and ((spot.street == 0 and self.s.preflop_max_raise_only)
                      or (spot.street > 0 and self.s.postflop_max_raise_only)):
            sizes = sizes[-1:]
        inputs = [self.size_inputs(spot, rng_range, x) for x in sizes]
        reweights, factors = [], []
        for inp in inputs:
            reweights += [inp["q_call"], inp["q_raise"]]
            factors += [0.9, 0.6]   # thin calling / re-raising ranges shrink toward cautious guesses
        exact = None
        # only with the real engine (test stubs inject fixed equities instead)
        if getattr(self.equity, "supports_exact", False) and hasattr(rng_range, "expected_outcomes"):
            exact = rng_range.expected_outcomes(spot.hole, self.bands)
        if exact is not None:
            base, subs = weighted_equities(rng_range.weights, exact, reweights, factors)
            used = 0
        else:
            base, subs, used = self.equity.estimate(
                spot.hole, spot.board, rng_range.hands, rng_range.weights, reweights,
                n_samples, rng, deadline=deadline, prior_factors=factors)
        options: List[Option] = []
        if spot.to_call > 0:
            options.append(Option(Action(FOLD), 0.0, {}))
            options.append(Option(Action(CALL), ev_call(spot, self.s, base), {}))
        else:
            options.append(Option(Action(CHECK), ev_check(spot, self.s, base), {}))
        for k, (x, inp) in enumerate(zip(sizes, inputs)):
            eq_c, eq_r = subs[2 * k], subs[2 * k + 1]
            if self.s.cap_reraise_equity:
                eq_r = min(eq_r, eq_c)   # a re-raising range is no weaker than a calling one
            f = inp["freqs"]
            ev, info = ev_raise(spot, self.s, x, f[FOLD_I], f[PASSIVE_I], f[AGGR_I], eq_c, eq_r)
            info.update({"eq_call": eq_c, "eq_reraise": eq_r, "p_fold": float(f[FOLD_I]),
                         "p_fold_raw": inp["p_fold_raw"],
                         "p_call": float(f[PASSIVE_I]), "p_raise": float(f[AGGR_I]),
                         "frac": inp["frac"]})
            options.append(Option(Action(RAISE, x), ev, info))
        return base, options, used

    def choose(self, spot: Spot, base_eq: float, options: List[Option], rng: np.random.Generator,
               value_rate: float = 0.3) -> tuple:
        """Pick the best-EV option, then maybe mix in a bluff.  Returns (option, reason)."""
        legal = [o for o in options if not (o.action.kind == FOLD and spot.to_call <= 0)]
        best = max(legal, key=lambda o: o.ev)
        reason = "ev"
        raises = [o for o in legal if o.action.kind == RAISE]
        if best.action.kind != RAISE and raises and spot.street > 0:
            if base_eq < self.s.bluff_eq_max[spot.street]:
                cand = max(raises, key=lambda o: o.ev)
                if cand.ev < best.ev - self.s.bluff_ev_tolerance * spot.pot:
                    return best, reason
                p = self.bluff_probability(spot, cand, value_rate)
                if p > 0 and rng.random() < p:
                    return cand, "bluff"
        return best, reason

    def bluff_probability(self, spot: Spot, cand: Option, value_rate: float) -> float:
        p_fold = cand.info.get("p_fold", 0.0)
        if p_fold < self.s.bluff_min_fold:
            return 0.0
        a, d = cand.info.get("a", 0.0), cand.info.get("d", 0.0)
        ratio = d / max(spot.pot + a, 1e-9)       # bluff : value that makes a call break even
        v = float(np.clip(value_rate, 0.02, 0.8))
        default_fold = self.model.priors.facing_bet[0] if spot.to_call == 0 else self.model.priors.facing_raise[0]
        fold_scale = float(np.clip(p_fold / max(default_fold, 1e-6), 0.0, 2.0))
        p = self.s.bluff_scale * ratio * v / (1.0 - v) * fold_scale
        if spot.to_call > 0:
            p *= self.s.bluff_raise_scale
        return float(np.clip(p, 0.0, self.s.max_bluff_prob))
