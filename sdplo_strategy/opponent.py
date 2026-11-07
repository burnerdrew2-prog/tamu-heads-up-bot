"""Per-opponent statistics with Bayesian priors.

Every frequency is a Dirichlet (Beta for two outcomes): observed counts plus
pseudo-counts pulled toward a prior mean, so early estimates are sensible and
they move toward the data as the match goes on.  Postflop "facing a bet"
cells are hierarchical: (street, size bucket) shrinks toward the street pool,
which shrinks toward the all-streets pool, which shrinks toward the default.

Decisions use the numbers (``freqs``, ``bluff_share``, ``informativeness``).
``summary()`` adds human-readable labels for logs only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .history import N_BUCKETS, DecisionPoint, size_bucket
from .types import CALL, CHECK, FOLD, OPP, RAISE

# outcome index used everywhere: 0 fold, 1 passive (check/call), 2 aggressive (bet/raise)
FOLD_I, PASSIVE_I, AGGR_I = 0, 1, 2
UNOPENED = ("lead", "checked_to", "bb_option")


def outcome_index(kind: str) -> int:
    if kind == FOLD:
        return FOLD_I
    if kind in (CHECK, CALL):
        return PASSIVE_I
    if kind == RAISE:
        return AGGR_I
    raise ValueError(kind)


class Dirichlet:
    """Counts over K outcomes with a Dirichlet prior (a Beta when K == 2)."""

    def __init__(self, prior_mean: Sequence[float], strength: float):
        p = np.asarray(prior_mean, dtype=float)
        self.prior = p / p.sum()
        self.strength = float(strength)
        self.counts = np.zeros(len(p))

    def observe(self, k: int, w: float = 1.0) -> None:
        self.counts[k] += w

    @property
    def n(self) -> float:
        return float(self.counts.sum())

    def mean(self, prior: Optional[np.ndarray] = None) -> np.ndarray:
        prior = self.prior if prior is None else prior
        return (self.counts + self.strength * prior) / (self.n + self.strength)


class Beta(Dirichlet):
    def __init__(self, prior_mean: float, strength: float):
        super().__init__([1.0 - prior_mean, prior_mean], strength)

    def observe_bool(self, success: bool, w: float = 1.0) -> None:
        self.observe(1 if success else 0, w)

    @property
    def p(self) -> float:
        return float(self.mean()[1])


class MeanStat:
    """Mean of a quantity in [0, 1] with a prior (pseudo-observations)."""

    def __init__(self, prior: float, strength: float):
        self.prior = prior
        self.strength = strength
        self.total = 0.0
        self.n = 0

    def observe(self, x: float) -> None:
        self.total += float(x)
        self.n += 1

    @property
    def mean(self) -> float:
        return (self.total + self.strength * self.prior) / (self.n + self.strength)


@dataclass
class Priors:
    """Default opponent: a fairly loose, fairly passive student bot."""
    preflop: Dict[str, tuple] = field(default_factory=lambda: {
        "sb_open": (0.15, 0.35, 0.50),
        "bb_option": (0.0, 0.65, 0.35),
        "vs_raise": (0.30, 0.55, 0.15),
        "limp_vs_raise": (0.12, 0.70, 0.18),   # limpers rarely fold to a raise
        "vs_reraise": (0.30, 0.50, 0.20),
    })
    preflop_strength: float = 6.0
    facing_bet: tuple = (0.40, 0.48, 0.12)
    facing_bet_fold_shift: tuple = (-0.08, 0.0, 0.08)   # by size bucket
    # Facing a raise postflop the opponent has already bet or raised this street;
    # student bots (equity thresholds, pot odds, charts) almost never fold there.
    facing_raise: tuple = (0.12, 0.58, 0.30)
    lead: tuple = (0.0, 0.60, 0.40)
    checked_to: tuple = (0.0, 0.50, 0.50)
    global_strength: float = 8.0
    street_strength: float = 8.0
    cell_strength: float = 6.0
    unopened_strength: float = 6.0
    vpip: float = 0.75
    pfr: float = 0.45
    rate_strength: float = 6.0
    bluff_share: float = 0.20      # share of shown aggressive lines that were weak
    bluff_strength: float = 6.0
    bluff_raise_share: float = 0.10   # share of shown raise-facing-a-bet lines that were weak
    split_bluff_share: bool = False   # True = own bluff share for raises facing a bet (tested worse vs drew_v2)
    showdown_strength_prior: Dict[str, float] = field(default_factory=lambda: {
        "river_aggressive": 0.78, "aggressive": 0.70, "passive": 0.55, "no_postflop": 0.60})
    showdown_strength: float = 4.0
    preflop_raise_strength: float = 0.72   # prior preflop percentile of shown hands that raised preflop
    call_strength_prior: float = 0.66      # prior showdown strength of hands that called a postflop bet
    weak_cutoff: float = 0.5       # shown hand below this percentile = weak


def _shift_fold(p: np.ndarray, shift: float) -> np.ndarray:
    q = p.copy()
    new_f = float(np.clip(q[0] + shift, 0.01, 0.97))
    rest = q[1:].sum()
    if rest > 0:
        q[1:] *= (1.0 - new_f) / rest
    q[0] = new_f
    return q


class OpponentModel:
    def __init__(self, priors: Optional[Priors] = None):
        self.priors = pr = priors or Priors()
        self.preflop = {k: Dirichlet(v, pr.preflop_strength) for k, v in pr.preflop.items()}
        self.fb_global = Dirichlet(pr.facing_bet, pr.global_strength)
        self.fb_street = {s: Dirichlet(pr.facing_bet, pr.street_strength) for s in (1, 2, 3)}
        self.fb_cell = {(s, b): Dirichlet(pr.facing_bet, pr.cell_strength)
                        for s in (1, 2, 3) for b in range(N_BUCKETS)}
        self.facing_raise = Dirichlet(pr.facing_raise, pr.street_strength)
        self.unopened = {(ctx, s): Dirichlet(getattr(pr, ctx), pr.unopened_strength)
                         for ctx in ("lead", "checked_to") for s in (1, 2, 3)}
        self.vpip = Beta(pr.vpip, pr.rate_strength)
        self.pfr = Beta(pr.pfr, pr.rate_strength)
        self.bluff = Beta(pr.bluff_share, pr.bluff_strength)
        # raises facing OUR bet are judged separately: semi-bluff bets and thin value
        # bets say nothing about how often a bot's raises are weak
        self.bluff_raise = Beta(pr.bluff_raise_share, pr.bluff_strength)
        self.showdown = {k: MeanStat(v, pr.showdown_strength)
                         for k, v in pr.showdown_strength_prior.items()}
        self.bet_size = {s: MeanStat(0.66, 2.0) for s in (1, 2, 3)}
        self.preflop_raise = MeanStat(pr.preflop_raise_strength, pr.showdown_strength)
        # shown strength of hands that CALLED a postflop bet (and never raised)
        self.call_strength = MeanStat(pr.call_strength_prior, pr.showdown_strength)
        # mean absolute strength of their range at the moments we bet / raised, per street
        self._bet_strength = {s: MeanStat(0.5, 3.0) for s in (0, 1, 2, 3)}
        self.hands = 0
        self.showdowns = 0

    # ------------------------------------------------------------ queries
    def facing_bet_freqs(self, street: int, frac: float) -> np.ndarray:
        g = self.fb_global.mean()
        s = self.fb_street[street].mean(prior=g)
        b = size_bucket(frac)
        cell_prior = _shift_fold(s, self.priors.facing_bet_fold_shift[b])
        return self.fb_cell[(street, b)].mean(prior=cell_prior)

    def freqs(self, context: str, street: int, frac: float = 0.0) -> np.ndarray:
        """(fold, passive, aggressive) probabilities for the opponent in this spot."""
        if street == 0 or context in self.preflop:
            return self.preflop[context].mean()
        if context == "facing_bet":
            return self.facing_bet_freqs(street, frac)
        if context == "facing_raise":
            return self.facing_raise.mean()
        return self.unopened[(context, street)].mean()

    def fold_to_bet(self, street: int, frac: float) -> float:
        return float(self.facing_bet_freqs(street, frac)[FOLD_I])

    def call_informativeness(self) -> float:
        """How much their calling (vs folding) tracks strength: shown hands that
        called postflop bets vs a random hand.  ~0 for a calling station."""
        exp = self.priors.call_strength_prior - 0.5
        got = self.call_strength.mean - 0.5
        return float(np.clip(got / exp, 0.05, 1.0)) if exp > 0 else 1.0

    def bet_strength(self, street: int) -> float:
        return self._bet_strength[street].mean

    def observe_bet_strength(self, street: int, strength: float) -> None:
        self._bet_strength[street].observe(strength)

    def bluff_share(self, context: Optional[str] = None) -> float:
        """Weak share of the opponent's aggressive range: for raises facing a bet
        (``facing_bet`` / ``facing_raise``) its own estimate, else that of bets."""
        if self.priors.split_bluff_share and context in ("facing_bet", "facing_raise"):
            return self.bluff_raise.p
        return self.bluff.p

    def informativeness(self, street: int = 1) -> float:
        """How much the opponent's aggression tracks hand strength, 0.05..1.

        Postflop: how far above a random hand (0.5) their shown hands on
        aggressive lines sit, relative to the gap the priors assume.  (Comparing
        against their PASSIVE lines instead, as before, called a nit or a chart
        bot "uninformative" because everything they show down is strong.)
        Preflop: the same for the preflop strength of shown hands that raised.
        Starts near 1 (trust the threshold model); drops toward 0.05 against bots
        whose aggression ignores their cards (maniacs, random bots).
        """
        if street == 0:
            exp = self.priors.preflop_raise_strength - 0.5
            got = self.preflop_raise.mean - 0.5
        else:
            pr = self.priors.showdown_strength_prior
            exp = 0.5 * (pr["river_aggressive"] + pr["aggressive"]) - 0.5
            got = 0.5 * (self.showdown["river_aggressive"].mean + self.showdown["aggressive"].mean) - 0.5
        return float(np.clip(got / exp, 0.05, 1.0)) if exp > 0 else 1.0

    # ------------------------------------------------------------ updates
    def observe_point(self, p: DecisionPoint) -> None:
        """Count one opponent decision."""
        if p.actor != OPP:
            return
        k = outcome_index(p.kind)
        if p.street == 0:
            if p.context in self.preflop:
                self.preflop[p.context].observe(k)
            return
        if p.context == "facing_bet":
            self.fb_global.observe(k)
            self.fb_street[p.street].observe(k)
            self.fb_cell[(p.street, p.bucket)].observe(k)
        elif p.context == "facing_raise":
            self.facing_raise.observe(k)
        elif p.context in ("lead", "checked_to"):
            self.unopened[(p.context, p.street)].observe(k)
        if p.kind == RAISE and p.to_call == 0 and p.pot > 0:
            self.bet_size[p.street].observe(min(p.amount / p.pot, 2.0) / 2.0)

    def record_hand(self, points: Sequence[DecisionPoint], line: Optional[str] = None,
                    shown_strength: Optional[float] = None,
                    shown_preflop_pct: Optional[float] = None) -> None:
        """Update every count from one finished hand."""
        self.hands += 1
        opp_pre = [p for p in points if p.actor == OPP and p.street == 0]
        if shown_preflop_pct is not None and any(p.kind == RAISE for p in opp_pre):
            self.preflop_raise.observe(shown_preflop_pct)
        for p in points:
            self.observe_point(p)
        if opp_pre:
            self.vpip.observe_bool(any(p.kind in (CALL, RAISE) for p in opp_pre))
            self.pfr.observe_bool(any(p.kind == RAISE for p in opp_pre))
        if shown_strength is not None and line is not None:
            post = [p for p in points if p.actor == OPP and p.street > 0]
            raised_vs_bet = any(p.kind == RAISE and p.context in ("facing_bet", "facing_raise") for p in post)
            self.record_showdown(line, shown_strength, raised_vs_bet)
            if line == "passive" and any(p.kind == CALL for p in post):
                self.call_strength.observe(shown_strength)

    def record_showdown(self, line: str, strength: float, raised_vs_bet: bool = False) -> None:
        self.showdowns += 1
        if line in self.showdown:
            self.showdown[line].observe(strength)
        weak = strength < self.priors.weak_cutoff
        if raised_vs_bet:
            self.bluff_raise.observe_bool(weak)
        if line in ("river_aggressive", "aggressive") and (not raised_vs_bet or not self.priors.split_bluff_share):
            self.bluff.observe_bool(weak)

    # ------------------------------------------------------------ summary
    def summary(self) -> dict:
        fold_tb = float(self.fb_global.mean()[FOLD_I])
        raise_tb = float(self.fb_global.mean()[AGGR_I])
        stab = float(np.mean([self.unopened[("checked_to", s)].mean()[AGGR_I] for s in (1, 2, 3)]))
        vpip, pfr = self.vpip.p, self.pfr.p
        labels = []
        if fold_tb < 0.25 and vpip > 0.6:
            labels.append("calling station")
        if fold_tb > 0.55:
            labels.append("over-folder")
        if pfr > 0.6 and (stab > 0.65 or raise_tb > 0.3):
            labels.append("maniac")
        if vpip < 0.45 and pfr < 0.25 and stab < 0.4:
            labels.append("tight-passive")
        return {
            "hands": self.hands, "showdowns": self.showdowns,
            "vpip": round(vpip, 3), "pfr": round(pfr, 3),
            "fold_to_bet": round(fold_tb, 3), "raise_vs_bet": round(raise_tb, 3),
            "bet_when_checked_to": round(stab, 3),
            "fold_to_bet_by_street_bucket": {
                f"{s}/{b}": round(float(self.facing_bet_freqs(s, (0.3, 0.66, 1.0)[b])[0]), 3)
                for s in (1, 2, 3) for b in range(N_BUCKETS)},
            "facing_raise": [round(float(x), 3) for x in self.facing_raise.mean()],
            "vs_raise": [round(float(x), 3) for x in self.preflop["vs_raise"].mean()],
            "limp_vs_raise": [round(float(x), 3) for x in self.preflop["limp_vs_raise"].mean()],
            "vs_reraise": [round(float(x), 3) for x in self.preflop["vs_reraise"].mean()],
            "bluff_share": round(self.bluff_share(), 3),
            "bluff_raise_share": round(self.bluff_raise.p, 3),
            "informativeness": round(self.informativeness(), 3),
            "informativeness_preflop": round(self.informativeness(0), 3),
            "call_informativeness": round(self.call_informativeness(), 3),
            "showdown_strength": {k: round(v.mean, 3) for k, v in self.showdown.items()},
            "labels": labels,
        }
