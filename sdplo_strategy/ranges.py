"""The opponent's range as a weighted sample of 5-card hands (a particle filter).

* Start of hand: ``range_size`` random hands that avoid our hole cards, equal
  weight.
* New board cards: drop hands that hold them, compute this street's strength
  proxy for every hand.
* Opponent action: multiply each hand's weight by P(action | hand) from the
  threshold model (likelihood.py) with the opponent model's frequencies.
* Low effective sample size: draw fresh hands consistent with all known cards
  and give each the weight it would have had, by replaying every stored
  opponent action against the strength CDF in force when it happened.  Old and
  new hands come from the same uniform proposal, so the merged weighted set is
  still a valid sample of the posterior.

Strength proxy: preflop, the preflop equity score.  Postflop, the mean
percentile of the hand's best Omaha value across a few shared random runouts
to the river (exact on the river), which credits draws as well as made hands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .cards import DECK_ARRAY, STREET_OF_BOARD_LEN, random_hands, remaining_deck
from .likelihood import StrengthCDF, action_probs, weighted_percentile


def eval_many(evaluate, evaluate_batch, hands: np.ndarray, board: tuple) -> np.ndarray:
    """Omaha value of each hand on a 5-card board; NaN where the hand overlaps the board."""
    hands = np.asarray(hands)
    out = np.full(len(hands), np.nan)
    b = np.array(board, dtype=np.int16)
    mark = np.zeros(36, dtype=bool)          # faster than np.isin for 36 cards
    mark[b] = True
    ok = ~mark[hands].any(axis=1)
    if not ok.any():
        return out
    if evaluate_batch is not None:
        sub = hands[ok]
        out[ok] = np.asarray(evaluate_batch(sub, np.tile(b, (len(sub), 1))), dtype=float)
        return out
    bt = tuple(int(c) for c in board)
    idx = np.nonzero(ok)[0]
    vals = [evaluate(tuple(int(c) for c in hands[i]), bt) for i in idx]
    out[idx] = vals
    return out


@dataclass
class StreetStrength:
    board: tuple
    runouts: list           # list of tuples completing the board to 5 cards
    cdfs: list              # StrengthCDF of hand values per runout

    def raw(self, hands, evaluate, evaluate_batch) -> np.ndarray:
        cols = []
        for ro, cdf in zip(self.runouts, self.cdfs):
            vals = eval_many(evaluate, evaluate_batch, hands, self.board + ro)
            pct = np.full(len(hands), np.nan)
            ok = ~np.isnan(vals)
            pct[ok] = cdf(vals[ok])
            cols.append(pct)
        return _nanmean_cols(cols)


def _nanmean_cols(cols) -> np.ndarray:
    m = np.vstack(cols)
    ok = ~np.isnan(m)
    cnt = ok.sum(axis=0)
    s = np.where(ok, m, 0.0).sum(axis=0)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), 0.5)


GRID = np.linspace(0.0, 1.0, 201)


@dataclass
class Record:
    street: int
    cdf: StrengthCDF        # where hands sat in the range when the action happened
    grid_lik: np.ndarray    # P(action | percentile) on GRID, exactly as applied


class RangeTracker:
    # every reassignment of ``hands`` bumps a version, so per-hand caches stay valid
    @property
    def hands(self) -> np.ndarray:
        return self._hands

    @hands.setter
    def hands(self, value) -> None:
        self._hands = value
        self._hands_ver = getattr(self, "_hands_ver", 0) + 1

    def __init__(self, hero_hole: Sequence[int], settings, rng: np.random.Generator,
                 evaluate: Callable, preflop_scorer, evaluate_batch: Optional[Callable] = None,
                 size: Optional[int] = None, runouts: Optional[int] = None):
        self.hero = tuple(int(c) for c in hero_hole)
        self.s = settings
        self.rng = rng
        self.evaluate = evaluate
        self.evaluate_batch = evaluate_batch
        self.scorer = preflop_scorer
        self.size = int(size or settings.range_size)
        self.runouts = int(runouts if runouts is not None else settings.strength_runouts)
        self.board: tuple = ()
        self.street = 0
        avail = remaining_deck(self.hero)
        self.hands = random_hands(rng, avail, self.size)
        self.weights = np.ones(len(self.hands))
        self.scale = 1.0           # true weight = weights * scale
        self.raw = self.scorer.equities(self.hands)
        self.street_strength: dict = {}
        self.records: list = []
        self.refills = 0
        self._pct = None

    # ------------------------------------------------------------ basics
    def ess(self) -> float:
        w = self.weights
        s2 = float((w * w).sum())
        return float(w.sum()) ** 2 / s2 if s2 > 0 else 0.0

    def normalized(self) -> np.ndarray:
        t = self.weights.sum()
        return self.weights / t if t > 0 else np.full(len(self.weights), 1.0 / max(len(self.weights), 1))

    def percentiles(self) -> np.ndarray:
        if self._pct is None:
            self._pct = weighted_percentile(self.raw, self.weights)
        return self._pct

    def _renorm(self) -> None:
        m = float(self.weights.mean()) if len(self.weights) else 0.0
        if m > 0:
            self.weights /= m
            self.scale *= m
        self._pct = None

    # ------------------------------------------------------------ board
    def set_board(self, board: Sequence[int]) -> None:
        board = tuple(int(c) for c in board)
        if board == self.board:
            return
        new_cards = [c for c in board if c not in self.board]
        if new_cards:
            keep = ~np.isin(self.hands, np.array(new_cards, dtype=np.int16)).any(axis=1)
            self.hands, self.weights, self.raw = self.hands[keep], self.weights[keep], self.raw[keep]
        self.board = board
        self.street = STREET_OF_BOARD_LEN[len(board)]
        if self.weights.sum() <= 0:
            self._reset_uniform()
        elif getattr(self.s, "range_top_up", True) and getattr(self, "top_up_enabled", True):
            self._top_up()
        self._compute_street_strength()
        self._renorm()
        self.maybe_refill()

    def _top_up(self) -> None:
        """Restore ``size`` particles after board cards removed the hands holding them.

        Without this the sample shrinks every street (about 42% of hands hold a
        flop card, 16% more a turn or river card) while the refill trigger still
        compares the ESS against the nominal ``size``, so the first informative
        action after the flop refills a set that is mostly fresh hands.  New
        hands come from the same uniform proposal (consistent with every known
        card) and carry their replayed weight, so the merged set stays a valid
        weighted sample of the posterior.  This street's strength is computed
        afterwards, over the whole set.
        """
        n_new = self.size - len(self.hands)
        if n_new <= 0:
            return
        avail = remaining_deck(self.hero + self.board)
        new = random_hands(self.rng, avail, n_new)
        w_true = self.replay_weights(new)
        self.hands = np.concatenate([self.hands, new])
        self.weights = np.concatenate([self.weights, w_true / self.scale])
        self.raw = np.concatenate([self.raw, np.full(n_new, 0.5)])   # recomputed next
        self._pct = None

    def _reset_uniform(self) -> None:
        avail = remaining_deck(self.hero + self.board)
        self.hands = random_hands(self.rng, avail, self.size)
        self.weights = np.ones(len(self.hands))
        self.scale = 1.0
        self.records = []
        self.raw = self.scorer.equities(self.hands) if self.street == 0 else np.full(len(self.hands), 0.5)

    def _compute_street_strength(self) -> None:
        if self.street == 0:
            return
        need = 5 - len(self.board)
        avail = remaining_deck(self.hero + self.board)
        if need == 0:
            runouts = [()]
        else:
            m = max(1, self.runouts)
            runouts = [tuple(int(c) for c in self.rng.choice(avail, size=need, replace=False))
                       for _ in range(m)]
        cols, cdfs, all_vals = [], [], []
        for ro in runouts:
            vals = eval_many(self.evaluate, self.evaluate_batch, self.hands, self.board + ro)
            all_vals.append(vals)
            ok = ~np.isnan(vals)
            pct = np.full(len(vals), np.nan)
            if ok.any():
                pct[ok] = weighted_percentile(vals[ok], self.weights[ok])
            cols.append(pct)
            cdfs.append(StrengthCDF(vals[ok], self.weights[ok]))
        self._street_vals = (self.street, self._hands_ver, all_vals)   # reused by abs_strength
        self.raw = _nanmean_cols(cols)
        self.street_strength[self.street] = StreetStrength(self.board, runouts, cdfs)
        self._pct = None

    # ------------------------------------------------------------ absolute strength
    def abs_strength(self) -> np.ndarray:
        """Each particle's strength percentile among RANDOM hands (equal weights:
        the particles are uniform draws), not within the posterior range."""
        key = (self.street, self._hands_ver, len(self.board))
        if getattr(self, "_abs_key", None) == key:
            return self._abs
        if self.street == 0 or self.street not in self.street_strength:
            # preflop, raw holds exactly scorer.equities(hands)
            out = self.scorer.percentiles(self.raw) if self.street == 0 \
                else np.full(len(self.hands), 0.5)
        else:
            ss = self.street_strength[self.street]
            cols = []
            ones = np.ones(len(self.hands))
            sv = getattr(self, "_street_vals", None)
            cached = sv[2] if (sv is not None and sv[0] == self.street and sv[1] == self._hands_ver) else None
            for i, ro in enumerate(ss.runouts):
                vals = cached[i] if cached is not None else \
                    eval_many(self.evaluate, self.evaluate_batch, self.hands, self.board + ro)
                ok = ~np.isnan(vals)
                pct = np.full(len(vals), np.nan)
                if ok.any():
                    pct[ok] = weighted_percentile(vals[ok], ones[ok])
                cols.append(pct)
            out = _nanmean_cols(cols)
        self._abs_key, self._abs = key, out
        return out

    def abs_strength_mean(self) -> float:
        """Posterior mean absolute strength of the opponent's range (0.5 = random)."""
        w = self.weights
        t = float(w.sum())
        if t <= 0 or len(w) == 0:
            return 0.5
        return float((w * self.abs_strength()).sum() / t)

    # ------------------------------------------------------------ exact equity
    def expected_outcomes(self, hole, bands=None) -> Optional[np.ndarray]:
        """Our expected showdown result (1 win, .5 tie, 0 loss) against each
        particle when it needs no Monte Carlo, else None:
          * river: exact, one batch evaluation of the whole range;
          * preflop with a band table: the precomputed equity of our hand vs the
            particle's preflop decile."""
        if len(self.board) == 5:
            hero = self.evaluate(tuple(int(c) for c in hole), self.board)
            vals = eval_many(self.evaluate, self.evaluate_batch, self.hands, self.board)
            out = np.where(vals < hero, 1.0, np.where(vals == hero, 0.5, 0.0))
            return np.where(np.isnan(vals), 0.5, out)
        if self.street == 0 and bands is not None:
            row = bands.row(hole)
            if row is None:
                return None
            from .preflop import band_of_pct
            b = band_of_pct(self.scorer.percentiles(self.raw), bands.n_bands)
            return row[b]
        return None

    # ------------------------------------------------------------ actions
    def action_probs(self, freqs, **params) -> np.ndarray:
        """(3, n) P(fold / passive / aggressive | hand) in the current range."""
        params.setdefault("width", self.s.smooth_width)
        params.setdefault("floor", self.s.likelihood_floor)
        return action_probs(self.percentiles(), freqs, weights=self.weights, **params)

    def observe(self, outcome: int, freqs, tilt=None, **params) -> None:
        """Bayes update for one opponent action (outcome: 0 fold, 1 passive, 2 aggressive).

        ``tilt`` (preflop only): (sum, n, prior_n, margin) of the preflop percentile
        of hands they showed after this action in this context.  When that measured
        mean (shrunk toward the model's own posterior mean with prior_n hands) is
        above the posterior mean by more than margin, the likelihood is tilted by
        exp(lam * (range percentile - 1)), lam chosen so the posterior mean preflop
        percentile matches it (never loosens the range)."""
        params.setdefault("width", self.s.smooth_width)
        params.setdefault("floor", self.s.likelihood_floor)
        live = self.weights > 0
        cdf = StrengthCDF(self.raw[live], self.weights[live])
        n = len(self.weights)
        # evaluate on the hands and on a percentile grid in one call (grid weight 0),
        # so the stored grid curve is exactly the calibrated curve applied here
        pct_all = np.concatenate([self.percentiles(), GRID])
        w_all = np.concatenate([self.weights, np.zeros(len(GRID))])
        probs = action_probs(pct_all, freqs, weights=w_all, **params)[outcome]
        if tilt is not None and self.street == 0:
            lam = self._tilt_lambda(probs[:n], pct_all[:n], tilt)
            if lam > 0:
                probs = probs * np.exp(lam * (pct_all - 1.0))
        lik = probs[:n]
        self.records.append(Record(self.street, cdf, probs[n:].copy()))
        self.weights = self.weights * lik
        if self.weights.sum() <= 0:
            self.weights = np.ones(len(self.weights))
        self._renorm()
        self.maybe_refill()

    def _tilt_lambda(self, lik: np.ndarray, pct: np.ndarray, tilt) -> float:
        total, n_shown, prior_n, margin = tilt
        a = self.abs_strength()                  # preflop: percentile among random hands
        w0 = self.weights * lik
        t0 = float(w0.sum())
        if t0 <= 0 or n_shown <= 0:
            return 0.0
        mu0 = float((w0 * a).sum() / t0)
        target = (float(total) + prior_n * mu0) / (n_shown + prior_n)
        if target <= mu0 + margin:
            return 0.0

        def mean_at(lam):
            w = w0 * np.exp(lam * (pct - 1.0))
            t = float(w.sum())
            return float((w * a).sum() / t) if t > 0 else mu0
        hi = 40.0
        if mean_at(hi) <= target:
            return hi
        lo = 0.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if mean_at(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------ refill
    def maybe_refill(self) -> bool:
        if self.ess() >= self.s.ess_min_frac * self.size:
            return False
        room = self.s.max_range_size - len(self.hands)
        n_new = min(int(self.s.refill_frac * self.size), room)
        if n_new < 16:
            # at the cap: drop the lightest hands, then refill
            keep = np.argsort(self.weights)[-max(self.size // 2, 1):]
            self.hands, self.weights, self.raw = self.hands[keep], self.weights[keep], self.raw[keep]
            self._pct = None
            n_new = min(int(self.s.refill_frac * self.size), self.s.max_range_size - len(self.hands))
            if n_new < 16:
                return False
        self._refill(n_new)
        return True

    def replay_weights(self, hands: np.ndarray) -> np.ndarray:
        """True posterior weight (product of likelihoods) for arbitrary hands."""
        w = np.ones(len(hands))
        raw_by_street = {}
        for rec in self.records:
            raw = raw_by_street.get(rec.street)
            if raw is None:
                if rec.street == 0:
                    raw = self.scorer.equities(hands)
                else:
                    raw = self.street_strength[rec.street].raw(hands, self.evaluate, self.evaluate_batch)
                raw_by_street[rec.street] = raw
            w *= np.interp(rec.cdf(raw), GRID, rec.grid_lik)
        return w

    def current_raw(self, hands: np.ndarray) -> np.ndarray:
        if self.street == 0:
            return self.scorer.equities(hands)
        return self.street_strength[self.street].raw(hands, self.evaluate, self.evaluate_batch)

    def _refill(self, n_new: int) -> None:
        avail = remaining_deck(self.hero + self.board)
        new = random_hands(self.rng, avail, n_new)
        w_true = self.replay_weights(new)
        raw_new = self.current_raw(new)
        self.hands = np.concatenate([self.hands, new])
        self.weights = np.concatenate([self.weights, w_true / self.scale])
        self.raw = np.concatenate([self.raw, raw_new])
        self.refills += 1
        self._renorm()
