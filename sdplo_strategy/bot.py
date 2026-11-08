"""StrategyBot: range tracking + opponent model + EV decisions, wrapped in safety.

    bot = StrategyBot(evaluate=core_eval, equity_fn=core_equity)
    action = bot.act(observation)          # always legal, never raises
    bot.on_hand_result(result)             # learn from the finished hand

State (opponent model, timing) persists across hands for the life of the
object; per-hand state (the range) resets when the hand number or our hole
cards change.  If the result hook is never called, the bot learns from the
last history it saw instead.
"""
from __future__ import annotations

import collections
import itertools
import math
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .cards import BOARD_LEN, remaining_deck, random_hands
from .decision import DecisionEngine, Spot
from .equity import EquityEngine, make_outcomes_fn
from .history import infer_button, opponent_line, scare_card, walk
from .opponent import POT_SIZE, Beta, OpponentModel, Priors, outcome_index
from .preflop import PreflopScorer, PreflopTable
from .ranges import RangeTracker, eval_many
from .settings import Settings
from .types import CALL, CHECK, FOLD, OPP, RAISE, Action, HandResult, Observation


# ------------------------------------------------------------------ legality


def _is_int(x) -> bool:
    return x is None or float(x).is_integer()


def integer_chips(obs: Observation) -> bool:
    return all(_is_int(v) for v in (obs.pot, obs.my_stack, obs.to_call,
                                     obs.min_raise_to, obs.max_raise_to))


def fallback_action(obs: Observation, settings: Settings) -> Action:
    """Check if free, else call if cheap, else fold."""
    if obs.to_call <= 0:
        return Action(CHECK)
    cheap = (obs.to_call <= settings.cheap_call_pot_frac * max(obs.pot, 0)
             and obs.to_call <= settings.cheap_call_stack_frac * max(obs.my_stack, 0))
    return Action(CALL) if cheap else Action(FOLD)


def legalize(action: Action, obs: Observation, settings: Optional[Settings] = None) -> Action:
    """Map any action onto a legal one for this observation."""
    settings = settings or Settings()
    to_call = obs.to_call
    kind = action.kind
    if kind == FOLD:
        return Action(CHECK) if to_call <= 0 else Action(FOLD)
    if kind == CHECK:
        return Action(CHECK) if to_call <= 0 else fallback_action(obs, settings)
    if kind == CALL:
        return Action(CHECK) if to_call <= 0 else Action(CALL)
    # raise
    if not obs.can_raise:
        return Action(CHECK) if to_call <= 0 else Action(CALL)
    lo, hi = float(obs.min_raise_to), float(obs.max_raise_to)
    x = float(action.amount)
    if not math.isfinite(x):
        x = lo
    opts = [o for o in (obs.raise_options or ()) if o is not None]
    if opts:
        # discrete action space: the nearest allowed total (ties -> the smaller)
        best = min(opts, key=lambda o: (abs(float(o) - x), float(o)))
        return Action(RAISE, int(best) if integer_chips(obs) else best)
    x = min(max(x, lo), hi)
    if integer_chips(obs):
        x = float(int(math.floor(x + 0.5)))
        if x < lo:
            x = math.ceil(lo)
        if x > hi:
            x = math.floor(hi)
        x = int(x)
    return Action(RAISE, x)


def preflop_rules_on(s: Settings) -> bool:
    return (s.pf_war_raise_min_pct > 0 or s.pf_3bet_raise_min_pct > 0
            or s.pf_open_fold_below > 0 or s.pf_limp_raised_fold_below > 0)


def preflop_rule_action(s: Settings, history, hero_is_button: bool, pct: float, action: Action,
                        to_call: float) -> Action:
    """Override a preflop decision by the solver rules (Settings.pf_*); else ``action``."""
    pre = [r for r in history if r.street == 0]
    n_raise = sum(1 for r in pre if r.kind == RAISE)
    if not pre:
        if hero_is_button and to_call > 0 and pct < s.pf_open_fold_below:
            return Action(FOLD)
        return action
    if pre[0].kind == CALL:                                  # the button limped
        if (hero_is_button and len(pre) == 2 and pre[1].kind == RAISE and to_call > 0
                and pct < s.pf_limp_raised_fold_below):
            return Action(FOLD)
        if n_raise >= 2 and to_call > 0 and action.kind == RAISE and pct < s.pf_war_raise_min_pct:
            return Action(CALL)
    elif pre[0].kind == RAISE:                               # the button opened
        if (hero_is_button and len(pre) == 2 and n_raise == 2 and to_call > 0
                and action.kind == RAISE and pct < s.pf_3bet_raise_min_pct):
            return Action(CALL)
    return action


# ------------------------------------------------------------------ timing


class TimeKeeper:
    """Keeps decisions inside the per-decision limit and, when the rules give a
    time bank for the match (``match_time_budget``), inside the bank.

    Each decision is charged ``elapsed + overhead - grace`` (the official judge
    does not charge the first 1 ms of a request).  ``hint`` = this decision's
    allowance / the recent full cost of a decision, in [0, 1]: the bot shrinks
    the range sample and the Monte Carlo budget in proportion, and switches to
    the cheap path below ``cheap_hint``.  Without a bank the old multiplicative
    ``scale`` (time_limit * time_target_frac) still applies.
    """

    def __init__(self, settings: Settings):
        self.s = settings
        self.scale = 1.0
        self.n = 0
        self.total = 0.0        # charged seconds
        self.wall = 0.0         # measured seconds per decision (full request work)
        self.max = 0.0
        self.over_target = 0
        self.cheap = 0
        self.decisions_per_hand = 3.0
        self.last_hand = 0
        self.ema_cost = None    # recent full wall cost of a non-cheap decision
        self.ema_lite = None    # recent wall cost of a lite decision (lite_mode)
        self.load = 1.0         # recent cost / nominal cost of the mode used (lite_mode)
        self.last_elapsed = 0.0
        self.modes = {"full": 0, "lite": 0, "cheap": 0}
        self.contention = 1.0   # EMA of wall / CPU time of our own work (1 = a free core)
        self._load_seen = None  # host load average per CPU, refreshed every 25 decisions
        self._load_at = 0

    def overhead(self) -> float:
        """Per-request time we cannot see (server, JSON).  Under CPU contention it
        grows like our own decisions do, so it is scaled by how slow our recent
        decisions run vs ``overhead_ref_cost`` (never below the base estimate)."""
        o = self.s.request_overhead
        ref = self.s.overhead_ref_cost
        if self.s.contention_overhead:
            # the server's own parsing/JSON runs in our process and stretches like our
            # decisions do when the core is shared or throttled; on an oversubscribed
            # host most of the stretch is wake-up latency we cannot time, so the host's
            # load per CPU is a second signal
            return o * min(4.0, max(1.0, self.contention, self.load_factor()))
        if self.s.lite_mode:
            if ref > 0:
                o *= min(3.0, max(1.0, self.load))
        elif ref > 0 and self.ema_cost:
            o *= min(3.0, max(1.0, self.ema_cost / ref))
        return o

    def load_factor(self) -> float:
        k = self.s.load_overhead_slope
        if k <= 0:
            return 1.0
        if self._load_seen is None or self.n - self._load_at >= 25:
            try:
                import os
                self._load_seen = os.getloadavg()[0] / max(1, os.cpu_count() or 1)
            except Exception:
                self._load_seen = 0.0
            self._load_at = self.n
        return min(4.0, max(1.0, 1.0 + k * (self._load_seen - self.s.load_overhead_knee)))

    def observe_contention(self, wall: float, cpu: float) -> None:
        if cpu > 1e-4 and wall > 0:
            r = min(6.0, max(0.8, wall / cpu))
            self.contention = 0.9 * self.contention + 0.1 * r

    def _nominal(self, mode: str) -> float:
        ref = self.s.overhead_ref_cost or 0.0015
        return {"full": ref, "lite": ref * 0.6, "cheap": ref * 0.2}[mode]

    def mode(self, hand_number: int = 0) -> str:
        """full / lite / cheap for this decision.  Without ``lite_mode`` this is the
        old rule (cheap below cheap_hint, else full with scaled samples)."""
        if not self.s.lite_mode:
            return "cheap" if self.cheap_mode(hand_number) else "full"
        if self.s.match_time_budget is None:
            return "full"
        if self.remaining() <= 0:
            return "cheap"
        allow = self.limit(hand_number)
        ef = self.ema_cost or self._nominal("full")
        el = self.ema_lite or 0.6 * ef
        if allow >= ef:
            return "full"
        if allow >= el:
            return "lite"
        return "cheap"

    def charge(self, elapsed: float) -> float:
        return max(0.0, elapsed + self.overhead() - self.s.request_grace)

    def remaining(self) -> Optional[float]:
        if self.s.match_time_budget is None:
            return None
        return self.s.match_time_budget - self.total

    def limit(self, hand_number: int = 0) -> float:
        """Wall seconds this decision may use."""
        lim = self.s.time_limit
        if self.s.match_time_budget is not None:
            hands_left = max(1, self.s.hands_per_match - hand_number)
            per = max(0.0, self.remaining()) / (hands_left * self.decisions_per_hand)
            lim = min(lim, per + self.s.request_grace - self.overhead())
        return max(lim, 1e-4)

    def hint(self, hand_number: int = 0) -> float:
        if self.s.match_time_budget is None or not self.ema_cost:
            return 1.0
        return float(min(1.0, self.limit(hand_number) / self.ema_cost))

    def cheap_mode(self, hand_number: int = 0) -> bool:
        if self.s.match_time_budget is not None and self.remaining() <= 0:
            return True
        lim = self.limit(hand_number)
        if self.s.cheap_below > 0 and lim < self.s.cheap_below:
            return True
        return self.hint(hand_number) < self.s.cheap_hint

    def budget(self, base: int, hand_number: int = 0) -> int:
        if self.s.match_time_budget is not None:
            return max(16, int(base * self.hint(hand_number)))
        if self.s.time_free_without_bank:
            return max(16, int(base))       # no bank (tests, sweeps): never depend on the clock
        return max(16, int(base * self.scale))

    def record(self, elapsed: float, hand_number: int = 0, cheap: bool = False,
               mode: Optional[str] = None) -> None:
        if self.s.lite_mode and mode is not None:
            self._record_tiered(elapsed, hand_number, mode)
            return
        self.n += 1
        self.last_elapsed = elapsed
        self.wall += elapsed
        self.total += self.charge(elapsed)
        self.max = max(self.max, elapsed)
        self.last_hand = max(self.last_hand, hand_number)
        if self.n >= 30:
            self.decisions_per_hand = min(6.0, max(1.5, self.n / (self.last_hand + 1)))
        if not cheap:
            self._update_cost(elapsed)
        elif self.ema_cost is not None:
            # cheap decisions say nothing about the full cost: let the estimate decay so
            # a one-off spike (GC, a descheduled process) cannot lock in the cheap path
            self.ema_cost *= 0.97
        target = self.limit(hand_number) * self.s.time_target_frac
        if elapsed > target:
            self.over_target += 1
            self.scale = max(self.s.min_budget_scale, self.scale * max(0.5, target / elapsed))
        elif elapsed < 0.5 * target:
            self.scale = min(1.0, self.scale * 1.1)

    def _record_tiered(self, elapsed: float, hand_number: int, mode: str) -> None:
        self.n += 1
        self.modes[mode] = self.modes.get(mode, 0) + 1
        self.last_elapsed = elapsed
        self.wall += elapsed
        self.total += self.charge(elapsed)
        self.max = max(self.max, elapsed)
        self.last_hand = max(self.last_hand, hand_number)
        if self.n >= 30:
            self.decisions_per_hand = min(6.0, max(1.5, self.n / (self.last_hand + 1)))
        self._ema_update(mode, elapsed)
        self.load = 0.95 * self.load + 0.05 * min(4.0, max(0.3, elapsed / self._nominal(mode)))

    def _ema_update(self, mode: str, x: float) -> None:
        a = 0.05
        if mode == "full":
            self.ema_cost = x if self.ema_cost is None else (1 - a) * self.ema_cost + a * min(x, 3.0 * self.ema_cost)
            if self.ema_lite is not None:
                self.ema_lite *= 1.0
        elif mode == "lite":
            self.ema_lite = x if self.ema_lite is None else (1 - a) * self.ema_lite + a * min(x, 3.0 * self.ema_lite)
            if self.ema_cost is not None:
                self.ema_cost *= 0.995      # retry full now and then
        else:
            # slow decay so the tiers are retried now and then, never below half nominal
            if self.ema_cost is not None:
                self.ema_cost = max(0.5 * self._nominal("full"), self.ema_cost * 0.995)
            if self.ema_lite is not None:
                self.ema_lite = max(0.5 * self._nominal("lite"), self.ema_lite * 0.99)

    def _update_cost(self, elapsed: float) -> None:
        a = 0.05
        if self.ema_cost is None:
            self.ema_cost = elapsed
            return
        # robust: one slow outlier moves the estimate by at most a * 3 * ema
        self.ema_cost = (1 - a) * self.ema_cost + a * min(elapsed, 3.0 * self.ema_cost)

    def adjust(self, inner: float, full: float, cheap: bool = False, mode: Optional[str] = None) -> None:
        """The caller measured more work for this decision than act() saw (format
        conversion, result hooks delivered with the request): charge the rest."""
        if full <= inner:
            return
        self.total += self.charge(full) - self.charge(inner)
        self.wall += full - inner
        self.max = max(self.max, full)
        if self.s.lite_mode and mode is not None:
            if mode == "full" and self.ema_cost is not None:
                self.ema_cost += 0.05 * min(full - inner, 3.0 * self.ema_cost)
            elif mode == "lite" and self.ema_lite is not None:
                self.ema_lite += 0.05 * min(full - inner, 3.0 * self.ema_lite)
            return
        if not cheap and self.ema_cost is not None:
            self.ema_cost += 0.05 * min(full - inner, 3.0 * self.ema_cost)


# ------------------------------------------------------------------ bot


@dataclass
class _Hand:
    number: int
    hole: tuple
    range: RangeTracker
    points_seen: int = 0
    last_obs: Optional[Observation] = None


def full_house_or_better(hole, board) -> bool:
    """Made hand (exactly 2 hole + 3 board cards) is a full house, flush, quads or
    straight flush - every category from a full house up under either ranking."""
    if len(board) < 3:
        return False
    for h2 in itertools.combinations(hole, 2):
        for b3 in itertools.combinations(board, 3):
            cards = h2 + b3
            if len({c & 3 for c in cards}) == 1:
                return True
            cnt = sorted(collections.Counter(c >> 2 for c in cards).values(), reverse=True)
            if cnt[0] >= 4 or (cnt[0] == 3 and cnt[1] == 2):
                return True
    return False


class StrategyBot:
    def __init__(self, evaluate: Callable, equity_fn: Optional[Callable] = None,
                 outcomes_fn: Optional[Callable] = None, evaluate_batch: Optional[Callable] = None,
                 settings: Optional[Settings] = None, preflop_table: Optional[PreflopTable] = None,
                 priors: Optional[Priors] = None, backend: str = "outcomes", band_table=None):
        self.s = settings or Settings()
        self.rng = np.random.default_rng(self.s.seed)
        self.evaluate = evaluate
        self.evaluate_batch = evaluate_batch
        if backend == "scalar" and equity_fn is not None:
            self.equity = EquityEngine(equity_fn=equity_fn)
        else:
            self.equity = EquityEngine(outcomes_fn=outcomes_fn or make_outcomes_fn(evaluate, evaluate_batch))
        if priors is None:
            priors = Priors()
            for k, v in (self.s.extra or {}).items():
                if k.startswith("prior_") and hasattr(priors, k[6:]):
                    setattr(priors, k[6:], v)
        self.model = OpponentModel(priors)
        self.scorer = PreflopScorer(preflop_table)
        self.engine = DecisionEngine(self.s, self.equity, self.model, bands=band_table)
        self.timer = TimeKeeper(self.s)
        self.value_rate = {st: Beta(0.3, 6.0) for st in (1, 2, 3)}
        self.hand: Optional[_Hand] = None
        self.errors = 0
        self.fallbacks = 0
        self.last_error: Optional[str] = None
        self.pot_mismatches = 0
        self.last_decision: dict = {}
        self._was_cheap = False
        self._pending_bets: list = []     # (hand number, model's raw fold share) for our postflop bets
        self._pending_eq: list = []       # (spot key, raw equity estimate, hole, board) this hand
        self._eq_calib: dict = {}         # spot key -> [sum of (estimate - true equity), n]
        self._mode = "full"               # time tier of the current decision (full / lite / cheap)

    # -------------------------------------------------------------- public
    def act(self, obs: Observation, rng: Optional[np.random.Generator] = None) -> Action:
        """Always returns a legal action.  ``rng``: per-decision generator (the core
        seeds one from the situation so mirrored duplicate hands match)."""
        t0 = time.perf_counter()
        self._was_cheap = False
        self._mode = "full"
        if rng is not None:
            self.rng = rng
        try:
            action = self._act(obs, t0)
        except Exception:
            self.errors += 1
            self.fallbacks += 1
            self.last_error = traceback.format_exc(limit=4)
            action = self._safe_fallback(obs)
        try:
            action = legalize(action, obs, self.s)
        except Exception:
            self.errors += 1
            action = Action(CHECK) if getattr(obs, "to_call", 1) <= 0 else Action(FOLD)
        self.timer.record(time.perf_counter() - t0, getattr(obs, "hand_number", 0) or 0,
                          cheap=self._was_cheap, mode=("cheap" if self._was_cheap else self._mode))
        return action

    def on_hand_result(self, result: HandResult) -> None:
        try:
            self._learn(result.history, result.hero_is_button, result.small_blind, result.big_blind,
                        result.opp_hole, result.board, result.hero_hole)
        except Exception:
            self.errors += 1
            self.last_error = traceback.format_exc(limit=4)
        self.hand = None

    # -------------------------------------------------------------- internals
    def _safe_fallback(self, obs) -> Action:
        try:
            return fallback_action(obs, self.s)
        except Exception:
            return Action(FOLD)

    def _button(self, obs: Observation) -> bool:
        if obs.hero_is_button is not None:
            return bool(obs.hero_is_button)
        guess = infer_button(obs.history)
        if guess is not None:
            return guess
        return obs.street == 0   # nothing has happened yet: first to act preflop is the button

    def _start_hand(self, obs: Observation) -> _Hand:
        if self.hand is not None and self.hand.last_obs is not None:
            prev = self.hand.last_obs   # result hook never came: learn from what we saw
            try:
                self._learn(prev.history, self._button(prev), prev.small_blind, prev.big_blind,
                            None, prev.board, prev.hole)
            except Exception:
                self.errors += 1
        size = self.s.range_size
        runouts = self.s.strength_runouts
        if self.s.lite_mode:
            if self._mode != "full":
                size = min(size, self.s.lite_range_size)
        elif self.s.match_time_budget is not None:
            h = self.timer.hint(obs.hand_number)
            if h < 1.0:
                size = max(48, int(size * max(0.35, h)))
        elif self.timer.scale < 0.3 and not self.s.time_free_without_bank:
            size = max(200, int(size * max(0.25, 2 * self.timer.scale)))
            runouts = 1
        rt = RangeTracker(obs.hole, self.s, self.rng, self.evaluate, self.scorer,
                          self.evaluate_batch, size=size, runouts=runouts)
        return _Hand(obs.hand_number, tuple(obs.hole), rt)

    def _cheap_act(self, obs: Observation) -> Action:
        """Fast path when the time bank is nearly spent (well under a millisecond):
        preflop table equity, or a tiny Monte Carlo vs random hands postflop, then
        pot odds.  No range tracking."""
        self.timer.cheap += 1
        self._was_cheap = True
        if obs.street == 0:
            eq = self.scorer.equity(obs.hole)
        else:
            avail = remaining_deck(tuple(obs.hole) + tuple(obs.board))
            opp = random_hands(self.rng, avail, 32)
            _, out = self.equity.outcomes_fn(obs.hole, obs.board, opp, np.ones(len(opp)), 32, self.rng) \
                if self.equity.outcomes_fn is not None else (None, np.array([0.5]))
            eq = float(np.mean(out)) if len(out) else 0.5
        strong = eq >= (0.58 if obs.street == 0 else 0.75)
        if strong and obs.can_raise:
            opts = [o for o in (obs.raise_options or (obs.max_raise_to,)) if o is not None]
            return Action(RAISE, max(opts))
        if obs.to_call <= 0:
            return Action(CHECK)
        need = obs.to_call / max(obs.pot + obs.to_call, 1e-9)
        return Action(CALL) if eq >= need + 0.05 else Action(FOLD)

    def _cheap_act_v2(self, obs: Observation) -> Action:
        """Cheap path, v2 (~0.3 ms): preflop table equity priced against the range the
        opponent's raise frequency implies (band table), postflop a small Monte Carlo
        vs random hands shifted by their aggression this street; value bets and
        gated bluffs from the opponent model's fold rates.  No range tracking."""
        self.timer.cheap += 1
        self._was_cheap = True
        st = obs.street
        hist = obs.history
        raises_st = sum(1 for r in hist if r.street == st and r.kind == RAISE)
        opp_aggr = any(r.actor == OPP and r.kind == RAISE and r.street == st for r in hist)
        need = obs.to_call / max(obs.pot + obs.to_call, 1e-9) if obs.to_call > 0 else 0.0
        opts = sorted(o for o in (obs.raise_options or (obs.max_raise_to,)) if o is not None) \
            if obs.can_raise else []
        big = opts[-1] if opts else None
        small = opts[0] if opts else None
        if st == 0:
            eq = self.scorer.equity(obs.hole)
            pct = self.scorer.percentile(eq)
            if raises_st == 0:
                if obs.to_call > 0:                       # button, first in
                    if pct >= 0.5 and big is not None:
                        return Action(RAISE, big)
                    return Action(CALL) if pct >= 0.15 else Action(FOLD)
                if pct >= 0.7 and big is not None:        # big blind after a limp
                    return Action(RAISE, big)
                return Action(CHECK)
            q = min(0.9, max(0.05, self.model.pfr.p * (0.5 ** (raises_st - 1))))
            eq_vs = eq - 0.25 * (1.0 - q)
            bands = self.engine.bands
            if bands is not None:
                row = bands.row(obs.hole)
                if row is not None:
                    k = max(1, min(len(row), int(round(q * len(row)))))
                    eq_vs = float(np.mean(row[len(row) - k:]))
            if eq_vs >= 0.6 and big is not None:
                return Action(RAISE, big)
            if obs.to_call <= 0:
                return Action(CHECK)
            return Action(CALL) if eq_vs * 0.92 >= need else Action(FOLD)
        avail = remaining_deck(tuple(obs.hole) + tuple(obs.board))
        opp = random_hands(self.rng, avail, 48)
        _, out = self.equity.outcomes_fn(obs.hole, obs.board, opp, np.ones(len(opp)), 48, self.rng) \
            if self.equity.outcomes_fn is not None else (None, np.array([0.5]))
        eq = float(np.mean(out)) if len(out) else 0.5
        if opp_aggr:
            eq -= 0.15 * self.model.informativeness(st)
        if obs.to_call > 0:
            if eq >= 0.80 and big is not None:
                return Action(RAISE, big)
            return Action(CALL) if eq >= need + 0.04 else Action(FOLD)
        ftb = self.model.fold_to_bet(st, 1.0)
        if eq >= 0.66 and big is not None:
            return Action(RAISE, big)
        if eq >= 0.55 and small is not None and ftb < 0.3:
            return Action(RAISE, small)               # thin value against callers
        if eq < 0.35 and big is not None and ftb >= 0.55 and st < 3 and self.rng.random() < 0.6:
            return Action(RAISE, big)                 # bluff an over-folder
        return Action(CHECK)

    def _act(self, obs: Observation, t0: float) -> Action:
        self._mode = self.s.force_mode or self.timer.mode(obs.hand_number)
        if self._mode == "cheap":
            a = self._cheap_act_v2(obs) if self.s.cheap_version == 2 else self._cheap_act(obs)
            if obs.street == 0 and preflop_rules_on(self.s):
                a = preflop_rule_action(self.s, obs.history, self._button(obs),
                                        self.scorer.percentile(self.scorer.equity(obs.hole)), a, obs.to_call)
            return a
        lite = self._mode == "lite"
        if (self.hand is None or self.hand.number != obs.hand_number
                or self.hand.hole != tuple(obs.hole)):
            self.hand = self._start_hand(obs)
        hs = self.hand
        hs.last_obs = obs
        hs.range.rng = self.rng
        hs.range.top_up_enabled = not lite
        button = self._button(obs)
        points, state = walk(obs.history, button, obs.small_blind, obs.big_blind,
                             split_4bet=self.s.split_preflop_4bet, split_open_size=self.s.split_open_size)
        if abs(state.pot - obs.pot) > 1e-6:
            self.pot_mismatches += 1
        split_obs = self.s.split_bluff_observe

        def bluff_of(ctx, street=0):
            if (self.s.scare_bluff_split and ctx in ("lead", "checked_to")
                    and scare_card(obs.board, street)):
                return self.model.bluff_scare.p
            return self.model.bluff_share(ctx, split=True) if split_obs else self.model.bluff_share(ctx)
        kappa = (self.model.informativeness(0), self.model.informativeness(1))
        # folds/calls are at least as strength-driven as bets (max: equity bots read 1.0
        # on aggression; a chart bot's semi-bluffs read lower than its folds)
        kappa_call = max(kappa[1], self.model.call_informativeness()) if self.s.split_informativeness else None
        for p in points[hs.points_seen:]:
            if p.actor != OPP:
                continue
            hs.range.set_board(obs.board[:BOARD_LEN[p.street]])
            freqs = self.model.freqs(p.context, p.street, p.frac)
            if (self.s.size_split_aggr and p.kind == RAISE and p.size_frac >= POT_SIZE
                    and p.street in self.s.size_split_streets):
                q = self.model.pot_size_share(p, self.s.size_split_min_n, self.s.size_split_gate_n,
                                              self.s.size_split_margin)
                if q is not None and q < 1.0:
                    f0, p0, a0 = (float(x) for x in freqs)
                    freqs = np.array([f0, p0 + a0 * (1.0 - q), a0 * q])
            bshare = bluff_of(p.context, p.street) * (0.5 if p.street == 0 else 1.0)
            if (self.s.size_bluff_gate > 0 and p.kind == RAISE and p.street > 0
                    and p.context in ("lead", "checked_to")):
                sb = self.model.size_bluff_share(p.size_frac, p.street, self.s.size_bluff_gate)
                if sb is not None:
                    bshare = sb
            tilt = None
            if (self.s.pf_tilt_gate > 0 and p.street == 0 and p.kind == RAISE
                    and p.context in self.s.pf_tilt_contexts):
                shown = self.model.pre_shown.get(p.context)
                if shown is not None and shown[1] >= self.s.pf_tilt_gate:
                    tilt = (shown[0], shown[1], self.s.pf_tilt_prior, self.s.pf_tilt_margin)
            hs.range.observe(outcome_index(p.kind), freqs, tilt=tilt,
                             bluff_share=bshare,
                             informativeness=kappa[0] if p.street == 0 else kappa[1],
                             call_informativeness=None if p.street == 0 else kappa_call)
        hs.points_seen = len(points)
        hs.range.set_board(obs.board)

        same_street = state.street == obs.street
        my_bet = obs.my_street_bet if obs.my_street_bet is not None else (
            state.bets[0] if same_street else 0.0)
        opp_bet = obs.opp_street_bet if obs.opp_street_bet is not None else (
            state.bets[1] if same_street else 0.0)
        pre_pct = 0.5
        if obs.street == 0:
            pre_pct = self.scorer.percentile(self.scorer.equity(obs.hole))
        spot = Spot(hole=tuple(obs.hole), board=tuple(obs.board), street=obs.street,
                    pot=float(obs.pot), to_call=float(obs.to_call), my_bet=float(my_bet),
                    opp_bet=float(opp_bet), my_stack=float(obs.my_stack), opp_stack=float(obs.opp_stack),
                    min_raise_to=obs.min_raise_to, max_raise_to=obs.max_raise_to,
                    big_blind=float(obs.big_blind), in_position=button,
                    raises_this_street=state.raises if same_street else 0,
                    integer_chips=integer_chips(obs), preflop_pct=pre_pct,
                    raise_options=obs.raise_options)
        base = self.s.sample_budget * (self.s.preflop_budget_frac if obs.street == 0 else 1.0)
        if self.s.lite_mode:
            n = max(16, int(base * (self.s.lite_sample_frac if lite else 1.0)))
            deadline = t0 + self.s.time_limit * self.s.hard_stop_frac
        else:
            n = self.timer.budget(int(base), obs.hand_number)
            deadline = t0 + self.timer.limit(obs.hand_number) * self.s.hard_stop_frac
            if self.s.match_time_budget is None and self.s.time_free_without_bank:
                deadline = None
        calib_key = None
        facing_raise = False
        if obs.street > 0 and obs.to_call > 0:
            last = next((q for q in reversed(points) if q.actor == OPP), None)
            if last is not None and last.kind == RAISE and last.street == obs.street:
                kind = "raise" if last.context in ("facing_bet", "facing_raise") else "bet"
                facing_raise = kind == "raise"
                track = (self.s.eq_calibrate and obs.street in self.s.eq_calib_streets
                         and kind in self.s.eq_calib_kinds)
                if track or (self.s.raise_cap and facing_raise and self.s.raise_cap_gate == "calib"):
                    calib_key = (obs.street, kind, bool(last.size_frac >= POT_SIZE))
                if track:
                    spot.eq_shift = self._eq_shift(calib_key)
        eq, options, used = self.engine.evaluate(spot, hs.range, n, self.rng, deadline, lite=lite)
        if calib_key is not None:
            self._pending_eq.append((calib_key, self.engine.last_raw_eq, tuple(obs.hole), tuple(obs.board)))
        if (self.s.raise_cap and facing_raise and obs.street in self.s.raise_cap_streets
                and self._raise_cap_active(obs.street) and not full_house_or_better(obs.hole, obs.board)):
            kept = [o for o in options if o.action.kind != RAISE
                    or o.info.get("eq_reraise", 0.0) >= self.s.raise_cap_eq_min]
            if any(o.action.kind != RAISE for o in kept):
                options = kept
        vr = self.value_rate.get(obs.street)
        opt, reason = self.engine.choose(spot, eq, options, self.rng, vr.p if vr else 0.3)
        action = opt.action
        if obs.street == 0 and preflop_rules_on(self.s):
            action = preflop_rule_action(self.s, obs.history, button, pre_pct, action, obs.to_call)
            if action is not opt.action:
                reason = "preflop_rule"
        if opt.action.kind == RAISE and action is opt.action:
            # the conditions under which their response frequencies are learned
            if self.s.strength_fold_beta > 0 and not lite:
                self.model.observe_bet_strength(obs.street, hs.range.abs_strength_mean())
            if obs.street > 0 and "p_fold_raw" in opt.info:
                self._pending_bets.append((obs.hand_number, float(opt.info["p_fold_raw"])))
        if vr is not None and any(o.action.kind == RAISE for o in options):
            vr.observe_bool(opt.action.kind == RAISE and reason == "ev"
                            and opt.info.get("eq_call", 0.0) >= 0.5)
        self.last_decision = {"eq": eq, "samples": used, "reason": reason,
                              "ess": hs.range.ess(), "range": len(hs.range.hands),
                              "options": [(o.action.kind, o.action.amount, round(o.ev, 3),
                                           round(o.info.get("p_fold", 0.0), 3), round(o.info.get("p_raise", 0.0), 3),
                                           round(o.info.get("eq_call", 0.0), 3), round(o.info.get("eq_reraise", 0.0), 3))
                                          for o in options]}
        return action

    def _learn(self, history, hero_is_button, sb, bb, opp_hole, board, hero_hole) -> None:
        if hero_is_button is None:
            hero_is_button = infer_button(history)
            if hero_is_button is None:
                return
        points, _ = walk(history, hero_is_button, sb, bb, split_4bet=self.s.split_preflop_4bet,
                         split_open_size=self.s.split_open_size)
        line = opponent_line(points)
        self._settle_bets(points)
        pend, self._pending_eq = self._pending_eq, []
        if pend and opp_hole is not None and len(opp_hole) == 5 and board is not None and len(board) == 5:
            for key, est, hole, bd in pend:
                c = self._eq_calib.setdefault(key, [0.0, 0])
                c[0] += est - self._true_equity(hole, tuple(opp_hole), bd)
                c[1] += 1
        shown = shown_pre = None
        if opp_hole is not None and len(opp_hole) == 5 and board is not None and len(board) == 5:
            shown = self._showdown_strength(tuple(opp_hole), tuple(board))
            shown_pre = self.scorer.percentile(self.scorer.equity(tuple(opp_hole)))
        scare = None
        if self.s.scare_bluff_split and board is not None and len(board) == 5:
            scare = {st for st in (2, 3) if scare_card(board, st)}
        self.model.record_hand(points, line, shown, shown_pre, scare)

    def _settle_bets(self, points) -> None:
        """Match our postflop bets this hand with whether the opponent folded to them."""
        pend, self._pending_bets = self._pending_bets, []
        if not pend:
            return
        outcomes = []
        for i, p in enumerate(points):
            if p.actor == OPP or p.kind != RAISE or p.street == 0:
                continue
            nxt = next((q for q in points[i + 1:] if q.actor == OPP), None)
            outcomes.append(nxt is not None and nxt.kind == FOLD and nxt.street == p.street)
        for (hn, pred), folded in zip(pend[-len(outcomes):] if outcomes else [], outcomes[-len(pend):]):
            self.engine.record_fold_outcome(pred, folded)

    def _raise_cap_active(self, street: int) -> bool:
        """Raise cap gate (Settings.raise_cap): their raises of our bets proved value-heavy."""
        g = self.s.raise_cap_gate
        if g == "always":
            return True
        if g == "bluff":
            b = self.model.bluff_raise
            return b.n >= self.s.raise_cap_gate_n and b.p <= self.s.raise_cap_bluff_max
        tot = n = 0.0                  # "calib": pooled overestimate over streets and sizes
        for (st, kind, pot), c in self._eq_calib.items():
            if kind == "raise":
                tot += c[0]
                n += c[1]
        return n >= self.s.eq_calib_gate and tot / n >= self.s.raise_cap_bias_min

    def _eq_shift(self, key) -> float:
        """Measured overestimate of our equity in spots like ``key`` (street, kind,
        pot-size): the street's own mean shrunk toward the pool over streets, the
        pool shrunk toward 0; 0 until the pool has eq_calib_gate shown spots."""
        st, kind, pot = key
        k = self.s.eq_calib_k
        tot = n = 0.0
        for s2 in (1, 2, 3):
            c = self._eq_calib.get((s2, kind, pot))
            if c:
                tot += c[0]
                n += c[1]
        if n < self.s.eq_calib_gate:
            return 0.0
        pool = tot / (n + k)
        c = self._eq_calib.get(key, (0.0, 0))
        m = (c[0] + k * pool) / (c[1] + k)
        return float(min(self.s.eq_calib_max, max(0.0, m)))

    def _true_equity(self, hole: tuple, opp: tuple, board: tuple) -> float:
        """Our exact equity vs their actual hand on ``board`` (all runouts)."""
        rest = remaining_deck(tuple(hole) + tuple(opp) + tuple(board))
        need = 5 - len(board)
        runs = list(itertools.combinations(rest.tolist(), need)) if need > 0 else [()]
        boards = np.array([list(board) + list(r) for r in runs], dtype=np.int64)
        if self.evaluate_batch is not None:
            n = len(boards)
            a = np.asarray(self.evaluate_batch(np.tile(np.array(sorted(hole)), (n, 1)), boards))
            b = np.asarray(self.evaluate_batch(np.tile(np.array(sorted(opp)), (n, 1)), boards))
        else:
            a = np.array([self.evaluate(hole, tuple(bd)) for bd in boards])
            b = np.array([self.evaluate(opp, tuple(bd)) for bd in boards])
        return float(((a > b) + 0.5 * (a == b)).mean())

    def _showdown_strength(self, opp_hole: tuple, board: tuple) -> float:
        """Percentile of the shown hand vs random hands on the final board."""
        avail = remaining_deck(opp_hole + board)
        ref = random_hands(self.rng, avail, self.s.showdown_ref_hands)
        vals = eval_many(self.evaluate, self.evaluate_batch, ref, board)
        mine = self.evaluate(opp_hole, board)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return 0.5
        return float(((vals < mine).sum() + 0.5 * (vals == mine).sum()) / len(vals))


# ------------------------------------------------------------------ factory


def make_bot(seed: int = 0, settings: Optional[Settings] = None, name: str = "ours", **overrides):
    """Core-protocol bot for the field round-robin (``--add sdplo_strategy.bot:make_bot:ours``).

    Play-time hand evaluation is the core's fast evaluator (``sdplo.evaluator``
    omaha / omaha_batch, for the configured HandRanking); the preflop table
    (built with ``sdplo.equity``) is loaded if present, else the heuristic is
    used.  ``overrides`` are Settings fields, e.g. ``sample_budget=300``.
    """
    from .adapter import make_core_bot
    from .official import official_settings
    # the official-rules tuning (time bank, sample sizes, realization) is the default
    s = Settings(**vars(settings)) if settings is not None else official_settings()
    for k, v in overrides.items():
        if k.startswith("prior_"):                  # opponent-model prior, e.g. prior_bluff_raise_share
            s.extra = dict(s.extra or {}, **{k: v})
            continue
        if not hasattr(s, k):
            raise TypeError(f"unknown setting {k!r}")
        setattr(s, k, v)
    return make_core_bot(seed=seed, settings=s, name=name, preflop_table="auto")
