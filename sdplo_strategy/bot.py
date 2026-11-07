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

import math
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .cards import BOARD_LEN, remaining_deck, random_hands
from .decision import DecisionEngine, Spot
from .equity import EquityEngine, make_outcomes_fn
from .history import infer_button, opponent_line, walk
from .opponent import Beta, OpponentModel, Priors, outcome_index
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
        self.last_elapsed = 0.0

    def charge(self, elapsed: float) -> float:
        return max(0.0, elapsed + self.s.request_overhead - self.s.request_grace)

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
            lim = min(lim, per + self.s.request_grace - self.s.request_overhead)
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
        return max(16, int(base * self.scale))

    def record(self, elapsed: float, hand_number: int = 0, cheap: bool = False) -> None:
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

    def _update_cost(self, elapsed: float) -> None:
        a = 0.05
        if self.ema_cost is None:
            self.ema_cost = elapsed
            return
        # robust: one slow outlier moves the estimate by at most a * 3 * ema
        self.ema_cost = (1 - a) * self.ema_cost + a * min(elapsed, 3.0 * self.ema_cost)

    def adjust(self, inner: float, full: float, cheap: bool = False) -> None:
        """The caller measured more work for this decision than act() saw (format
        conversion, result hooks delivered with the request): charge the rest."""
        if full <= inner:
            return
        self.total += self.charge(full) - self.charge(inner)
        self.wall += full - inner
        self.max = max(self.max, full)
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

    # -------------------------------------------------------------- public
    def act(self, obs: Observation, rng: Optional[np.random.Generator] = None) -> Action:
        """Always returns a legal action.  ``rng``: per-decision generator (the core
        seeds one from the situation so mirrored duplicate hands match)."""
        t0 = time.perf_counter()
        self._was_cheap = False
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
                          cheap=self._was_cheap)
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
        if self.s.match_time_budget is not None:
            h = self.timer.hint(obs.hand_number)
            if h < 1.0:
                size = max(48, int(size * max(0.35, h)))
        elif self.timer.scale < 0.3:
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

    def _act(self, obs: Observation, t0: float) -> Action:
        if self.timer.cheap_mode(obs.hand_number):
            return self._cheap_act(obs)
        if (self.hand is None or self.hand.number != obs.hand_number
                or self.hand.hole != tuple(obs.hole)):
            self.hand = self._start_hand(obs)
        hs = self.hand
        hs.last_obs = obs
        hs.range.rng = self.rng
        button = self._button(obs)
        points, state = walk(obs.history, button, obs.small_blind, obs.big_blind)
        if abs(state.pot - obs.pot) > 1e-6:
            self.pot_mismatches += 1
        bluff_of = self.model.bluff_share
        kappa = (self.model.informativeness(0), self.model.informativeness(1))
        # folds/calls are at least as strength-driven as bets (max: equity bots read 1.0
        # on aggression; a chart bot's semi-bluffs read lower than its folds)
        kappa_call = max(kappa[1], self.model.call_informativeness()) if self.s.split_informativeness else None
        for p in points[hs.points_seen:]:
            if p.actor != OPP:
                continue
            hs.range.set_board(obs.board[:BOARD_LEN[p.street]])
            freqs = self.model.freqs(p.context, p.street, p.frac)
            hs.range.observe(outcome_index(p.kind), freqs,
                             bluff_share=bluff_of(p.context) * (0.5 if p.street == 0 else 1.0),
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
        n = self.timer.budget(int(base), obs.hand_number)
        deadline = t0 + self.timer.limit(obs.hand_number) * self.s.hard_stop_frac
        eq, options, used = self.engine.evaluate(spot, hs.range, n, self.rng, deadline)
        vr = self.value_rate.get(obs.street)
        opt, reason = self.engine.choose(spot, eq, options, self.rng, vr.p if vr else 0.3)
        if opt.action.kind == RAISE:
            # the conditions under which their response frequencies are learned
            self.model.observe_bet_strength(obs.street, hs.range.abs_strength_mean())
        if vr is not None and any(o.action.kind == RAISE for o in options):
            vr.observe_bool(opt.action.kind == RAISE and reason == "ev"
                            and opt.info.get("eq_call", 0.0) >= 0.5)
        self.last_decision = {"eq": eq, "samples": used, "reason": reason,
                              "ess": hs.range.ess(), "range": len(hs.range.hands),
                              "options": [(o.action.kind, o.action.amount, round(o.ev, 3),
                                           round(o.info.get("p_fold", 0.0), 3), round(o.info.get("p_raise", 0.0), 3),
                                           round(o.info.get("eq_call", 0.0), 3), round(o.info.get("eq_reraise", 0.0), 3))
                                          for o in options]}
        return opt.action

    def _learn(self, history, hero_is_button, sb, bb, opp_hole, board, hero_hole) -> None:
        if hero_is_button is None:
            hero_is_button = infer_button(history)
            if hero_is_button is None:
                return
        points, _ = walk(history, hero_is_button, sb, bb)
        line = opponent_line(points)
        shown = shown_pre = None
        if opp_hole is not None and len(opp_hole) == 5 and board is not None and len(board) == 5:
            shown = self._showdown_strength(tuple(opp_hole), tuple(board))
            shown_pre = self.scorer.percentile(self.scorer.equity(tuple(opp_hole)))
        self.model.record_hand(points, line, shown, shown_pre)

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
