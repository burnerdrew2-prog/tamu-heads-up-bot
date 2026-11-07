"""Adapter to the OFFICIAL TAMU Datathon interface (tamu-datathon-org/heads-up-bundle).

The starter's agent.py calls three functions; wire them to ours:

    from sdplo_strategy.official import new_half, choose_action, on_hand_end

Official rules this relies on (game.py in the starter repo is the source of truth):
  * discrete actions FOLD (only when to_call > 0), CHECK_CALL (always legal),
    RAISE_HALF and RAISE_POT, whose exact totals are in ``obs.raise_to``;
  * 80-chip stacks reset every hand, blinds 1/2, at most 4 raises per street;
  * ranking: trips beat a straight, flush beats a full house (OFFICIAL_RANKING);
  * 8 s time bank per 1,000-hand half, first 1 ms of each request free, 1 s hard
    cap per request; each half is a fresh process and ``memory`` is wiped;
  * HandResult carries no action history: the opponent's closing action is
    reconstructed from the last observation, our action and the outcome.

Everything is wrapped: choose_action never raises and always returns one of
``obs.legal_actions`` (fallback: check if free, else call if cheap, else fold;
CHECK_CALL if even that fails).  on_hand_end never raises.
"""
from __future__ import annotations

import os
import time
import traceback
from typing import Optional

from .bot import StrategyBot, fallback_action
from .preflop import band_table_for_ranking, table_for_ranking
from .settings import Settings
from .types import CALL, CHECK, FOLD, HERO, OPP, RAISE, Action, ActionRecord, HandResult, Observation

# Rule assumptions from the official guide, kept in one place.
OFFICIAL_RANKING = {"trips_beat_straight": True, "flush_beats_full_house": True}
DEFAULT_BANK = 8.0          # seconds per half
REQUEST_GRACE = 0.001       # first 1 ms of every request is not charged
HARD_REQUEST_LIMIT = 1.0    # seconds; longer is an error


def official_settings(**overrides) -> Settings:
    """Settings sized for an ~8 s bank over ~3,000 decisions (a few ms each)."""
    s = Settings(
        range_size=128, ess_min_frac=0.15, refill_frac=0.5, max_range_size=320,
        strength_runouts=1, sample_budget=80, preflop_budget_frac=0.5,
        time_limit=0.25, time_target_frac=0.6, hard_stop_frac=0.5,
        match_time_budget=DEFAULT_BANK, hands_per_match=1000,
        request_grace=REQUEST_GRACE, request_overhead=0.0007, bank_reserve=2.0,
        cheap_below=0.0009, showdown_ref_hands=60, seed=20261108,
        preflop_score_realization=2.0,   # 3 seeds x 8 bots: +2.67 vs +2.38 bb/deal at 0.25
        fold_scale=0.8,                  # postflop fold-equity discount; 0.65 was +0.04 vs the field but -0.3 vs drew_v2
        postflop_eq_realization=2.5,     # strong hands realise more on flop/turn, weak less (see commit)
    )
    for k, v in overrides.items():
        if not hasattr(s, k):
            raise TypeError(f"unknown setting {k!r}")
        setattr(s, k, v)
    return s


# ------------------------------------------------------------------ evaluator


def load_evaluator(ranking_flags: Optional[dict] = None):
    """(evaluate, evaluate_batch) for the official ranking.

    Prefers the core's fast numpy evaluator (``sdplo.evaluator``: on sys.path,
    vendored next to this package, or at SDPLO_CORE_PATH).  Falls back to the
    starter's ``evaluator.omaha_score`` (scalar, slower) if the core is absent.
    """
    flags = dict(OFFICIAL_RANKING if ranking_flags is None else ranking_flags)
    import importlib
    import sys
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        try:
            E = importlib.import_module("sdplo.evaluator")     # vendored copy or already on the path
        except ImportError:
            from .adapter import CORE_PATH
            if CORE_PATH not in sys.path:
                sys.path.insert(0, CORE_PATH)
            E = importlib.import_module("sdplo.evaluator")
        ev = E.get_evaluator(E.HandRanking(**flags))
        return ev.omaha, ev.omaha_batch, ev.ranking
    except Exception:
        pass
    finally:
        sys.dont_write_bytecode = old
    import evaluator as starter  # the official evaluator.py (ranking fixed by the rules)
    return starter.omaha_score, None, None


# ------------------------------------------------------------------ conversion


def _official():
    import game  # the official game.py, imported lazily so tests can run without it
    return game


def _kind(a, amount: int, to_call_before: Optional[int] = None) -> str:
    name = a.name if hasattr(a, "name") else str(a)
    if name == "FOLD":
        return FOLD
    if name == "CHECK_CALL":
        return CALL if amount > 0 else CHECK
    return RAISE


def to_history(action_history) -> list:
    out = []
    for r in action_history:
        k = _kind(r.action, int(r.amount))
        out.append(ActionRecord(actor=HERO if r.is_me else OPP, street=int(r.street), kind=k,
                                amount=float(r.street_total) if k == RAISE else 0.0))
    return out


def _raise_totals(obs) -> dict:
    """{official raise action: raise-to total} for the legal raises."""
    out = {}
    for a, amt in (obs.raise_to or {}).items():
        if a in obs.legal_actions:
            out[a] = int(amt)
    return out


def to_observation(obs, small_blind: int = 1) -> Observation:
    totals = _raise_totals(obs)
    opts = tuple(sorted(set(totals.values()))) or None
    return Observation(
        hole=tuple(int(c) for c in obs.hole_cards),
        board=tuple(int(c) for c in obs.board),
        pot=int(obs.pot), my_stack=int(obs.my_stack), opp_stack=int(obs.opp_stack),
        to_call=int(obs.to_call),
        min_raise_to=min(opts) if opts else None, max_raise_to=max(opts) if opts else None,
        street=int(obs.street), history=tuple(to_history(obs.action_history)),
        hand_number=int(obs.hand_index), small_blind=small_blind, big_blind=int(obs.big_blind),
        hero_is_button=bool(obs.is_button), my_street_bet=int(obs.my_bet),
        opp_street_bet=int(obs.opp_bet), raise_options=opts,
    )


def to_official_action(a: Action, obs):
    """Our action -> one of obs.legal_actions."""
    A = type(obs.legal_actions[0]) if obs.legal_actions else _official().Action
    legal = set(obs.legal_actions)
    if a.kind == FOLD and A.FOLD in legal:
        return A.FOLD
    if a.kind == RAISE:
        totals = _raise_totals(obs)
        if totals:
            # the raise action whose total is nearest our amount (ties -> the smaller raise)
            return min(totals, key=lambda k: (abs(totals[k] - float(a.amount)), totals[k]))
    return A.CHECK_CALL


# ------------------------------------------------------------------ the agent


class OfficialAgent:
    """One half's state: the strategy bot, the time bank, the current hand."""

    def __init__(self, config=None, settings: Optional[Settings] = None, evaluator=None,
                 ranking_flags: Optional[dict] = None):
        self.config = config
        g = (lambda k, d: getattr(config, k, d) if config is not None else d)
        self.small_blind = int(g("small_blind", 1))
        self.bank = float(g("time_bank_seconds", DEFAULT_BANK))
        s = settings or official_settings()
        s.match_time_budget = max(0.5, self.bank - s.bank_reserve)
        s.hands_per_match = int(g("num_hands", s.hands_per_match))
        if evaluator is None:
            evaluate, batch, ranking = load_evaluator(ranking_flags)
        else:
            evaluate, batch, ranking = evaluator
        table = table_for_ranking(ranking if ranking is not None
                                  else _Flags(**(ranking_flags or OFFICIAL_RANKING)))
        rk = ranking if ranking is not None else _Flags(**(ranking_flags or OFFICIAL_RANKING))
        bands = band_table_for_ranking(rk) if (s.use_band_table and table is not None) else None
        self.s = s
        self.bot = StrategyBot(evaluate, evaluate_batch=batch, settings=s, preflop_table=table,
                               band_table=bands)
        self.errors = 0
        self.last_error: Optional[str] = None
        self._hand = None            # hand_index of the last decision
        self._obs = None             # last official observation this hand
        self._act = None             # our last official action this hand
        self.decision_ms: list = []
        self._pending_end = 0.0      # hook time since the last decision (same request)

    # -------------------------------------------------------------- decide
    def choose(self, obs):
        t0 = time.perf_counter()
        n_before = self.bot.timer.n
        try:
            o = to_observation(obs, self.small_blind)
            a = self.bot.act(o, rng=None)
            out = to_official_action(a, obs)
            if out not in obs.legal_actions:
                raise ValueError(f"illegal {out}")
        except Exception:
            self.errors += 1
            self.last_error = traceback.format_exc(limit=4)
            out = safe_action(obs)
        self._hand, self._obs, self._act = int(obs.hand_index), obs, out
        full = time.perf_counter() - t0 + self._pending_end
        self._pending_end = 0.0
        try:
            t = self.bot.timer
            if t.n > n_before:     # charge conversion + the result hooks of this request too
                t.adjust(t.last_elapsed, full, cheap=self.bot._was_cheap)
        except Exception:
            pass
        self.decision_ms.append(1000 * full)
        return out

    # -------------------------------------------------------------- learn
    def hand_end(self, result) -> None:
        t0 = time.perf_counter()
        try:
            hist, button, hole = self._reconstruct(result)
            self.bot.on_hand_result(HandResult(
                hand_number=int(result.hand_index), history=tuple(hist),
                board=tuple(int(c) for c in result.board), hero_hole=hole,
                opp_hole=None if result.opp_hole is None else tuple(int(c) for c in result.opp_hole),
                hero_delta=result.my_net, hero_is_button=button,
                small_blind=self.small_blind, big_blind=2 if self._obs is None else int(self._obs.big_blind)))
        except Exception:
            self.errors += 1
            self.last_error = traceback.format_exc(limit=4)
        self._obs = self._act = None
        self._pending_end += time.perf_counter() - t0

    def _reconstruct(self, result):
        """Full action history of a finished hand from our last view of it."""
        obs, act = self._obs, self._act
        if obs is None or int(obs.hand_index) != int(result.hand_index):
            # We never acted: we were the big blind and the button folded preflop.
            return [ActionRecord(OPP, 0, FOLD)], False, ()
        hist = to_history(obs.action_history)
        street = int(obs.street)
        to_call = int(obs.to_call)
        name = act.name
        if name == "FOLD":
            hist.append(ActionRecord(HERO, street, FOLD))
            return hist, bool(obs.is_button), tuple(obs.hole_cards)
        if name == "CHECK_CALL":
            hist.append(ActionRecord(HERO, street, CALL if to_call > 0 else CHECK))
        else:
            total = _raise_totals(obs).get(act, int(obs.opp_bet) + int(obs.pot) + to_call)
            hist.append(ActionRecord(HERO, street, RAISE, float(total)))
        if not result.went_to_showdown:
            hist.append(ActionRecord(OPP, street, FOLD))      # only way it ends without us
        else:
            opp_acted = any(r.actor == OPP and r.street == street for r in hist[:-1])
            closed = name == "CHECK_CALL" and opp_acted
            if not closed:
                # opponent closed the street: a call of our raise, or a check / call
                hist.append(ActionRecord(OPP, street, CALL if (name != "CHECK_CALL" or to_call > 0) else CHECK))
        return hist, bool(obs.is_button), tuple(obs.hole_cards)


class _Flags:
    def __init__(self, trips_beat_straight=True, flush_beats_full_house=True):
        self.trips_beat_straight = trips_beat_straight
        self.flush_beats_full_house = flush_beats_full_house


def safe_action(obs):
    """Check if free, else call if cheap, else fold; CHECK_CALL if anything fails."""
    try:
        A = type(obs.legal_actions[0])
        o = Observation(hole=tuple(obs.hole_cards), board=tuple(obs.board), pot=int(obs.pot),
                        my_stack=int(obs.my_stack), opp_stack=int(obs.opp_stack),
                        to_call=int(obs.to_call), min_raise_to=None, max_raise_to=None,
                        street=int(obs.street))
        k = fallback_action(o, Settings()).kind
        if k == FOLD and A.FOLD in obs.legal_actions:
            return A.FOLD
        return A.CHECK_CALL
    except Exception:
        try:
            return type(obs.legal_actions[0]).CHECK_CALL
        except Exception:
            return _official().Action.CHECK_CALL


# ------------------------------------------------------------------ agent.py hooks

_KEY = "sdplo_agent"


def new_half(config, memory: dict, **kw) -> None:
    try:
        memory[_KEY] = OfficialAgent(config, **kw)
    except Exception:
        memory[_KEY] = None
        memory["sdplo_init_error"] = traceback.format_exc(limit=6)


def choose_action(obs, memory: dict):
    try:
        agent = memory.get(_KEY)
        if agent is None:
            if _KEY not in memory:          # new_half was never called: build lazily
                new_half(None, memory)
                agent = memory.get(_KEY)
            if agent is None:
                return safe_action(obs)
        return agent.choose(obs)
    except Exception:
        return safe_action(obs)


def on_hand_end(result, memory: dict) -> None:
    try:
        agent = memory.get(_KEY)
        if agent is not None:
            agent.hand_end(result)
    except Exception:
        pass


def warmup() -> None:
    """Do the slow one-time work (core evaluator tables for the official ranking,
    preflop and band tables, numpy code paths) at import, which the judge does not
    charge, instead of in the first new_half / decision of a half."""
    try:
        evaluate, batch, ranking = load_evaluator()
        rk = ranking if ranking is not None else _Flags(**OFFICIAL_RANKING)
        table_for_ranking(rk)
        band_table_for_ranking(rk)
        mem: dict = {}
        new_half(None, mem)
        agent = mem.get(_KEY)
        if agent is not None:
            import numpy as np
            hole = (0, 5, 10, 15, 20)
            agent.bot.scorer.equities(np.array([hole]))
            batch is not None and batch(np.array([hole]), np.array([[24, 29, 34, 3, 8]]))
    except Exception:
        pass


warmup()
