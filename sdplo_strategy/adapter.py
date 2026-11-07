"""The only module that knows about the core (sdplo-bot).

    from sdplo_strategy.adapter import make_core_bot
    bot = make_core_bot(seed=1)            # an sdplo.bots.Bot
    sdplo.match.run_match(bot, sdplo.bots.make_bot("equity", 2), hands=2000, seed=1)

Core facts this relies on (sdplo-bot @ bf1032f):
  * cards are ints 0..35 = rank*4 + suit, ranks 6..A, suits c d h s -- the same
    encoding as sdplo_strategy.cards, so no translation is needed;
  * Observation: seat (0 = button/SB), hole_cards, board, street (IntEnum), pot
    (includes this street's bets), stacks/bets as (ours, opponent's), to_call
    (may exceed our stack), legal (LegalActions), history of
    ActionRecord(street, seat, kind, added, to) without blinds, blinds;
  * Action.fold() / check() / call() / raise_to(int), raise-to = street total;
  * HandSummary: hand_number, seat, hole_cards, opponent_hole_cards (shown
    only), board, history, net, showdown;
  * evaluator: get_evaluator(ranking).omaha(hole, board) and
    .omaha_batch(holes, boards); equity: equity_vs_hands(...).
If the official interface differs, change this file only.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from .bot import StrategyBot, fallback_action
from .preflop import band_table_for_ranking, table_for_ranking
from .settings import Settings
from .types import CALL, CHECK, FOLD, HERO, OPP, RAISE, Action, ActionRecord, HandResult, Observation

CORE_PATH = os.environ.get("SDPLO_CORE_PATH", os.path.expanduser("~/sdplo-bot"))
_KIND_NAMES = {0: FOLD, 1: CHECK, 2: CALL, 3: RAISE}


def import_core(path: Optional[str] = None):
    """Import the core (never writing bytecode into its folder)."""
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        try:
            import sdplo  # noqa: F401
        except ImportError:
            p = path or CORE_PATH
            if p not in sys.path:
                sys.path.insert(0, p)
            import sdplo  # noqa: F401
        import sdplo.bots, sdplo.engine, sdplo.equity, sdplo.evaluator, sdplo.match  # noqa: E401,F401
    finally:
        sys.dont_write_bytecode = old
    return sys.modules["sdplo"]


# ------------------------------------------------------------------ conversions


def to_history(core_history, hero_seat: int) -> tuple:
    out = []
    for r in core_history:
        kind = _KIND_NAMES[int(r.kind)]
        out.append(ActionRecord(actor=HERO if r.seat == hero_seat else OPP, street=int(r.street),
                                kind=kind, amount=float(r.to) if kind == RAISE else 0.0))
    return tuple(out)


def to_observation(o) -> Observation:
    L = o.legal
    return Observation(
        hole=tuple(int(c) for c in o.hole_cards),
        board=tuple(int(c) for c in o.board),
        pot=o.pot,
        my_stack=o.stacks[0], opp_stack=o.stacks[1],
        to_call=o.to_call,
        min_raise_to=L.min_raise_to if L.can_raise else None,
        max_raise_to=L.max_raise_to if L.can_raise else None,
        street=int(o.street),
        history=to_history(o.history, o.seat),
        hand_number=o.hand_number,
        small_blind=o.small_blind, big_blind=o.big_blind,
        hero_is_button=(o.seat == 0),
        my_street_bet=o.bets[0], opp_street_bet=o.bets[1],
        # discrete action space (official rules): the only legal raise-to totals
        raise_options=(tuple(L.raise_sizes) if L.can_raise and getattr(L, "raise_sizes", None)
                       else None),
    )


def to_hand_result(s) -> HandResult:
    return HandResult(
        hand_number=s.hand_number,
        history=to_history(s.history, s.seat),
        board=tuple(int(c) for c in s.board),
        hero_hole=tuple(int(c) for c in s.hole_cards),
        opp_hole=None if s.opponent_hole_cards is None else tuple(int(c) for c in s.opponent_hole_cards),
        hero_delta=s.net,
        hero_is_button=(s.seat == 0),
    )


def to_core_action(a: Action, core_Action):
    if a.kind == FOLD:
        return core_Action.fold()
    if a.kind == CHECK:
        return core_Action.check()
    if a.kind == CALL:
        return core_Action.call()
    return core_Action.raise_to(int(a.amount))


def core_functions(ranking=None, core=None) -> dict:
    """The core's evaluator / equity in the shapes StrategyBot takes."""
    core = core or import_core()
    ev = core.evaluator.get_evaluator(ranking) if ranking is not None else core.evaluator.get_evaluator()
    equity_vs_hands = core.equity.equity_vs_hands

    def equity_fn(hole, board, opp_hands, weights, n_samples, rng):
        return equity_vs_hands(hole, board, opp_hands, samples=int(n_samples), weights=weights,
                               rng=rng, evaluator=ev).equity

    return {"evaluate": ev.omaha, "evaluate_batch": ev.omaha_batch, "equity_fn": equity_fn}


# ------------------------------------------------------------------ the bot


def make_core_bot(seed: int = 0, settings: Optional[Settings] = None, name: str = "strategy",
                  preflop_table=None, backend: str = "outcomes", core_path: Optional[str] = None):
    core = import_core(core_path)
    Bot = core.bots.Bot
    CoreAction = core.engine.Action
    decision_rng = core.bots.decision_rng

    class StrategyCoreBot(Bot):
        def __init__(self):
            self.name = name
            self.seed = seed
            self.inner: Optional[StrategyBot] = None
            self.config = None

        def _build(self, ranking=None, num_hands=None):
            s = Settings(**vars(settings)) if settings is not None else Settings()
            if num_hands:
                s.hands_per_match = int(num_hands)
            fns = core_functions(ranking, core)
            table = preflop_table
            if isinstance(table, str) and table == "auto":
                table = table_for_ranking(ranking)    # None -> heuristic fallback
            bands = band_table_for_ranking(ranking) if (table is not None and s.use_band_table) else None
            self.inner = StrategyBot(fns["evaluate"], equity_fn=fns["equity_fn"],
                                     evaluate_batch=fns["evaluate_batch"], settings=s,
                                     preflop_table=table, backend=backend, band_table=bands)

        def on_match_start(self, opponent_name, num_hands, config):
            self.config = config
            self._build(getattr(config, "ranking", None), num_hands)   # fresh opponent model

        def act(self, obs):
            try:
                if self.inner is None:
                    self._build()
                o = to_observation(obs)
                a = self.inner.act(o, rng=decision_rng(self.seed, obs))
                out = to_core_action(a, CoreAction)
                if obs.legal.is_legal(out):
                    return out
                self.inner.errors += 1
            except Exception:
                if self.inner is not None:
                    self.inner.errors += 1
            return self._core_fallback(obs)

        def _core_fallback(self, obs):
            L = obs.legal
            if L.can_check:
                return CoreAction.check()
            try:
                cheap = fallback_action(to_observation(obs), Settings()).kind == CALL
            except Exception:
                cheap = False
            return CoreAction.call() if (cheap and L.can_call) else CoreAction.fold()

        def on_hand_end(self, summary):
            try:
                if self.inner is not None:
                    self.inner.on_hand_result(to_hand_result(summary))
            except Exception:
                if self.inner is not None:
                    self.inner.errors += 1

    return StrategyCoreBot()
