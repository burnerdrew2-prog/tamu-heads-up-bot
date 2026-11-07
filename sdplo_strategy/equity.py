"""Equity against a weighted opponent range.

Two backends, chosen by what the core provides:

* outcomes backend (preferred): ``outcomes(hole, board, hands, weights, n, rng)
  -> (idx, outcome)`` draws n opponent hands from the weighted range, deals one
  runout for each and returns which range row was drawn and the result
  (1 win, 0.5 tie, 0 loss).  One pass then answers equity against ANY
  reweighting of the range (the calling range for each raise size, the
  re-raising range) by self-normalised importance weighting.  ``make_outcomes_fn``
  builds this from any plain evaluator, so the core's fast evaluator plugs in
  directly.
* scalar backend: the injected ``equity(hole, board, hands, weights, n, rng) ->
  float``.  Each reweighted range costs another call (budget is split).
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import numpy as np

from .cards import remaining_deck

EvalFn = Callable[[Sequence[int], Sequence[int]], int]
EquityFn = Callable[..., float]


def make_outcomes_fn(evaluate: EvalFn, evaluate_batch: Optional[Callable] = None):
    """Build an outcomes function from an Omaha evaluator ``evaluate(hole, board5)``.

    ``evaluate_batch(holes (n,5), boards (n,5)) -> (n,)`` is used instead of the
    Python loop when given.
    """

    def outcomes(hole, board, hands, weights, n, rng):
        hands = np.asarray(hands, dtype=np.int16)
        w = np.asarray(weights, dtype=float)
        total = float(w.sum())
        if n <= 0 or total <= 0 or len(hands) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        idx = rng.choice(len(hands), size=n, p=w / total)
        opp = hands[idx]
        board = tuple(int(c) for c in board)
        hero = tuple(int(c) for c in hole)
        need = 5 - len(board)
        if need > 0:
            avail = remaining_deck(hero + board)
            keys = rng.random((n, len(avail)))
            conflict = (avail[None, :, None] == opp[:, None, :]).any(axis=2)
            keys[conflict] = 2.0
            pick = np.argpartition(keys, need - 1, axis=1)[:, :need]
            runouts = avail[pick]
        else:
            runouts = np.zeros((n, 0), dtype=np.int16)

        if evaluate_batch is not None:
            boards = np.concatenate([np.tile(np.array(board, dtype=np.int16), (n, 1)), runouts], axis=1)
            hv = np.asarray(evaluate_batch(np.tile(np.array(hero, dtype=np.int16), (n, 1)), boards))
            ov = np.asarray(evaluate_batch(opp, boards))
        else:
            hv = np.empty(n, dtype=np.int64)
            ov = np.empty(n, dtype=np.int64)
            hero_cache: dict = {}
            opp_cache: dict = {}
            for i in range(n):
                ro = tuple(int(c) for c in runouts[i])
                full = board + ro
                h = hero_cache.get(ro)
                if h is None:
                    h = evaluate(hero, full)
                    hero_cache[ro] = h
                key = (int(idx[i]), ro) if need <= 1 else None
                o = opp_cache.get(key) if key is not None else None
                if o is None:
                    o = evaluate(tuple(int(c) for c in opp[i]), full)
                    if key is not None:
                        opp_cache[key] = o
                hv[i] = h
                ov[i] = o
        out = np.where(hv > ov, 1.0, np.where(hv == ov, 0.5, 0.0))
        return idx, out

    return outcomes


class EquityEngine:
    """Equity vs a weighted range, plus vs reweighted sub-ranges, within a budget."""

    supports_exact = True   # DecisionEngine may bypass sampling with exact per-hand outcomes

    def __init__(self, outcomes_fn=None, equity_fn: Optional[EquityFn] = None,
                 chunk: int = 128, min_samples: int = 32):
        if outcomes_fn is None and equity_fn is None:
            raise ValueError("need an outcomes function or an equity function")
        self.outcomes_fn = outcomes_fn
        self.equity_fn = equity_fn
        self.chunk = chunk
        self.min_samples = min_samples

    def estimate(self, hole, board, hands, weights, reweights: Sequence[np.ndarray],
                 n: int, rng: np.random.Generator, deadline: Optional[float] = None,
                 prior_factors: Optional[Sequence[float]] = None, prior_strength: float = 8.0):
        """Return (eq_base, [eq for each reweight], samples_used).

        ``reweights[k]`` is a per-hand multiplier q_k (e.g. P(call | hand)); the
        k-th answer is equity against weights * q_k.  Thin sub-range estimates
        are shrunk toward ``prior_factors[k] * eq_base`` (default eq_base).
        """
        n = max(int(n), self.min_samples)
        if self.outcomes_fn is not None:
            return self._estimate_outcomes(hole, board, hands, weights, reweights, n, rng,
                                           deadline, prior_factors, prior_strength)
        return self._estimate_scalar(hole, board, hands, weights, reweights, n, rng, deadline)

    def _estimate_outcomes(self, hole, board, hands, weights, reweights, n, rng, deadline,
                           prior_factors, prior_strength):
        idx_parts, out_parts, done = [], [], 0
        while done < n:
            m = min(self.chunk, n - done)
            idx, out = self.outcomes_fn(hole, board, hands, weights, m, rng)
            if len(out) == 0:
                break
            idx_parts.append(idx)
            out_parts.append(out)
            done += m
            if deadline is not None and done >= self.min_samples and time.perf_counter() > deadline:
                break
        if not out_parts:
            return 0.5, [0.5] * len(reweights), 0
        idx = np.concatenate(idx_parts)
        out = np.concatenate(out_parts)
        base = float(out.mean())
        res = []
        for k, q in enumerate(reweights):
            qi = np.asarray(q, dtype=float)[idx]
            s = qi.sum()
            prior = base if prior_factors is None else base * float(prior_factors[k])
            if s <= 1e-12:
                res.append(prior)
                continue
            est = float((qi * out).sum() / s)
            n_eff = s * s / max(float((qi * qi).sum()), 1e-12)
            res.append((n_eff * est + prior_strength * prior) / (n_eff + prior_strength))
        return base, res, len(out)

    def _estimate_scalar(self, hole, board, hands, weights, reweights, n, rng, deadline):
        w = np.asarray(weights, dtype=float)
        base = float(self.equity_fn(hole, board, hands, w, n, rng))
        used = n
        res = []
        sub_n = max(self.min_samples, n // 2)
        for q in reweights:
            if deadline is not None and time.perf_counter() > deadline:
                res.append(base)
                continue
            wq = w * np.asarray(q, dtype=float)
            if wq.sum() <= 1e-12:
                res.append(base)
                continue
            res.append(float(self.equity_fn(hole, board, hands, wq, sub_n, rng)))
            used += sub_n
        return base, res, used
