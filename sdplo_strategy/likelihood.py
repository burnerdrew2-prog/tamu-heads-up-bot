"""P(opponent action | hand): a soft strength-threshold model.

Given the opponent's frequencies in a spot (fold f, passive p, aggressive a),
and where each hand sits in their current range (strength percentile 0..1):

  * value aggression comes from the top a*(1-b) of the range,
  * folds and bluffs share the bottom f + a*b (bluff share b),
  * everything in between plays passively (check / call).

Edges are smooth (logistic), each piece is rescaled so its weighted mean over
the current range equals the frequency the opponent model predicts, and the
result is mixed with the flat base rate (``floor``) so no hand is ever ruled
out by one action.  A low ``informativeness`` (opponent's aggression does not
track strength at showdown) raises that mix toward flat.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .opponent import AGGR_I, FOLD_I, PASSIVE_I


# Strength values that differ by less than this are treated as tied.  Replayed
# strengths (refill, replay_weights) are recomputed by a different arithmetic
# path than the live ones, so they can differ in the last bit; an exact
# searchsorted would then move a hand from the middle of a tie group to its edge.
TIE_TOL = 1e-9


class StrengthCDF:
    """Frozen weighted mid-rank CDF of strengths, to place hands into a range.

    The live percentile (``weighted_percentile``) and every later lookup of a
    replayed hand go through this one class, so both see identical numbers.
    """

    def __init__(self, values: np.ndarray, weights: np.ndarray):
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="stable")
        self.sv = values[order]
        sw = np.asarray(weights, dtype=float)[order]
        self.cum = np.concatenate([[0.0], np.cumsum(sw)])
        self.total = float(self.cum[-1])

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.total <= 0 or len(self.sv) == 0:
            return np.full(len(values), 0.5)
        lo = np.searchsorted(self.sv, values - TIE_TOL, side="left")
        hi = np.searchsorted(self.sv, values + TIE_TOL, side="right")
        return (self.cum[lo] + self.cum[hi]) / (2.0 * self.total)


def weighted_percentile(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mid-rank weighted percentile of each value within the weighted set."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(0)
    return StrengthCDF(values, weights)(values)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _calibrate(shape: np.ndarray, target: float, weights: Optional[np.ndarray]) -> np.ndarray:
    if target <= 0:
        return np.zeros_like(shape)
    m = float(np.average(shape, weights=weights)) if weights is not None else float(shape.mean())
    if m <= 1e-12:
        return np.full_like(shape, target)
    out = shape * (target / m)
    # clipping at 1 loses mass; push the excess down once
    over = out > 1.0
    if over.any():
        out = np.minimum(out, 1.0)
        m2 = float(np.average(out, weights=weights)) if weights is not None else float(out.mean())
        if 1e-12 < m2 < target:
            under = ~over
            if under.any():
                out[under] = np.minimum(1.0, out[under] * (1 + (target - m2) / max(m2, 1e-12)))
    return out


def action_probs(pct: np.ndarray, freqs, bluff_share: float = 0.2, width: float = 0.06,
                 floor: float = 0.06, informativeness: float = 1.0,
                 weights: Optional[np.ndarray] = None,
                 call_informativeness: Optional[float] = None) -> np.ndarray:
    """Return a (3, n) array: P(fold), P(passive), P(aggressive) for each hand."""
    pct = np.asarray(pct, dtype=float)
    f, p, a = (float(x) for x in freqs)
    s = f + p + a
    f, p, a = f / s, p / s, a / s
    b = float(np.clip(bluff_share, 0.0, 0.9))
    value_top = a * (1.0 - b)
    bottom = f + a * b
    w = max(width, 1e-3)
    out = np.zeros((3, len(pct)))
    if value_top > 0:
        up = _sigmoid((pct - (1.0 - value_top)) / w)
        out[AGGR_I] += _calibrate(up, value_top, weights)
    if bottom > 0:
        down = _sigmoid((bottom - pct) / w)
        down = _calibrate(down, bottom, weights)
        out[FOLD_I] = down * (f / bottom)
        out[AGGR_I] += down * (a * b / bottom)
    tot = out[FOLD_I] + out[AGGR_I]
    over = tot > 1.0
    if over.any():
        out[FOLD_I, over] /= tot[over]
        out[AGGR_I, over] /= tot[over]
    out[PASSIVE_I] = np.clip(1.0 - out[FOLD_I] - out[AGGR_I], 0.0, 1.0)
    kappa = float(np.clip(informativeness, 0.0, 1.0))
    eps = floor + (1.0 - floor) * (1.0 - kappa) * 0.9
    if call_informativeness is None:
        base = np.array([f, p, a])[:, None]
        return (1.0 - eps) * out + eps * base
    # aggression and folding can track strength to different degrees (a chart bot
    # semi-bluffs, so its bets say less than its folds): flatten each separately,
    # keep each row's mean, passive takes the rest
    kc = float(np.clip(call_informativeness, 0.0, 1.0))
    eps_c = floor + (1.0 - floor) * (1.0 - kc) * 0.9
    res = np.empty_like(out)
    res[AGGR_I] = (1.0 - eps) * out[AGGR_I] + eps * a
    res[FOLD_I] = (1.0 - eps_c) * out[FOLD_I] + eps_c * f
    tot = res[FOLD_I] + res[AGGR_I]
    over = tot > 1.0
    if over.any():
        res[FOLD_I, over] /= tot[over]
        res[AGGR_I, over] /= tot[over]
    res[PASSIVE_I] = np.clip(1.0 - res[FOLD_I] - res[AGGR_I], 0.0, 1.0)
    return res


def likelihood(pct: np.ndarray, freqs, outcome: int, **kw) -> np.ndarray:
    return action_probs(pct, freqs, **kw)[outcome]
