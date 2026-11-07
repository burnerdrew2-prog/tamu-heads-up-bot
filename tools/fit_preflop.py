"""Fit a linear preflop-strength formula for Short Deck PLO5 heads-up.

Target: equity against one random hand (Monte Carlo). Features: pairs by
rank, suitedness, connectedness, high cards. Prints the weights to paste
into agent.py. Explicit seed; a minute of one core.
"""
import random
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, ".")
from evaluator import omaha_score, warmup  # noqa: E402


def features(hole):
    ranks = sorted((c >> 2 for c in hole), reverse=True)
    suits = [c & 3 for c in hole]
    cnt = {r: ranks.count(r) for r in set(ranks)}
    f = []
    f.append(sum(1 for r, k in cnt.items() if k == 2 and r >= 6))          # QQ+ pairs
    f.append(sum(1 for r, k in cnt.items() if k == 2 and r < 6))           # lower pairs
    f.append(sum(1 for k in cnt.values() if k >= 3))                       # trips (dead)
    sc = [suits.count(s) for s in range(4)]
    f.append(sum(1 for k in sc if k >= 2))                                 # suited groups
    f.append(sum(1 for s in range(4) if sc[s] >= 2 and any(c >> 2 == 8 and c & 3 == s for c in hole)))  # ace-suited
    f.append(sum(1 for k in sc if k >= 4))                                 # 4+ of a suit (wasted)
    distinct = sorted(set(ranks))
    best = run = 1
    for a, b in zip(distinct, distinct[1:]):
        run = run + 1 if b - a <= 2 else 1
        best = max(best, run)
    f.append(best)                                                         # connected run (gaps <= 1)
    f.append(sum(ranks) / 8.0)                                             # high-card mass
    f.append(1.0)
    return f


def equity(hole, rng, n):
    deck = [c for c in range(36) if c not in hole]
    w = 0.0
    for _ in range(n):
        d = rng.sample(deck, 10)
        a, b = omaha_score(hole, d[5:]), omaha_score(d[:5], d[5:])
        w += 1.0 if a > b else (0.5 if a == b else 0.0)
    return w / n


warmup()
rng = random.Random(3)
X, y = [], []
for i in range(1500):
    hole = rng.sample(range(36), 5)
    X.append(features(hole))
    y.append(equity(hole, rng, 160))
X, y = np.array(X), np.array(y)
w, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ w
print("weights", [round(float(v), 4) for v in w])
print("R2 %.3f  resid sd %.3f  equity range %.2f-%.2f" % (1 - ((y - pred) ** 2).mean() / y.var(), (y - pred).std(), y.min(), y.max()))
print("equity percentiles 50/70/85/95:", [round(float(np.percentile(y, q)), 3) for q in (50, 70, 85, 95)])
