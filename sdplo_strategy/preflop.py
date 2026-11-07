"""Preflop hand strength.

* Suit isomorphism: two 5-card hands are equivalent when one is a suit
  relabelling of the other.  The canonical key is the sorted list of per-suit
  rank tuples; this cuts C(36,5) = 376,992 hands to a few thousand classes.
* ``PreflopTable``: all-in equity vs a random hand for every class, built by
  ``scripts/build_preflop_table.py`` and saved as a small .npz.
* ``heuristic_equity``: a vectorised linear score on hand features (pairs,
  suits, nut-flush potential, straight coverage, high cards), fitted to a
  sample of table equities, used for any hand the table does not cover and
  whenever no table is loaded.
* ``PreflopScorer`` gives both an equity estimate and a percentile among all
  starting hands.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

from .cards import ACE, DECK_ARRAY, random_hands

# --------------------------------------------------------------- isomorphism


def canonical_key(hand) -> tuple:
    by_suit = ([], [], [], [])
    for c in hand:
        c = int(c)
        by_suit[c & 3].append(c >> 2)
    groups = [tuple(sorted(g, reverse=True)) for g in by_suit if g]
    groups.sort(key=lambda g: (len(g), g), reverse=True)
    return tuple(groups)


def hand_from_key(key: tuple) -> tuple:
    cards = []
    for suit, ranks in enumerate(key):
        cards.extend(r * 4 + suit for r in ranks)
    return tuple(sorted(cards))


def canonical_hand(hand) -> tuple:
    return hand_from_key(canonical_key(hand))


def enumerate_classes():
    """All suit-isomorphism classes: (hands (M,5) int16 canonical, counts (M,) int64)."""
    counts: dict = {}
    for combo in combinations(range(36), 5):
        k = canonical_key(combo)
        counts[k] = counts.get(k, 0) + 1
    keys = sorted(counts, key=lambda k: hand_from_key(k))
    hands = np.array([hand_from_key(k) for k in keys], dtype=np.int16)
    mult = np.array([counts[k] for k in keys], dtype=np.int64)
    return hands, mult


# --------------------------------------------------------------- heuristic

_WINDOWS = np.zeros((6, 9), dtype=bool)
_WINDOWS[0, [ACE, 0, 1, 2, 3]] = True          # A-6-7-8-9
for _i in range(5):
    _WINDOWS[_i + 1, _i:_i + 5] = True           # 6-T ... T-A

FEATURE_NAMES = ["bias", "mean_rank", "max_rank", "n_pairs", "top_pair_rank", "trips_in_hand",
                 "n_suited_groups", "three_plus_suited", "ace_suited", "suited_high",
                 "max_window", "windows_ge2", "n_distinct"]


def hand_features(hands: np.ndarray) -> np.ndarray:
    hands = np.asarray(hands, dtype=np.int64).reshape(-1, 5)
    n = len(hands)
    ranks = hands >> 2
    suits = hands & 3
    rc = np.zeros((n, 9), dtype=np.int64)
    sc = np.zeros((n, 4), dtype=np.int64)
    rows = np.arange(n)[:, None]
    np.add.at(rc, (np.repeat(rows, 5, 1), ranks), 1)
    np.add.at(sc, (np.repeat(rows, 5, 1), suits), 1)
    paired = rc >= 2
    n_pairs = paired.sum(1)
    top_pair = np.where(paired.any(1), 8 - np.argmax(paired[:, ::-1], axis=1), -1)
    trips = (rc >= 3).any(1)
    suited = sc >= 2
    n_suited = suited.sum(1)
    three_plus = sc.max(1) >= 3
    # ace with a suited companion
    ace_mask = ranks == ACE
    ace_suit_count = np.where(ace_mask, np.take_along_axis(sc, suits, 1), 0)
    ace_suited = (ace_suit_count >= 2).any(1)
    # highest rank inside any suited group
    suited_card = np.take_along_axis(sc, suits, 1) >= 2
    suited_high = np.where(suited_card, ranks, -1).max(1)
    present = rc > 0
    win = (present[:, None, :] & _WINDOWS[None]).sum(2)
    feats = np.stack([
        np.ones(n),
        ranks.mean(1) / 8.0,
        ranks.max(1) / 8.0,
        n_pairs / 2.0,
        np.where(top_pair >= 0, (top_pair + 1) / 9.0, 0.0),
        trips.astype(float),
        n_suited / 2.0,
        three_plus.astype(float),
        ace_suited.astype(float),
        np.where(suited_high >= 0, (suited_high + 1) / 9.0, 0.0),
        win.max(1) / 5.0,
        (win >= 2).sum(1) / 6.0,
        present.sum(1) / 5.0,
    ], axis=1)
    return feats


# Least-squares fit to the official-ranking table (trips > straight, flush > full
# house; scripts/build_preflop_table.py --trips-beat-straight), count-weighted
# RMSE 0.015 equity.  Used only when no table file can be loaded.
HEURISTIC_COEF = np.array([
    -0.5603, 0.3115, -0.0509, 0.4166, 0.0304, 0.0418, 0.0570, -0.0208, -0.0027, 0.0132,
    0.0308, 0.0552, 0.7954,
])


def heuristic_equity(hands: np.ndarray, coef: Optional[np.ndarray] = None) -> np.ndarray:
    c = HEURISTIC_COEF if coef is None else coef
    return np.clip(hand_features(hands) @ c, 0.05, 0.95)


# --------------------------------------------------------------- table

_POPCOUNT = np.array([bin(i).count("1") for i in range(512)], dtype=np.int64)


def canonical_codes(hands) -> np.ndarray:
    """Vectorised class id (int64) per hand, equal exactly when ``canonical_key`` is.

    Per suit, a 9-bit rank mask; the four masks are sorted by (cards in suit,
    mask), which for equal sizes is the same order as the descending rank
    tuples ``canonical_key`` sorts by, and packed 9 bits each.
    """
    hands = np.asarray(hands, dtype=np.int64).reshape(-1, 5)
    bits = np.left_shift(1, hands >> 2)
    suits = hands & 3
    masks = np.stack([np.where(suits == k, bits, 0).sum(axis=1) for k in range(4)], axis=1)
    key = np.sort(_POPCOUNT[masks] * 512 + masks, axis=1)[:, ::-1] & 511
    return (key[:, 0] << 27) | (key[:, 1] << 18) | (key[:, 2] << 9) | key[:, 3]


class PreflopTable:
    def __init__(self, hands: np.ndarray, equity: np.ndarray, counts: np.ndarray,
                 meta: Optional[dict] = None):
        self.hands = np.asarray(hands, dtype=np.int16)
        self.equity = np.asarray(equity, dtype=float)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.meta = dict(meta or {})
        codes = canonical_codes(self.hands)
        order = np.argsort(codes, kind="stable")
        self._codes = codes[order]
        self._eq_by_code = self.equity[order]

    @classmethod
    def load(cls, path) -> "PreflopTable":
        with np.load(Path(path), allow_pickle=False) as d:
            eq = d["equity"].astype(float)
            ok = np.isfinite(eq)
            meta = {k: d[k].item() for k in d.files
                    if k not in ("hands", "equity", "counts", "se") and d[k].ndim == 0}
            if "coef" in d.files:
                meta["coef"] = d["coef"].astype(float)
            return cls(d["hands"][ok], eq[ok], d["counts"][ok], meta)

    def lookup_many(self, hands) -> np.ndarray:
        """Equity per hand; NaN for a hand whose class is not in the table."""
        codes = canonical_codes(hands)
        if len(self._codes) == 0:
            return np.full(len(codes), np.nan)
        i = np.minimum(np.searchsorted(self._codes, codes), len(self._codes) - 1)
        return np.where(self._codes[i] == codes, self._eq_by_code[i], np.nan)

    def lookup(self, hand) -> Optional[float]:
        v = float(self.lookup_many(np.array([tuple(int(c) for c in hand)]))[0])
        return None if not np.isfinite(v) else v

    @property
    def coverage(self) -> float:
        return float(self.counts.sum()) / 376_992.0


# The table ships inside the package so a single-folder upload carries it.
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_TABLE_PATH = DATA_DIR / "preflop_table.npz"
_TABLE_CACHE: dict = {}


def table_filename(trips_beat_straight: bool = False, flush_beats_full_house: bool = True) -> str:
    """File name of the table for one HandRanking (the core's default ranking -> the default file)."""
    if not trips_beat_straight and flush_beats_full_house:
        return DEFAULT_TABLE_PATH.name
    return f"preflop_table_tbs{int(trips_beat_straight)}_fbfh{int(flush_beats_full_house)}.npz"


def table_for_ranking(ranking=None) -> Optional[PreflopTable]:
    """Table built for ``ranking`` (any object with the core HandRanking's two flags);
    else the default-ranking table (the preflop equity ORDER barely depends on these
    flags); else None, and the scorer uses the heuristic."""
    if ranking is not None:
        name = table_filename(bool(getattr(ranking, "trips_beat_straight", False)),
                              bool(getattr(ranking, "flush_beats_full_house", True)))
        t = load_default_table(DATA_DIR / name)
        if t is not None:
            return t
    return load_default_table()


def load_default_table(path=None) -> Optional[PreflopTable]:
    """The bundled preflop table, or None if it is missing or unreadable (the
    scorer then falls back to ``heuristic_equity``).  Cached per path."""
    p = Path(path) if path is not None else DEFAULT_TABLE_PATH
    key = str(p)
    if key not in _TABLE_CACHE:
        try:
            _TABLE_CACHE[key] = PreflopTable.load(p) if p.is_file() else None
        except Exception:
            _TABLE_CACHE[key] = None
    return _TABLE_CACHE[key]


class PreflopScorer:
    """Equity-vs-random estimate and percentile for any 5-card hand."""

    def __init__(self, table: Optional[PreflopTable] = None, coef: Optional[np.ndarray] = None,
                 ref_size: int = 20000, seed: int = 7):
        self.table = table
        if coef is None and table is not None and "coef" in table.meta:
            coef = np.asarray(table.meta["coef"], dtype=float)
        self.coef = coef
        if table is not None and table.coverage > 0.99:
            order = np.argsort(table.equity)
            self._sorted = table.equity[order]
            self._cum = np.cumsum(table.counts[order]).astype(float)
        else:
            ref = random_hands(np.random.default_rng(seed), DECK_ARRAY, ref_size)
            self._sorted = np.sort(self.equities(ref))
            self._cum = np.arange(1, ref_size + 1, dtype=float)
        self._cache: dict = {}

    def equity(self, hand) -> float:
        key = tuple(sorted(int(c) for c in hand))
        v = self._cache.get(key)
        if v is None:
            v = float(self.equities(np.array([key]))[0])
            self._cache[key] = v
        return v

    def equities(self, hands: np.ndarray) -> np.ndarray:
        hands = np.asarray(hands).reshape(-1, 5)
        if self.table is None:
            return heuristic_equity(hands, self.coef)
        out = self.table.lookup_many(hands)
        miss = ~np.isfinite(out)
        if miss.any():
            out[miss] = heuristic_equity(hands[miss], self.coef)
        return out

    def percentiles(self, equities) -> np.ndarray:
        """Vectorised ``percentile``."""
        e = np.asarray(equities, dtype=float)
        total = self._cum[-1]
        lo = np.searchsorted(self._sorted, e, side="left")
        hi = np.searchsorted(self._sorted, e, side="right")
        below = np.where(lo > 0, self._cum[np.maximum(lo - 1, 0)], 0.0)
        upto = np.where(hi > 0, self._cum[np.maximum(hi - 1, 0)], 0.0)
        return (below + upto) / (2.0 * total)

    def percentile(self, equity: float) -> float:
        total = self._cum[-1]
        lo = np.searchsorted(self._sorted, equity, side="left")
        hi = np.searchsorted(self._sorted, equity, side="right")
        below = self._cum[lo - 1] if lo > 0 else 0.0
        upto = self._cum[hi - 1] if hi > 0 else 0.0
        return float((below + upto) / (2.0 * total))


def band_of_pct(pct, n_bands: int = 10) -> np.ndarray:
    """Decile (or n-tile) index of preflop percentiles."""
    return np.minimum((np.asarray(pct, dtype=float) * n_bands).astype(np.int64), n_bands - 1)


class BandTable:
    """Preflop equity of every class vs each n-tile of starting hands
    (scripts/build_band_table.py).  eq vs a weighted range = sum_b W_b eq[b]."""

    def __init__(self, codes: np.ndarray, equity: np.ndarray):
        order = np.argsort(codes, kind="stable")
        self._codes = np.asarray(codes, dtype=np.int64)[order]
        self.equity = np.asarray(equity, dtype=float)[order]
        self.n_bands = self.equity.shape[1]

    @classmethod
    def load(cls, path) -> "BandTable":
        with np.load(Path(path), allow_pickle=False) as d:
            return cls(d["codes"], d["equity"].astype(float))

    def row(self, hand) -> Optional[np.ndarray]:
        code = canonical_codes(np.array([tuple(int(c) for c in hand)]))[0]
        i = int(np.searchsorted(self._codes, code))
        if i >= len(self._codes) or self._codes[i] != code:
            return None
        return self.equity[i]


def band_table_for_ranking(ranking=None) -> Optional[BandTable]:
    tbs = bool(getattr(ranking, "trips_beat_straight", True)) if ranking is not None else True
    fb = bool(getattr(ranking, "flush_beats_full_house", True)) if ranking is not None else True
    p = DATA_DIR / table_filename(tbs, fb).replace("preflop_table", "preflop_bands")
    key = "bands:" + str(p)
    if key not in _TABLE_CACHE:
        try:
            _TABLE_CACHE[key] = BandTable.load(p) if p.is_file() else None
        except Exception:
            _TABLE_CACHE[key] = None
    return _TABLE_CACHE[key]


# Load once at import; None (heuristic fallback) when the file is absent.
DEFAULT_TABLE = load_default_table()
