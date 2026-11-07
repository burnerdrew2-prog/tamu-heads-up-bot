"""Short-deck hand evaluation, including the Omaha rule (exactly 2 hole + 3 board).

Strengths are small positive ints: higher is better, equal is a tie. They are only
comparable within one ``HandRanking`` (the category order is configurable).

How it works
------------
A five-card hand's value depends only on its rank multiset and on whether all five
cards share a suit. So two small tables cover everything:

* ``rank_table[key]``: value of a non-flush hand, ``key = sum(5 ** rank)`` over the
  five cards (base 5 because a rank appears at most 4 times). 1,278 live entries.
* ``flush_table[mask]``: value of a flush (or straight flush), ``mask`` = 9-bit
  rank mask. 126 live entries.

For Omaha, keys are additive, so the 10 hole pairs and 10 board triples combine by
an outer sum and one table gather gives all 100 combinations at once (numpy path).
Flushes only need checking for the single suit with 3+ cards on the board.

``five_card_table()`` is the full lookup over all C(36,5) = 376,992 five-card
hands (indexed by the colex combinatorial index), built from the same scoring
code. It is used for the category-count test and ``evaluate5_indexed``; it builds
in well under a second, so it is cached in memory only (see ``five_card_table``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, combinations_with_replacement
from math import comb
from typing import Sequence

import numpy as np

from .cards import DECK_SIZE, NUM_RANKS

# Category ids are fixed; their *order* comes from HandRanking.
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8
CATEGORY_NAMES = (
    "high card", "one pair", "two pair", "three of a kind", "straight",
    "flush", "full house", "four of a kind", "straight flush",
)

# Straight rank masks, weakest first. Rank index 0 = six, 8 = ace.
# A-6-7-8-9 (the "wheel" in short deck) is the lowest straight, T-J-Q-K-A the highest.
WHEEL_MASK = (1 << 8) | 0b1111
STRAIGHT_MASKS: tuple[int, ...] = (WHEEL_MASK,) + tuple(0b11111 << i for i in range(5))

POW5 = np.array([5 ** r for r in range(NUM_RANKS)], dtype=np.int64)
_POW5_T = tuple(int(x) for x in POW5)
RANK_KEY_SPACE = 5 ** NUM_RANKS  # 1,953,125

PAIRS_5 = tuple(combinations(range(5), 2))  # 10 hole pairs
_TRIPLES = {n: tuple(combinations(range(n), 3)) for n in (3, 4, 5)}


@dataclass(frozen=True)
class HandRanking:
    """Category order. Defaults: the official TAMU Datathon order (game.py /
    evaluator.py in the starter repo, cross-checked in tests/test_official.py).

    Weakest to strongest by default: high card, pair, two pair, straight, trips,
    full house, flush, quads, straight flush. (PokerKit's ShortDeckHoldemLookup uses
    ``trips_beat_straight=False``.)
    """

    trips_beat_straight: bool = True
    flush_beats_full_house: bool = True

    @property
    def order(self) -> tuple[int, ...]:
        """Category ids from weakest to strongest."""
        mid = (STRAIGHT, THREE_OF_A_KIND) if self.trips_beat_straight else (THREE_OF_A_KIND, STRAIGHT)
        top = (FULL_HOUSE, FLUSH) if self.flush_beats_full_house else (FLUSH, FULL_HOUSE)
        return (HIGH_CARD, ONE_PAIR, TWO_PAIR) + mid + top + (FOUR_OF_A_KIND, STRAIGHT_FLUSH)


DEFAULT_RANKING = HandRanking()


def _score(ranks: np.ndarray, flush: np.ndarray, ranking: HandRanking) -> tuple[np.ndarray, np.ndarray]:
    """Score five-card hands. ranks: (N, 5) rank indices; flush: (N,) bool.

    Returns (score int64, category int64). Scores order hands correctly; they are
    sparse, and get compressed to dense strengths by the table builder.
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    n = ranks.shape[0]
    flush = np.asarray(flush, dtype=bool)
    counts = (ranks[:, :, None] == np.arange(NUM_RANKS)).sum(axis=1)  # (N, 9)
    present = counts > 0
    distinct = present.sum(axis=1)
    mask = present.astype(np.int64) @ (1 << np.arange(NUM_RANKS, dtype=np.int64))
    maxc = counts.max(axis=1)

    straight_val = np.full(n, -1, dtype=np.int64)
    for v, m in enumerate(STRAIGHT_MASKS):
        straight_val[mask == m] = v
    is_straight = straight_val >= 0
    five = distinct == 5

    cat = np.full(n, HIGH_CARD, dtype=np.int64)
    cat[(maxc == 2) & (distinct == 4)] = ONE_PAIR
    cat[(maxc == 2) & (distinct == 3)] = TWO_PAIR
    cat[(maxc == 3) & (distinct == 3)] = THREE_OF_A_KIND
    cat[(maxc == 3) & (distinct == 2)] = FULL_HOUSE
    cat[maxc == 4] = FOUR_OF_A_KIND
    cat[five & is_straight & ~flush] = STRAIGHT
    cat[five & flush & ~is_straight] = FLUSH
    cat[five & flush & is_straight] = STRAIGHT_FLUSH

    # Tiebreak: cards ordered by (count desc, rank desc), read as base-16 digits.
    cnt_of_card = np.take_along_axis(counts, ranks, axis=1)
    key = -np.sort(-(cnt_of_card * 16 + ranks), axis=1) % 16
    tb = (((key[:, 0] * 16 + key[:, 1]) * 16 + key[:, 2]) * 16 + key[:, 3]) * 16 + key[:, 4]
    tb = np.where(is_straight, straight_val, tb)

    pos = np.empty(9, dtype=np.int64)
    pos[list(ranking.order)] = np.arange(9)
    return pos[cat] * (16 ** 5) + tb, cat


@dataclass(frozen=True, eq=False)
class Tables:
    ranking: HandRanking
    rank_table: np.ndarray       # uint16[5**9], non-flush strength by rank key
    flush_table: np.ndarray      # uint16[512], flush strength by rank mask (0 = none)
    category_of: np.ndarray      # int8[num_strengths + 1], category id by strength
    scores: np.ndarray           # sorted unique sparse scores; strength = index + 1
    rank_dict: dict = field(repr=False)
    flush_dict: dict = field(repr=False)

    @property
    def num_strengths(self) -> int:
        return len(self.scores)


@lru_cache(maxsize=None)
def get_tables(ranking: HandRanking = DEFAULT_RANKING) -> Tables:
    multisets = np.array(
        [c for c in combinations_with_replacement(range(NUM_RANKS), 5) if len(set(c)) > 1],
        dtype=np.int64,
    )  # 1,278 rank multisets (five of a kind impossible)
    flushsets = np.array(list(combinations(range(NUM_RANKS), 5)), dtype=np.int64)  # 126
    s1, c1 = _score(multisets, np.zeros(len(multisets), bool), ranking)
    s2, c2 = _score(flushsets, np.ones(len(flushsets), bool), ranking)
    scores = np.unique(np.concatenate([s1, s2]))
    st1 = np.searchsorted(scores, s1) + 1
    st2 = np.searchsorted(scores, s2) + 1

    rank_table = np.zeros(RANK_KEY_SPACE, dtype=np.uint16)
    keys = POW5[multisets].sum(axis=1)
    rank_table[keys] = st1
    flush_table = np.zeros(1 << NUM_RANKS, dtype=np.uint16)
    masks = (1 << flushsets).sum(axis=1)
    flush_table[masks] = st2
    category_of = np.zeros(len(scores) + 1, dtype=np.int8)
    category_of[st1] = c1
    category_of[st2] = c2
    for t in (rank_table, flush_table, category_of):
        t.setflags(write=False)
    return Tables(
        ranking=ranking,
        rank_table=rank_table,
        flush_table=flush_table,
        category_of=category_of,
        scores=scores,
        rank_dict={int(k): int(v) for k, v in zip(keys, st1)},
        flush_dict={int(k): int(v) for k, v in zip(masks, st2)},
    )


# ---------------------------------------------------------------- 5-card full table

_BINOM = np.array([[comb(n, k) for k in range(6)] for n in range(DECK_SIZE + 1)], dtype=np.int64)
NUM_FIVE_CARD_HANDS = comb(DECK_SIZE, 5)  # 376,992


def colex_index(cards: Sequence[int]) -> int:
    """Combinatorial (colex) index of a set of 5 distinct cards, in 0..376,991."""
    s = sorted(int(c) for c in cards)
    return sum(comb(c, i + 1) for i, c in enumerate(s))


def all_five_card_hands() -> np.ndarray:
    """(376992, 5) int8 array of every five-card hand, row i has colex index i."""
    combos = np.array(list(combinations(range(DECK_SIZE), 5)), dtype=np.int64)
    idx = _BINOM[combos, np.arange(1, 6)].sum(axis=1)
    out = np.empty_like(combos)
    out[idx] = combos
    return out.astype(np.int8)


@lru_cache(maxsize=None)
def five_card_table(ranking: HandRanking = DEFAULT_RANKING) -> np.ndarray:
    """uint16[376992]: strength of every five-card hand by colex index.

    Builds in about 0.3 s from scratch, so it is only cached in memory;
    ``five_card_table.cache_clear()`` forces a rebuild.
    """
    t = get_tables(ranking)
    hands = all_five_card_hands().astype(np.int64)
    ranks = hands >> 2
    suits = hands & 3
    flush = (suits == suits[:, :1]).all(axis=1)
    score, _ = _score(ranks, flush, ranking)
    strength = np.searchsorted(t.scores, score) + 1
    assert np.array_equal(t.scores[strength - 1], score)
    out = strength.astype(np.uint16)
    out.setflags(write=False)
    return out


# ---------------------------------------------------------------- evaluator


class Evaluator:
    """Evaluates short-deck hands under one HandRanking."""

    def __init__(self, ranking: HandRanking = DEFAULT_RANKING):
        self.ranking = ranking
        self.tables = get_tables(ranking)
        self._rd = self.tables.rank_dict
        self._fd = self.tables.flush_dict

    # -- single hands (pure Python, no numpy overhead)

    def evaluate5(self, cards: Sequence[int]) -> int:
        """Strength of exactly five cards."""
        key = 0
        suit = cards[0] & 3
        flush = True
        mask = 0
        for c in cards:
            key += _POW5_T[c >> 2]
            mask |= 1 << (c >> 2)
            if c & 3 != suit:
                flush = False
        if flush:
            return self._fd[mask]
        return self._rd[key]

    def evaluate5_indexed(self, cards: Sequence[int]) -> int:
        """Same as evaluate5, but through the full 376,992-entry table."""
        return int(five_card_table(self.ranking)[colex_index(cards)])

    def omaha(self, hole: Sequence[int], board: Sequence[int]) -> int:
        """Best strength using exactly 2 of the hole cards and exactly 3 board cards.

        Works for any hole size >= 2 and board size 3..5.
        """
        rd = self._rd
        hk = [_POW5_T[c >> 2] for c in hole]
        bk = [_POW5_T[c >> 2] for c in board]
        nh = len(hk)
        pk = {hk[a] + hk[b] for a in range(nh) for b in range(a + 1, nh)}
        tk = {bk[a] + bk[b] + bk[c] for a, b, c in _TRIPLES[len(bk)]}
        best = max([rd[p + t] for p in pk for t in tk])

        sc = [0, 0, 0, 0]
        for c in board:
            sc[c & 3] += 1
        for s in range(4):
            if sc[s] >= 3:
                hr = [c >> 2 for c in hole if c & 3 == s]
                if len(hr) >= 2:
                    br = [c >> 2 for c in board if c & 3 == s]
                    fd = self._fd
                    tmasks = [(1 << x) | (1 << y) | (1 << z) for x, y, z in combinations(br, 3)]
                    for a, b in combinations(hr, 2):
                        pm = (1 << a) | (1 << b)
                        for tm in tmasks:
                            v = fd[pm | tm]
                            if v > best:
                                best = v
                break  # at most one suit can have 3+ cards on a 5-card board
        return best

    def category(self, strength: int) -> int:
        return int(self.tables.category_of[strength])

    def category_name(self, strength: int) -> str:
        return CATEGORY_NAMES[self.category(strength)]

    # -- vectorized

    def evaluate5_batch(self, cards: np.ndarray) -> np.ndarray:
        """(N, 5) cards -> (N,) strengths."""
        cards = np.asarray(cards, dtype=np.int64)
        r = cards >> 2
        s = cards & 3
        flush = (s == s[:, :1]).all(axis=1)
        nonflush = self.tables.rank_table[POW5[r].sum(axis=1)]
        fl = self.tables.flush_table[np.bitwise_or.reduce(1 << r, axis=1)]
        return np.where(flush, fl, nonflush)

    def omaha_batch(self, holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
        """(N, 5) hole cards and (N, B) boards (B = 3, 4 or 5) -> (N,) strengths.

        Either argument may also be a single 1-D hand, broadcast against the other.
        """
        holes = np.asarray(holes, dtype=np.int64)
        boards = np.asarray(boards, dtype=np.int64)
        if holes.ndim == 1:
            holes = holes[None, :]
        if boards.ndim == 1:
            boards = boards[None, :]
        n = max(len(holes), len(boards))
        holes = np.broadcast_to(holes, (n, holes.shape[1]))
        boards = np.broadcast_to(boards, (n, boards.shape[1]))
        nh = holes.shape[1]
        pa, pb = (np.array(x) for x in zip(*combinations(range(nh), 2)))
        ta, tb, tc = (np.array(x) for x in zip(*_TRIPLES[boards.shape[1]]))

        hr, hs = holes >> 2, holes & 3
        br, bs = boards >> 2, boards & 3
        hp, bp = POW5[hr], POW5[br]
        pk = hp[:, pa] + hp[:, pb]
        tk = bp[:, ta] + bp[:, tb] + bp[:, tc]
        best = self.tables.rank_table[pk[:, :, None] + tk[:, None, :]].max(axis=(1, 2))

        # Flushes: only rows whose board has 3+ of one suit.
        suit_counts = (bs[:, :, None] == np.arange(4)).sum(axis=1)
        rows = np.nonzero(suit_counts.max(axis=1) >= 3)[0]
        if len(rows):
            hr_, hs_, br_, bs_ = hr[rows], hs[rows], br[rows], bs[rows]
            psuit = np.where(hs_[:, pa] == hs_[:, pb], hs_[:, pa], 4)
            tsuit = np.where((bs_[:, ta] == bs_[:, tb]) & (bs_[:, tb] == bs_[:, tc]), bs_[:, ta], 5)
            pm = (1 << hr_[:, pa]) | (1 << hr_[:, pb])
            tm = (1 << br_[:, ta]) | (1 << br_[:, tb]) | (1 << br_[:, tc])
            fl = self.tables.flush_table[pm[:, :, None] | tm[:, None, :]]
            fl = np.where(psuit[:, :, None] == tsuit[:, None, :], fl, 0).max(axis=(1, 2))
            best = best.copy()
            best[rows] = np.maximum(best[rows], fl)
        return best


@lru_cache(maxsize=None)
def get_evaluator(ranking: HandRanking = DEFAULT_RANKING) -> Evaluator:
    return Evaluator(ranking)


def evaluate_omaha(hole: Sequence[int], board: Sequence[int], ranking: HandRanking = DEFAULT_RANKING) -> int:
    return get_evaluator(ranking).omaha(hole, board)


def evaluate_omaha_batch(holes, boards, ranking: HandRanking = DEFAULT_RANKING) -> np.ndarray:
    return get_evaluator(ranking).omaha_batch(holes, boards)


def evaluate5(cards: Sequence[int], ranking: HandRanking = DEFAULT_RANKING) -> int:
    return get_evaluator(ranking).evaluate5(cards)
