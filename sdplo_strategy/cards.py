"""Card encoding used inside sdplo_strategy.

A card is an int 0..35: ``rank * 4 + suit``.
Ranks: 0..8 = 6 7 8 9 T J Q K A.  Suits: 0..3 = c d h s.

The core may use another encoding; ``adapter.py`` translates at the boundary
so nothing else in this package needs to know.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

RANK_CHARS = "6789TJQKA"
SUIT_CHARS = "cdhs"
N_RANKS = 9
N_SUITS = 4
N_CARDS = 36
DECK = tuple(range(N_CARDS))
DECK_ARRAY = np.arange(N_CARDS, dtype=np.int16)
ACE = 8

# Street index -> number of board cards.
BOARD_LEN = {0: 0, 1: 3, 2: 4, 3: 5}
STREET_OF_BOARD_LEN = {0: 0, 3: 1, 4: 2, 5: 3}


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


def make_card(rank: int, suit: int) -> int:
    return rank * 4 + suit


def parse_card(text: str) -> int:
    text = text.strip()
    if len(text) != 2:
        raise ValueError(f"bad card {text!r}")
    r = RANK_CHARS.find(text[0].upper())
    s = SUIT_CHARS.find(text[1].lower())
    if r < 0 or s < 0:
        raise ValueError(f"bad card {text!r}")
    return make_card(r, s)


def parse_cards(text: str | Sequence) -> tuple[int, ...]:
    """'AhKd6c' or 'Ah Kd 6c' or a sequence of str/int -> tuple of ints."""
    if isinstance(text, str):
        t = text.replace(" ", "").replace(",", "")
        return tuple(parse_card(t[i:i + 2]) for i in range(0, len(t), 2))
    return tuple(parse_card(c) if isinstance(c, str) else int(c) for c in text)


def card_str(card: int) -> str:
    return RANK_CHARS[rank_of(card)] + SUIT_CHARS[suit_of(card)]


def cards_str(cards: Iterable[int]) -> str:
    return "".join(card_str(int(c)) for c in cards)


def remaining_deck(dead: Iterable[int]) -> np.ndarray:
    dead_set = set(int(c) for c in dead)
    return np.array([c for c in DECK if c not in dead_set], dtype=np.int16)


def random_hands(rng: np.random.Generator, available: np.ndarray, n: int, k: int = 5) -> np.ndarray:
    """n random k-card hands (rows sorted) drawn without replacement from ``available``.

    Rows are independent draws (two rows may share cards).
    """
    m = len(available)
    keys = rng.random((n, m))
    idx = np.argpartition(keys, k - 1, axis=1)[:, :k]
    hands = available[idx]
    hands.sort(axis=1)
    return hands.astype(np.int16)
