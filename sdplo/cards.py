"""Short-deck cards.

A short deck has 36 cards: ranks 6, 7, 8, 9, T, J, Q, K, A and suits c, d, h, s.
Internally a card is an int in 0..35::

    card = rank_index * 4 + suit_index

where rank_index 0..8 means 6..A and suit_index 0..3 means c, d, h, s.
So ``card >> 2`` is the rank index and ``card & 3`` is the suit index.
"""

from __future__ import annotations

from typing import Iterable, Sequence

RANKS = "6789TJQKA"
SUITS = "cdhs"
NUM_RANKS = len(RANKS)
NUM_SUITS = len(SUITS)
DECK_SIZE = NUM_RANKS * NUM_SUITS  # 36
FULL_DECK: tuple[int, ...] = tuple(range(DECK_SIZE))


def make_card(rank_index: int, suit_index: int) -> int:
    if not (0 <= rank_index < NUM_RANKS and 0 <= suit_index < NUM_SUITS):
        raise ValueError(f"bad rank/suit index {rank_index}, {suit_index}")
    return rank_index * 4 + suit_index


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


def parse_card(text: str) -> int:
    """Parse one card such as "As", "6h" or "10d" (10 is accepted for T)."""
    s = text.strip()
    if len(s) == 3 and s[:2] == "10":
        s = "T" + s[2]
    if len(s) != 2:
        raise ValueError(f"cannot parse card {text!r}")
    r = RANKS.find(s[0].upper())
    u = SUITS.find(s[1].lower())
    if r < 0 or u < 0:
        raise ValueError(f"cannot parse card {text!r} (short deck ranks are {RANKS})")
    return r * 4 + u


def parse_cards(text: str | Iterable[str]) -> tuple[int, ...]:
    """Parse "As6h7d", "As 6h 7d", "As,6h" or an iterable of card strings."""
    if isinstance(text, str):
        s = text.replace(",", " ").split()
        out: list[int] = []
        for chunk in s:
            chunk = chunk.replace("10", "T")
            if len(chunk) % 2:
                raise ValueError(f"cannot parse cards {text!r}")
            out.extend(parse_card(chunk[i:i + 2]) for i in range(0, len(chunk), 2))
        return tuple(out)
    return tuple(parse_card(t) for t in text)


def format_card(card: int) -> str:
    if not 0 <= card < DECK_SIZE:
        raise ValueError(f"card {card} out of range")
    return RANKS[card >> 2] + SUITS[card & 3]


def format_cards(cards: Sequence[int], sep: str = "") -> str:
    return sep.join(format_card(int(c)) for c in cards)


def validate_distinct(cards: Iterable[int]) -> None:
    seen: set[int] = set()
    for c in cards:
        c = int(c)
        if not 0 <= c < DECK_SIZE:
            raise ValueError(f"card {c} out of range")
        if c in seen:
            raise ValueError(f"duplicate card {format_card(c)}")
        seen.add(c)
