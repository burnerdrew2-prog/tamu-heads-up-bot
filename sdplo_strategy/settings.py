"""All tunable knobs in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    # --- range tracking -------------------------------------------------
    range_size: int = 1500            # opponent hands in the weighted sample
    ess_min_frac: float = 0.30        # refill when ESS < this * range_size
    refill_frac: float = 0.75         # fresh hands added on refill, * range_size
    max_range_size: int = 3500        # hard cap after refills
    strength_runouts: int = 2         # shared runouts for the postflop strength proxy
    # --- likelihood (opponent action given hand) --------------------------
    likelihood_floor: float = 0.06    # mix-in of the base rate: no hand is ever ruled out
    smooth_width: float = 0.06        # softness of the strength thresholds
    # --- equity / time --------------------------------------------------
    sample_budget: int = 600          # Monte Carlo samples per decision
    preflop_budget_frac: float = 0.6  # preflop gets this share of the budget
    time_limit: float = 1.0           # seconds per decision we must stay under (hard)
    time_target_frac: float = 0.35    # aim to use this share of the limit
    hard_stop_frac: float = 0.7       # Monte Carlo stops at this share of the limit
    min_budget_scale: float = 0.05
    match_time_budget: Optional[float] = None  # total seconds per match, if the rules cap it
    hands_per_match: int = 2000
    request_grace: float = 0.0        # seconds of each decision not charged to the bank
    request_overhead: float = 0.0     # charged time per decision we cannot see (server, JSON)
    bank_reserve: float = 0.0         # seconds of the bank never planned for
    cheap_below: float = 0.0          # per-decision allowance (s) under which the cheap path runs
    cheap_hint: float = 0.3           # with a bank: allowance / recent decision cost under which it runs
    # --- decisions ------------------------------------------------------
    raise_fracs: tuple = (1 / 3, 2 / 3, 1.0)   # fractions of the pot-limit raise
    realization_ip: float = 0.95
    realization_oop: float = 0.85
    preflop_score_realization: float = 0.25    # better preflop hands realise more equity
    bluff_eq_max: tuple = (0.0, 0.40, 0.35, 0.25)  # "weak hand" threshold by street (no preflop bluff mix)
    bluff_scale: float = 1.0
    bluff_raise_scale: float = 0.4    # bluff-raises are mixed in less than bluff-bets
    bluff_min_fold: float = 0.15      # never balance-bluff an opponent who folds less than this
    max_bluff_prob: float = 0.6
    bluff_ev_tolerance: float = 0.15  # mixed-in bluffs may cost at most this * pot vs the best option
    commit_fold_ref: float = 0.30     # opponent's fold share shrinks when their call needs < this equity
    use_band_table: bool = True       # preflop equity vs a range from the band table when loaded
    strength_fold_beta: float = 6.0   # fold odds * exp(-beta * (range strength now - usual)); 0 = off
    split_informativeness: bool = True  # fold/call informativeness measured apart from aggression
    called_realization_oop: float = 1.0  # extra realization factor when our postflop bet is called OOP (at spr>=1)
    called_realization_ip: float = 1.0
    preflop_max_raise_only: bool = False   # preflop, consider only the largest legal raise
    fold_scale: float = 1.0           # multiply the predicted fold share of our bets (1 = off)
    postflop_eq_realization: float = 0.0  # flop/turn realization slope by equity (0 = off)
    rz_min: float = 0.5               # ... clipped to [rz_min, rz_max]
    rz_max: float = 1.5
    fold_scale_from_street: int = 1   # ... on this street and later
    postflop_max_raise_only: bool = False
    ev_epsilon: float = 1e-9
    # --- fallback ---------------------------------------------------------
    cheap_call_pot_frac: float = 0.34   # call in fallback if to_call <= this * pot
    cheap_call_stack_frac: float = 0.10 # ... and <= this * our stack
    # --- misc -------------------------------------------------------------
    seed: Optional[int] = None
    showdown_ref_hands: int = 120     # random hands to rank a shown hand against
    extra: dict = field(default_factory=dict)
