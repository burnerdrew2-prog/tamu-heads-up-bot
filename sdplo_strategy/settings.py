"""All tunable knobs in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    # --- range tracking -------------------------------------------------
    range_size: int = 1500            # opponent hands in the weighted sample
    range_top_up: bool = True         # refill to range_size when board cards remove hands
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
    time_free_without_bank: bool = True   # no bank: no clock-dependent sample cuts or deadlines
    hands_per_match: int = 2000
    request_grace: float = 0.0        # seconds of each decision not charged to the bank
    request_overhead: float = 0.0     # charged time per decision we cannot see (server, JSON)
    bank_reserve: float = 0.0         # seconds of the bank never planned for
    cheap_below: float = 0.0          # per-decision allowance (s) under which the cheap path runs
    cheap_hint: float = 0.3           # with a bank: allowance / recent decision cost under which it runs
    overhead_ref_cost: float = 0.0    # >0: scale request_overhead by recent decision cost / this
    split_open_size: bool = False     # preflop: their response to our pot-size open is its own context
    pot_pressure_gate_n: int = 0      # >0: no fold discount on pot bets once they fold >= rate over n
    pot_pressure_gate_rate: float = 0.55
    lite_mode: bool = False           # tiered time control: full / lite / cheap per decision
    contention_overhead: bool = False # scale request_overhead by measured wall/CPU of our work
    cheap_version: int = 1            # 1 = original cheap path, 2 = model-aware cheap path
    size_bluff_gate: int = 0          # >0: their bets use a bluff share by size and street group
    size_split_aggr: bool = False     # their pot-size aggression = the top (pot share) of their aggression
    size_split_min_n: int = 12        # ... once this many aggressive actions in the context group
    size_split_gate_n: int = 4        # ... and their shown pot-size aggression beat smaller aggression
    size_split_margin: float = 0.05   #     by this much in mean strength over >= gate_n hands each
    size_split_streets: tuple = (0, 1, 2, 3)   # ... on these streets
    pf_tilt_gate: int = 0             # >0: once this many of their preflop raises in a context were shown,
    pf_tilt_prior: float = 4.0        #     tilt the range after such a raise toward the shown strength
    pf_tilt_margin: float = 0.02      #     (shrunk with this many prior hands; only when it is stronger)
    pf_tilt_contexts: tuple = ("vs_raise", "limp_vs_raise", "vs_reraise", "vs_4bet")
    eq_calibrate: bool = False        # facing their bet/raise: subtract the measured overestimate of our
    eq_calib_kinds: tuple = ("raise",)  # equity (est vs true equity vs the hands they showed) for these
    eq_calib_streets: tuple = (1, 2)  # spots ("raise" = they raised our bet, "bet" = their first bet)
    eq_calib_gate: int = 6            # ... once this many such spots (pooled over streets) were shown
    eq_calib_k: float = 10.0          # ... shrunk with k pseudo-spots toward the pool (the pool toward 0)
    eq_calib_max: float = 0.25        # ... never more than this; never added
    raise_cap: bool = False           # facing a raise of our bet: no re-raise without a full house or better
    raise_cap_streets: tuple = (1,)   # ... on these streets
    raise_cap_gate: str = "calib"     # ... once their raises proved value-heavy: "calib" = our measured equity
    raise_cap_bias_min: float = 0.05  #     overestimate vs their shown raises >= this (eq_calib_gate spots);
    raise_cap_bluff_max: float = 0.15 #     "bluff" = bluff_raise share <= this after raise_cap_gate_n
    raise_cap_gate_n: int = 6         #     shown raises; "always" = no gate
    raise_cap_eq_min: float = 1.01    # ... a re-raise also stays allowed at eq_reraise >= this (1.01 = never)
                                      #     once that many such bets were shown down
    force_mode: str = ""              # testing: "cheap" / "lite" / "full" for every decision
    load_overhead_slope: float = 0.0  # ... and by 1 + slope * (loadavg per CPU - knee), 1-4x
    load_overhead_knee: float = 0.85
    lite_range_size: int = 64         # lite: particles in a hand started in lite mode
    lite_sample_frac: float = 0.5     # lite: share of the Monte Carlo budget
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
    rz_on_call: bool = True           # apply the slope when pricing a call too
    split_bluff_decide: bool = False  # their re-raise range uses the raise-specific bluff share
    split_bluff_observe: bool = False # ... and so does the range update after their raises
    cap_reraise_equity: bool = False  # equity vs their re-raising range <= vs their calling range
    split_preflop_4bet: bool = False  # preflop: facing a 4-bet+ is its own context (vs 3-bet)
    strength_fold_from_street: int = 0  # strength_adjust applies from this street on
    preflop_range_realization: float = 0.0  # preflop rho *= 1 - c * (their range strength - 0.5)+
    scare_bluff_split: bool = False   # own bluff share for their bets on scare-card turns / rivers
    rz_min: float = 0.5               # ... clipped to [rz_min, rz_max]
    rz_max: float = 1.5
    fold_scale_from_street: int = 1   # ... on this street and later
    fold_calibrate: bool = False      # learn the fold multiplier online (fold_scale = its prior)
    fold_calibrate_strength: float = 20.0   # prior weight, in bets
    fold_calibrate_min: float = 0.4
    fold_calibrate_max: float = 1.4
    postflop_max_raise_only: bool = False
    # preflop rules from the CFR preflop solve of this game (percentile = PreflopScorer; 0 = off)
    pf_war_raise_min_pct: float = 0.0       # limped pot, facing a re-raise: raise only at pct >= this, else call
    pf_3bet_raise_min_pct: float = 0.0      # button facing a 3-bet after opening: 4-bet only at pct >= this, else call
    pf_open_fold_below: float = 0.0         # button first in: fold below this pct
    pf_limp_raised_fold_below: float = 0.0  # button after limping, facing the BB's raise: fold below this pct
    ev_epsilon: float = 1e-9
    # --- fallback ---------------------------------------------------------
    cheap_call_pot_frac: float = 0.34   # call in fallback if to_call <= this * pot
    cheap_call_stack_frac: float = 0.10 # ... and <= this * our stack
    # --- misc -------------------------------------------------------------
    seed: Optional[int] = None
    showdown_ref_hands: int = 120     # random hands to rank a shown hand against
    extra: dict = field(default_factory=dict)
