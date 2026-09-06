"""Core Operational Invariant 1 — GATE INVARIANCE.

No upgrade may loosen a gate to raise passing rates. This file pins every
gate threshold, convention, and disciplinary behavior. Changing a gate
requires bumping config.GATE_VERSION with written justification; CI fails
otherwise. Tightening is allowed; loosening is a defect.
"""
import numpy as np

from config import (GATE_VERSION, GATE_INVARIANTS, get_cutoff,
                    subuniverse_cutoff)
from simulator import _turnover_series
from agents import dimension_check, _score


def test_gate_version_pinned():
    assert GATE_VERSION == 1


def test_cutoffs_match_invariants_exactly():
    assert get_cutoff("sharpe_min") == GATE_INVARIANTS["sharpe_min"] == 1.25
    assert get_cutoff("fitness_min") == GATE_INVARIANTS["fitness_min"] == 1.0
    assert get_cutoff("turnover_min") == GATE_INVARIANTS["turnover_min"] == 0.01
    assert get_cutoff("turnover_max") == GATE_INVARIANTS["turnover_max"] == 0.70
    assert get_cutoff("weight_conc_max") == GATE_INVARIANTS["weight_conc_max"] == 0.10
    assert get_cutoff("self_corr_max") == GATE_INVARIANTS["self_corr_max"] == 0.70
    assert get_cutoff("corr_sharpe_improve") == GATE_INVARIANTS["corr_sharpe_improve"] == 1.10
    assert get_cutoff("self_corr_window_years") == GATE_INVARIANTS["self_corr_window_years"] == 4


def test_sub_universe_uses_s1_relative_formula():
    # Relative (scales with candidate Sharpe), never a fixed fallback.
    assert subuniverse_cutoff(1000, alpha_size=3000, alpha_sharpe=2.73) > \
        subuniverse_cutoff(1000, alpha_size=3000, alpha_sharpe=1.0) > 0


def test_turnover_convention_is_gross():
    Wd = np.array([[0.5, -0.5], [-0.5, 0.5]])
    assert float(_turnover_series(Wd)[0]) == 2.0  # half convention gives 1.0


def test_dimension_gate_still_rejects():
    assert dimension_check("close + volume") != []
    assert dimension_check("vec_avg(returns)") != []


def test_parsimony_tiebreak_intact():
    rep = {"metrics": {"fitness": 1.0, "sharpe": 2.0},
           "criteria": {"a": {"pass": True}}}
    assert _score(rep, "rank(returns)") > _score(rep, "rank(ts_mean(returns, 120))")
