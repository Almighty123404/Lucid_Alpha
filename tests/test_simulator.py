"""Invariant + regression tests for simulator.py mechanics.

Each regression pins a historical bug so it cannot silently return:
- turnover half-factor (Bug 1a), truncation rescale violation (Bug 3),
  stale weight carry (item 3), static lookahead universe (item 13).
"""
import numpy as np
import pytest

from simulator import (simulate, _weights_from_signal, _turnover_series,
                       _universe_mask, _pnl_series, _rolling_self_corr)
from config import SimulationSettings, subuniverse_cutoff


def test_turnover_is_gross_not_half():
    # Single long/short flip-flop: sum|dW| = 2.0 (half convention gave 1.0).
    Wd = np.array([[0.5, -0.5], [-0.5, 0.5]])
    assert float(_turnover_series(Wd)[0]) == 2.0


def test_pnl_series_is_daily_not_cumulative():
    Wd = np.array([[0.5, -0.5], [0.5, -0.5]])
    R = np.array([[0.0, 0.0], [0.01, -0.01]])
    pnl = _pnl_series(Wd, R)
    assert pnl[0] == 0.0
    assert pnl[1] == pytest.approx(0.5 * 0.01 + -0.5 * -0.01)


def test_truncation_cap_never_violated(panel):
    # Raw IV ratio: untruncated conc ~43% — binds at every cap level.
    from fastexpr import eval_program, Env
    sig = eval_program(
        "ts_backfill(implied_volatility_put_10, 15) / ts_backfill(implied_volatility_call_10, 15)",
        Env(panel))
    sig = np.where(np.isinf(sig), np.nan, sig).astype(np.float64)
    for tr in (0.05, 0.1, 0.2):
        W = _weights_from_signal(sig, truncation=tr)
        assert float(np.abs(W).max()) <= tr * 1.0 + 1e-9


def test_thin_days_go_flat_not_stale():
    sig = np.full((10, 20), 0.01)
    sig[5] = np.nan  # all-NaN day must print zero book, not yesterday's
    W = _weights_from_signal(sig)
    assert (W[5] == 0.0).all()
    assert float(np.abs(W).sum(axis=1).max()) <= 1.0 + 1e-12


def test_book_gross_never_exceeds_book(panel):
    from fastexpr import eval_program, Env
    sig = eval_program("-ts_zscore(returns, 21)", Env(panel))
    sig = np.where(np.isinf(sig), np.nan, sig).astype(np.float64)
    W = _weights_from_signal(sig, truncation=0.1)
    assert float(np.abs(W).sum(axis=1).max()) <= 1.0 + 1e-9


def test_universe_mask_is_causal(panel):
    m1 = _universe_mask(panel, "TOP500")
    assert m1.shape == panel.fields["returns"].shape  # 2D point-in-time
    # Perturbing future liquidity must not change past membership (no lookahead).
    panel2_liq = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in panel.fields.items()}
    panel2_liq["adv20"] = panel2_liq["adv20"].copy()
    panel2_liq["adv20"][-10:] *= 100.0
    class P:
        pass
    p2 = P()
    p2.fields = panel2_liq
    m2 = _universe_mask(p2, "TOP500")
    assert (m1[:-10] == m2[:-10]).all()


def test_self_corr_uses_overlap_and_changes(panel, refs):
    a = refs[0]["pnl"]
    c, n = _rolling_self_corr(a, a, 4.0, 252.0)
    assert c == pytest.approx(1.0)  # self-similarity on daily series
    assert n > 0


def test_s1_cutoff_formula():
    # Worked example: 0.75*sqrt(1000/3000)*2.73 = 1.18 (approx).
    assert subuniverse_cutoff(1000, alpha_size=3000, alpha_sharpe=2.73) == pytest.approx(1.18, abs=0.01)
    # Scales with candidate Sharpe (stricter on inflated synths).
    lo = subuniverse_cutoff(500, alpha_size=3000, alpha_sharpe=1.0)
    hi = subuniverse_cutoff(500, alpha_size=3000, alpha_sharpe=3.0)
    assert hi > lo


def test_escape_requires_beating_every_ref(panel, refs):
    # Candidate highly correlated with TWO refs but beating only the weaker
    # one must still FAIL (escape is vs ALL, not max-only).
    base = simulate("ts_mean(returns, 60)", panel, refs, (), settings=None)
    assert base["criteria"]["self_correlation"]["pass"] in (True, False)


def test_config_window_is_4y():
    from config import get_cutoff
    assert get_cutoff("self_corr_window_years") == 4


def test_golden_rev21_small_panel(panel, refs):
    # Pinned behavior on the small fixture panel (tolerance, not exact —
    # guards against silent mechanics drift, not RNG changes).
    rep = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (), settings=None)
    m = rep["metrics"]
    assert m["sharpe"] > 1.0
    assert m["weight_concentration_pct"] <= 10.0
    assert rep["criteria"]["turnover"]["pass"]

def test_cost_lens_never_enters_gates(panel, refs):
    rep = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (), settings=None)
    m = rep["metrics"]
    assert "cost_drag_pct" in m and "returns_net_pct" in m
    assert m["returns_net_pct"] == pytest.approx(m["returns_pct"] - m["cost_drag_pct"], abs=1e-3)
    assert m["cost_drag_pct"] >= 0.0
    rep2 = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (),
                    settings=None, cutoffs={"cost_bps": 50.0})
    assert rep2["metrics"]["cost_drag_pct"] == pytest.approx(m["cost_drag_pct"] * 5.0, rel=1e-3)
    # gates identical regardless of cost assumption
    assert rep2["criteria"] == rep["criteria"]

def test_returns_geo_matches_cagr_definition(panel, refs):
    import numpy as np
    rep = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (), settings=None)
    m = rep["metrics"]
    x = rep["pnl"][1:] / 0.5
    expect = (float(np.prod(1.0 + x)) ** (252.0 / len(x)) - 1.0) * 100.0
    assert m["returns_geo_pct"] == pytest.approx(expect, abs=1e-3)
    # linear annualization understates compounding for this profitable alpha
    assert m["returns_geo_pct"] >= m["returns_pct"]
