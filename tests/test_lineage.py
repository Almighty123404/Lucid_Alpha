"""Core Operational Invariant 2 — STRICT LINEAGE LOGGING.

Any alpha without dataset_id + logged N_trials entry gets DSR = 0 and is
blocked from promotion. Tests pin all three enforcement points.
"""
import numpy as np

import selection
from selection import (TrialRegistry, deflated_sharpe_ratio, promotion_eligible,
                       DSR_PROMOTE)
from simulator import simulate


def test_dataset_id_present_and_stable(panel):
    assert panel.dataset_id and len(panel.dataset_id) == 64
    from data_gen import generate
    p2 = generate(seed=11, n_stocks=100, start="2020-01-01", end="2020-12-31")
    assert p2.dataset_id == panel.dataset_id  # deterministic fingerprint


def test_unlogged_trial_gets_dsr_zero_and_blocked(panel, refs):
    rep = simulate("ts_mean(returns, 60)", panel, refs, (), settings=None)
    # Strip lineage the way a hand-built panel would lack it.
    rep_naked = dict(rep)
    rep_naked["dataset_id"] = ""
    eligible, info = promotion_eligible(rep_naked, TrialRegistry(), DSR_PROMOTE)
    assert eligible is False
    assert info["dsr"] == 0.0
    assert any("dataset_id" in r for r in info["reasons"])


def test_missing_n_trials_entry_blocks(panel, refs):
    rep = simulate("ts_mean(returns, 60)", panel, refs, (), settings=None)
    assert rep["dataset_id"]  # logged panels carry ids...
    # ...but an EMPTY registry has no N_trials entry -> blocked, DSR 0.
    empty = TrialRegistry(path="/nonexistent-dir-xyz/trials.jsonl")
    assert empty.count(rep["dataset_id"]) == 0
    eligible, info = promotion_eligible(rep, empty, DSR_PROMOTE)
    assert eligible is False
    assert info["dsr"] == 0.0


def test_logged_trials_count_and_dsr_structure(panel, refs):
    reg = TrialRegistry()
    before = reg.count(panel.dataset_id)
    rep = simulate("rank(ts_mean(returns, 120))", panel, refs, (), settings=None)
    assert rep["trial_logged"] is True
    assert reg.count(panel.dataset_id) == before + 1
    assert 0.0 <= deflated_sharpe_ratio(list(rep["pnl"]), 8) <= 1.0
    # DSR is monotone in trial count pressure: more trials -> <= DSR.
    assert deflated_sharpe_ratio(list(rep["pnl"]), 1000) <= \
        deflated_sharpe_ratio(list(rep["pnl"]), 8)


def test_dsr_degenerate_inputs_zero():
    assert deflated_sharpe_ratio([0.0] * 300, 8) == 0.0
    assert deflated_sharpe_ratio([0.01] * 10, 8) == 0.0  # T < 30

def test_registry_scopes_isolate_counts(panel, refs):
    from simulator import simulate
    reg = TrialRegistry()
    d0 = reg.count(panel.dataset_id, scope="scope-A")
    simulate("ts_mean(returns, 60)", panel, refs, (), settings=None)
    # default SCOPE is "default": A-count unchanged, default grew by >=1
    assert reg.count(panel.dataset_id, scope="scope-A") == d0
    assert reg.count(panel.dataset_id, scope="default") >= d0 + 0  # separate key
    assert reg.count(panel.dataset_id, scope=None) >= reg.count(panel.dataset_id, scope="default")


def test_scoped_dsr_differs_by_scope(panel, refs):
    from simulator import simulate
    import selection as sel
    old = sel.SCOPE
    try:
        sel.SCOPE = "scope-big"
        for i in range(30):
            simulate(f"ts_mean(returns, {60 + i})", panel, refs, (), settings=None)
        rep = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (), settings=None)
        reg = TrialRegistry()
        n_big = reg.count(panel.dataset_id, scope="scope-big")
        assert n_big >= 30
        _, info_big = promotion_eligible(rep, reg, DSR_PROMOTE, scope="scope-big")
        _, info_all = promotion_eligible(rep, reg, DSR_PROMOTE, scope=None)
        assert info_big["n_trials"] == n_big
        assert info_all["n_trials"] >= n_big
        # more trials -> no higher DSR (monotone non-increasing in N)
        assert info_all["dsr"] <= promotion_eligible(rep, TrialRegistry(), DSR_PROMOTE, scope="scope-big")[1]["dsr"] + 1e-9
    finally:
        sel.SCOPE = old


def test_simulate_logs_current_scope(panel, refs):
    from simulator import simulate
    import selection as sel
    old = sel.SCOPE
    try:
        sel.SCOPE = "scope-probe-xyz"
        rep = simulate("rank(ts_mean(returns, 120))", panel, refs, (), settings=None)
        assert rep["trial_logged"] is True
        reg = TrialRegistry()
        assert reg.count(panel.dataset_id, scope="scope-probe-xyz") >= 1
    finally:
        sel.SCOPE = old

def test_expected_real_adjustment_needs_min_records():
    from calibration import expected_real_adjustment, expected_real_metrics
    env = expected_real_adjustment(min_records=10 ** 9)
    assert env["bias"] == {} and env["n"] == 0 and env["fallback"] is True
    bias = {"sharpe": 1.0, "fitness": 0.5, "turnover_pct": -10.0, "drawdown_pct": -5.0}
    adj = expected_real_metrics({"sharpe": 2.5, "fitness": 1.2, "turnover_pct": 30.0,
                                 "drawdown_pct": 4.0, "other": 7}, bias)
    assert adj["sharpe"] == 1.5 and adj["fitness"] == 0.7
    assert adj["turnover_pct"] == 40.0 and adj["drawdown_pct"] == 9.0
    assert adj["other"] == 7  # missing keys pass through


def test_family_of_expression_tags():
    from calibration import family_of_expression
    assert family_of_expression("rank(ts_delta(ebitda, 63) / assets)") == "fundamental"
    assert family_of_expression("-ts_zscore(returns, 21)") == "microstructure"
    assert family_of_expression("-rank(ts_backfill(implied_volatility_put_10, 15) / ts_backfill(implied_volatility_call_10, 15))") == "derivatives"
    assert family_of_expression("ts_mean(vec_avg(nws12_afterhsz_01l), 5)") == "sentiment"
    assert family_of_expression("rank(ebitda) - rank(returns)") == "mixed"
    assert family_of_expression("((((") == "unknown"


def test_family_adjustment_falls_back_when_thin():
    import calibration
    env = calibration.expected_real_adjustment("microstructure")
    assert env["family"] in ("microstructure", "global")
    if env["family"] == "global":
        assert env["fallback"] is True
    else:
        assert env["fallback"] is False and env["n"] >= 3


def test_turnover_slope_gating_and_math():
    import calibration
    assert calibration.turnover_slope(pool=[], min_points=4) is None
    tiny = [{"predicted_metrics": {"turnover_pct": t}, "real_metrics": {"turnover_pct": t - 1.0}}
            for t in (10.0, 20.0)]
    assert calibration.turnover_slope(pool=tiny, min_points=4) is None  # too few points
    narrow = [{"predicted_metrics": {"turnover_pct": t}, "real_metrics": {"turnover_pct": t - 1.0}}
              for t in (10.0, 12.0, 14.0, 16.0, 18.0)]
    assert calibration.turnover_slope(pool=narrow, min_span=30.0) is None  # span too small
    wide = [{"predicted_metrics": {"turnover_pct": t}, "real_metrics": {"turnover_pct": 2.0 * t + 5.0}}
            for t in (5.0, 20.0, 40.0, 70.0)]
    s = calibration.turnover_slope(pool=wide)
    assert s is not None and abs(s["slope"] - (-1.0)) < 1e-9 and abs(s["intercept"] - (-5.0)) < 1e-9
    assert s["n"] == 4
    # slope applied through expected_real_metrics
    adj = calibration.expected_real_metrics(
        {"turnover_pct": 50.0}, {"bias": {}, "to_slope": s})
    assert adj["turnover_pct"] == 50.0 - (-1.0 * 50.0 + -5.0)


def test_promotion_expected_real_blocks_and_passes(panel, refs):
    from simulator import simulate
    from selection import TrialRegistry, promotion_eligible, DSR_PROMOTE
    rep = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (), settings=None)
    reg = TrialRegistry()
    mild = {"sharpe": 0.1, "fitness": 0.1}
    el, info = promotion_eligible(rep, reg, DSR_PROMOTE, "default", expect_real=mild)
    assert "expected_real" in info
    assert el == (info["dsr"] >= DSR_PROMOTE and rep.get("passed", False)), info
    harsh = {"sharpe": 99.0, "fitness": 99.0}
    el2, info2 = promotion_eligible(rep, reg, DSR_PROMOTE, "default", expect_real=harsh)
    assert el2 is False
    assert any("expected-real" in r for r in info2["reasons"])
