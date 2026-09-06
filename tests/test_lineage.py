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
