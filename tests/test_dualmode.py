"""Phase 2 dual-mode: Fast Gate bit-parity + High-Fidelity diagnostics."""
from simulator import simulate


def test_fast_gate_default_bit_parity(panel, refs):
    a = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (), settings=None)
    b = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (),
                 settings=None, mode="fast_gate")
    assert a["metrics"] == b["metrics"]
    assert a["criteria"] == b["criteria"]
    assert "high_fidelity" not in a and b["mode"] == "fast_gate"


def test_high_fidelity_lens_never_gates(panel, refs):
    expr = "ts_decay_linear(-ts_zscore(returns, 21), 10)"
    base = simulate(expr, panel, refs, (), settings=None)
    hf = simulate(expr, panel, refs, (), settings=None, mode="high_fidelity")
    assert hf["criteria"] == base["criteria"]  # gates untouched
    assert hf["metrics"]["sharpe"] == base["metrics"]["sharpe"]
    rep = hf["high_fidelity"]
    for k in ("net_sharpe", "net_fitness", "net_returns_pct", "filled_turnover_pct",
              "fill_rate", "capped_cell_frac", "shortfall_frac", "realized_cost_bps"):
        assert k in rep, k
    assert 0.0 <= rep["fill_rate"] <= 1.0
    assert 0.0 <= rep["capped_cell_frac"] <= 1.0
    assert 0.0 <= rep["shortfall_frac"] <= 1.0
    assert rep["realized_cost_bps"] >= 0.0
    # Book ($1) << synthetic ADV scale, so the 10% ADV cap rarely binds at
    # default settings (calibration fact, not a bug). Force binds with a tiny
    # cap to exercise the partial-fill path honestly.
    thin = simulate("ts_backfill(implied_volatility_put_10, 15) / ts_backfill(implied_volatility_call_10, 15)",
                    panel, refs, (), settings=None, mode="high_fidelity",
                    exec_settings={"max_participation": 1e-9})
    assert thin["high_fidelity"]["fill_rate"] < 1.0
    assert thin["high_fidelity"]["capped_cell_frac"] > 0.0
    assert thin["high_fidelity"]["realized_cost_bps"] > 0.0
    assert thin["high_fidelity"]["filled_turnover_pct"] < thin["metrics"]["turnover_pct"]


def test_high_fidelity_exec_settings_override(panel, refs):
    expr = "rank(volume / ts_mean(volume, 20))"
    calm = simulate(expr, panel, refs, (), settings=None, mode="high_fidelity",
                    exec_settings={"spread_bps": 0.0, "commission_bps": 0.0,
                                   "lambda_perm": 0.0, "eta_temp": 0.0})
    assert calm["high_fidelity"]["realized_cost_bps"] == 0.0


def test_high_fidelity_validates_raw_execution_dict(panel, refs):
    import pytest
    with pytest.raises(ValueError):
        simulate("rank(volume)", panel, refs, (), mode="high_fidelity",
                 exec_settings={"spread_bps": -1.0})
    with pytest.raises(ValueError):
        simulate("rank(volume)", panel, refs, (), mode="high_fidelity",
                 exec_settings={"max_participation": 0.0})


def test_fast_gate_ignores_execution_settings(panel, refs):
    base = simulate("rank(volume)", panel, refs, (), mode="fast_gate")
    ignored = simulate("rank(volume)", panel, refs, (), mode="fast_gate",
                       exec_settings={"spread_bps": float("nan"),
                                       "max_participation": 0.0})
    assert ignored["metrics"] == base["metrics"]
    assert ignored["criteria"] == base["criteria"]


def test_execution_settings_reject_nonfinite_numeric_values():
    import pytest
    from config import ExecutionSettings
    for field in ("spread_bps", "urgency", "lambda_perm", "eta_temp",
                  "alpha", "max_participation", "commission_bps",
                  "portfolio_notional"):
        with pytest.raises(ValueError):
            ExecutionSettings(**{field: float("inf")})


def test_high_fidelity_reports_borrow_cost(panel, refs):
    rep = simulate("rank(-returns)", panel, refs, (), mode="high_fidelity",
                   exec_settings={"spread_bps": 0.0, "commission_bps": 0.0,
                                   "lambda_perm": 0.0, "eta_temp": 0.0})
    assert "borrow_cost_bps" in rep["high_fidelity"]
    assert rep["high_fidelity"]["borrow_cost_bps"] >= 0.0


def test_capacity_frontier_declines_and_gates_fill():
    import numpy as _np
    from simulator import capacity_frontier
    from config import DEFAULT_EXECUTION
    rng = _np.random.default_rng(1)
    T, N = 250, 60
    R = 0.0008 + rng.normal(0, 0.01, (T, N))
    Wd = _np.zeros_like(R); Wd[:-1] = 0.05 * _np.sign(R[1:])
    adv = _np.full((T, N), 5e6)
    out = capacity_frontier(Wd, R, adv, dict(DEFAULT_EXECUTION), hurdle=1.0, n_grid=8, lo=1e5, hi=1e11)
    ss = [s for _, s, _ in out['curve']]
    assert ss[0] > ss[-1]  # superlinear costs erode net Sharpe with size
    assert out['capacity_notional'] > 1e5
    assert out['net_sharpe_at_capacity'] >= 1.0
    unmet = capacity_frontier(Wd, R, adv, dict(DEFAULT_EXECUTION), hurdle=999.0, n_grid=4, lo=1e5, hi=1e11)
    assert unmet['feasible'] is False
    assert unmet['capacity_notional'] is None
    assert unmet.get('note') == 'sampled feasible frontier; hurdle unmet on sampled grid'
    import pytest as _pt
    with _pt.raises(ValueError):
        capacity_frontier(Wd, R, adv, dict(DEFAULT_EXECUTION), lo=0)
    with _pt.raises(ValueError):
        capacity_frontier(Wd, R, adv, dict(DEFAULT_EXECUTION), min_fill_rate=0.0)
    with _pt.raises(ValueError):
        capacity_frontier(Wd, R, adv, dict(DEFAULT_EXECUTION), n_grid=1)
    with _pt.raises(ValueError):
        capacity_frontier(Wd, R, adv, dict(DEFAULT_EXECUTION), hurdle=float('inf'))
