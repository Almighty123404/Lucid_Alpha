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
