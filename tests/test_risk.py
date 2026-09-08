"""Risk lens tests: textbook values, calibration behavior, integration."""
import numpy as np

from risk import (var_es_normal, var_es_t, var_es_historical, risk_report,
                  conditional_risk, exceedance_backtest, _t_ppf)


def test_normal_closed_form_matches_textbook():
    v, e = var_es_normal(0.0, 1.0, 0.99)
    assert abs(v - 2.3263) < 1e-3
    assert abs(e - 2.6652) < 1e-3


def test_t_quantiles_match_textbook():
    assert abs(_t_ppf(0.99, 6) - 3.1427) < 0.01  # t(6) 99% (Remark 1: nu~5-6)
    assert abs(_t_ppf(0.975, 6) - 2.4469) < 0.01


def test_t_tail_heavier_than_normal():
    vn, en = var_es_normal(0.0, 1.0, 0.99)
    vt, et = var_es_t(0.0, 1.0, 6, 0.99)
    assert vt > vn and et > en  # heavy tails: worse quantile AND worse shortfall
    assert et / en > 1.0


def test_historical_on_seeded_normal():
    rng = np.random.default_rng(0)
    pnl = np.concatenate([[0.0], rng.normal(0.001, 0.01, 5000)])
    out = var_es_historical(pnl)
    assert abs(out[0.95]["var"] - (0.001 * -1 + 0.01 * 1.6449)) < 0.004
    assert out[0.99]["es"] >= out[0.99]["var"]  # ES >= VaR by construction


def test_exceedance_calibrated_on_normal():
    rng = np.random.default_rng(1)
    pnl = np.concatenate([[0.0], rng.normal(0.0, 0.01, 5000)])
    r = exceedance_backtest(pnl, alpha=0.95)
    assert r["verdict"] == "calibrated"
    assert r["n"] == 5000


def test_conditional_shapes_and_warmup():
    rng = np.random.default_rng(2)
    pnl = np.concatenate([[0.0], rng.normal(0.0, 0.01, 400)])
    c = conditional_risk(pnl, window=252, alpha=0.95)
    assert c["hist_var"].shape == pnl.shape
    assert bool(np.isnan(c["hist_var"][:252]).all())  # warmup NaN
    assert bool(np.isfinite(c["hist_var"][252:]).all())
    assert bool((c["t_es"][252:] >= c["hist_var"][252:] - 1e-12).all())  # ES>=VaR pathwise


def test_integration_report_keys(panel, refs):
    from simulator import simulate
    rep = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (), settings=None)
    rr = rep["risk"]
    assert rr["n_days"] > 100
    assert rr["hist"]["0.95"]["es"] >= rr["hist"]["0.95"]["var"]
    assert rr["es_gap_t_vs_normal_99"] > 0  # t tails heavier, always
    assert rep["criteria"]["sharpe"]["pass"] in (True, False)  # gates untouched by lens

def test_fit_nu_kurtosis_mapping():
    from risk import fit_nu
    rng = np.random.default_rng(0)
    # Gaussian sample: thin tail -> high df (t -> normal limit)
    assert fit_nu(rng.normal(0, 1, 2000)) >= 10.0
    assert fit_nu(rng.normal(0, 1, 50)) == 6.0  # n < 100 fallback
    # heavy mixture (~kurt 6-7): mid single-digit df
    x = np.concatenate([rng.normal(0, 1, 1900), rng.normal(0, 5, 100)])
    assert 4.0 <= fit_nu(x) <= 8.0


def test_risk_report_fits_nu_by_default(panel, refs):
    from simulator import simulate
    rep = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (), settings=None)
    assert 4.0 <= rep["risk"]["nu"] <= 30.0


def test_trade_lens_textbook_and_guards():
    import numpy as _np
    from risk import trade_lens
    rng = _np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 1000)
    r = trade_lens(x)
    mu, dsd = x.mean(), x[x < 0].std(ddof=1)
    assert abs(r['sortino'] - round(float(mu / dsd * _np.sqrt(252.0)), 4)) < 1e-9
    assert r['hit_rate'] == round(float((x > 0).mean()), 4)
    assert r['max_drawdown'] >= 0.0
    g = trade_lens(_np.ones(50))
    assert g['sortino'] == 0.0 and g['hit_rate'] == 1.0
    assert trade_lens(_np.zeros(40))['sortino'] == 0.0
    assert len(trade_lens(_np.ones(10))) == 1  # n<30 -> n_days only

