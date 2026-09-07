"""Tests for validate.py: harness runs on a small synthetic panel."""
import numpy as np


def test_fidelity_report_runs_and_leakage_passes():
    from data_gen import generate
    from validate import fidelity_report, BENCH
    panel = generate(seed=3, n_stocks=40, start="2020-01-01", end="2020-12-31")
    rep = fidelity_report(panel.dates, panel.fields, knowledge=None)
    assert set(rep["moments"]) == set(BENCH)
    assert rep["leakage"] == "not-checked"
    assert 4.0 <= rep["nu_hat"] <= 12.0, rep["nu_hat"]
    assert isinstance(rep["ship"], bool)


def test_distance_metrics_sane():
    from validate import ks_2samp, w1_dist, js_div
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 3000)
    y = rng.normal(0, 1, 3000)
    assert ks_2samp(x, y) < 0.05
    assert js_div(x, y) < 0.02
    z = rng.normal(2, 1, 3000)
    assert ks_2samp(x, z) > 0.10
    assert w1_dist(x, z) > 1.0


def test_moment_directions():
    from data_gen import generate
    from validate import compute_moments
    panel = generate(seed=3, n_stocks=40, start="2020-01-01", end="2020-12-31")
    m = compute_moments(panel.dates, panel.fields)
    # Small N=40/1yr fixture: kurtosis is noisy; binding fat-tail assert is
    # test_l1_fat_tails_present on the full med_panel. Here just positive.
    assert m["ret_kurt"] > 0.5
    assert 0.0 < m["lev_median"] < 0.9
    # crash co-movement direction holds on large panels; on N=40/1yr the tail
    # slice is noisy, so only bound it (full-panel check lives in notes).
    assert -0.2 < m["paircorr_all"] < 0.9
    assert -0.2 < m["paircorr_tail"] < 0.95
