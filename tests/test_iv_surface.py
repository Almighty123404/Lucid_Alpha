"""Tests for iv_surface.py: SSVI no-arb asserts + forward variance + BL + masks."""
import numpy as np
import pytest

from iv_surface import (
    TENOR_DAYS, K_GRID, term_mult, regime_mult, ssvi_params,
    generate_iv_surface, assert_no_calendar_arb, assert_no_butterfly_arb,
    assert_ssvi_bounds, forward_variance, forward_vol,
    breeden_litzenberger_density, assert_coverage_nesting,
    garman_klass_estimate,
)


def _surf(T=30, N=20, seed=7):
    rng = np.random.default_rng(seed)
    iv_base = 20.0 + rng.normal(0, 2, (T, N))
    liq = rng.uniform(0, 1, N)
    skew0 = rng.normal(0.03, 0.05, N)
    crisis = np.zeros(T)
    crisis[10:15] = 1.0
    return generate_iv_surface(iv_base, liq, skew0, crisis)


def test_term_gradient_preserved():
    assert term_mult(10.0) < term_mult(60.0) < term_mult(720.0)
    assert abs(term_mult(720.0) - 1.08) < 1e-9


def test_regime_cap():
    assert regime_mult(1, 1, 1, 1) <= 1.9


def test_surface_passes_all_asserts():
    out, thetas = _surf()
    assert_no_calendar_arb(thetas)
    j60 = list(out.keys()).index(60.0)
    sample = np.stack([out[d][0, 0] for d in sorted(out)])
    assert_no_butterfly_arb(K_GRID, sample)
    rho, eta = ssvi_params(np.array([[0.5]]), np.array([[0.03]]), np.array([[0.0]]))
    assert_ssvi_bounds(thetas[:1, :1], rho, eta)


def test_forward_variance_calendar_breach_raises():
    with pytest.raises(ValueError, match="calendar arb"):
        forward_variance(0.30, 1.0, 0.10, 2.0)  # less total var at longer T
    wF = forward_variance(0.18, 0.25, 0.21, 0.50)
    assert abs(float(np.mean(wF)) - 0.01395) < 1e-6  # PDF worked example
    assert abs(float(np.mean(forward_vol(0.18, 0.25, 0.21, 0.50))) - 0.2362) < 1e-3


def test_put_call_skew_emerges():
    out, _ = _surf()
    s60 = out[60.0]
    # OTM-put wing (k<0) above OTM-call wing (k>0): equity skew, no ad-hoc mult
    assert float(np.nanmean(s60[:, :, 1] - s60[:, :, 3])) > 0


def test_bl_density_nonnegative_on_clean_grid():
    # Black-76 calls on a uniform strike grid under flat vol: convex, BL >= 0
    from math import erf, sqrt, log, exp
    F, T, sig = 100.0, 1.0, 0.25
    ks = np.linspace(60, 140, 41)

    def N(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    def call(K):
        d1 = (log(F / K) + 0.5 * sig ** 2 * T) / (sig * sqrt(T))
        d2 = d1 - sig * sqrt(T)
        return exp(0) * (F * N(d1) - K * N(d2))

    dens, mn = breeden_litzenberger_density(np.array([call(k) for k in ks]), ks)
    assert mn >= -1e-6, mn


def test_coverage_nesting():
    rng = np.random.default_rng(0)
    m60 = rng.random((10, 5)) < 0.5
    m10 = m60 & (rng.random((10, 5)) < 0.5)
    m720 = m60 & (rng.random((10, 5)) < 0.3)
    assert_coverage_nesting(m60, m10, m720)
    with pytest.raises(ValueError):
        assert_coverage_nesting(m60, ~m60 & True, m720)


def test_garman_klass_nonnegative():
    rng = np.random.default_rng(1)
    o = 100 + rng.normal(0, 1, 50)
    c = o + rng.normal(0, 0.5, 50)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 1, 50))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 1, 50))
    gk = garman_klass_estimate(o, h, l, c)
    assert bool(np.all(gk >= 0))
