"""Tests for fundamentals.py (Agent-2 blueprint §5 falsifiers, unwired module).

Falsifiers: margin persistence phi ranges, CFO-EBITDA coupling rho,
wedge variance, size-variance psi, C1-C4 identity audit (hard fail).
"""
import numpy as np

from fundamentals import (
    FUND_DEFAULTS, generate_fundamentals_q, apply_equity_floor,
    check_accounting_identities,
)


def _panel(seed=7, N=200, Q=13):
    rng = np.random.default_rng(seed)
    size = np.exp(rng.normal(10.0, 1.0, N))
    f_mbar = rng.normal(0.0, 0.01, Q)
    return generate_fundamentals_q(rng, size, Q, f_mbar), size


def test_identities_hold_exactly():
    out, _ = _panel()
    Q, N = out["sales_q"].shape
    rng = np.random.default_rng(0)
    curr = out["assets_q"] * rng.uniform(0.08, 0.3, N)[None, :]
    debt_adj, equity, _ = apply_equity_floor(out["assets_q"], out["debt_q"], curr)
    liab = debt_adj + curr
    aud = check_accounting_identities(out["sales_q"], out["assets_q"], debt_adj,
                                      out["ebitda_q"], out["cfo_q"], liab, equity)
    assert aud["c2_identity"] == 0.0, aud
    assert aud["c4_cashflow_bound"] == 0.0, aud
    assert aud["c1_margin_ebitda"] == 0.0, aud
    assert aud["c1_margin_cfo"] == 0.0, aud
    assert aud["c3_equity_floor"] == 0.0, aud


def test_margin_persistence_in_range():
    out, _ = _panel(N=400, Q=13)
    for key, lo, hi in (("ebitda_q", 0.6, 0.9), ("cfo_q", 0.35, 0.7)):
        m = out[key] / np.maximum(out["sales_q"], 1e-12)
        a, b = m[:-1].ravel(), m[1:].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        phi = float(np.corrcoef(a[ok], b[ok])[0, 1])
        assert lo <= phi <= hi, (key, phi)


def test_cfo_ebitda_innovation_coupling():
    out, _ = _panel(N=400, Q=13)
    me = out["ebitda_q"] / np.maximum(out["sales_q"], 1e-12)
    mc = out["cfo_q"] / np.maximum(out["sales_q"], 1e-12)
    # innovation proxy: quarterly changes (levels are persistent by design)
    de, dc = np.diff(me, axis=0).ravel(), np.diff(mc, axis=0).ravel()
    ok = np.isfinite(de) & np.isfinite(dc)
    rho = float(np.corrcoef(de[ok], dc[ok])[0, 1])
    assert 0.4 <= rho <= 0.85, rho


def test_wedge_variance_bounded():
    out, _ = _panel()
    w = (out["ebitda_q"] - out["cfo_q"]) / np.maximum(out["sales_q"], 1e-12)
    v = float(np.nanvar(w))
    assert 0.002 <= v <= 0.02, v


def test_size_variance_gradient():
    rng = np.random.default_rng(7)
    N = 600
    size = np.exp(rng.normal(10.0, 1.0, N))
    f_mbar = np.zeros(13)
    out = generate_fundamentals_q(np.random.default_rng(11), size, 13, f_mbar)
    g = np.diff(np.log(np.maximum(out["sales_q"], 1e-12)), axis=0)
    v = np.nanvar(g, axis=0)
    ls = np.log(size)
    ok = np.isfinite(v) & np.isfinite(ls)
    slope = float(np.polyfit(ls[ok], np.log(v[ok] + 1e-12), 1)[0])
    psi_hat = -slope / 2.0
    assert 0.05 <= psi_hat <= 0.4, psi_hat


def test_determinism():
    a, _ = _panel()
    b, _ = _panel()
    assert np.array_equal(a["sales_q"], b["sales_q"])
    assert np.array_equal(a["cfo_q"], b["cfo_q"])
