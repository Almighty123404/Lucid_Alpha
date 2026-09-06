"""Data-generator contract tests: determinism, coverage, gate behavior.

Suite hygiene: all full-window tests share the session med_panel/big_panel
fixtures (one n=200 + one n=1000 build per session) instead of rebuilding
the DuckDB PIT grids per test.
"""
import numpy as np
import pytest

from data_gen import generate, validate_panel, LAST_PANIC_RATE


def test_determinism_same_seed_identical():
    a = generate(seed=11, n_stocks=50, start="2020-01-01", end="2020-06-30")
    b = generate(seed=11, n_stocks=50, start="2020-01-01", end="2020-06-30")
    assert (a.dates == b.dates).all()
    assert np.array_equal(a.fields["close"], b.fields["close"], equal_nan=True)


def test_sentiment_coverage_sparse(med_panel):
    cov = float(np.isfinite(med_panel.vector_fields["nws12_afterhsz_01l"][0]).mean())
    assert 0.01 <= cov <= 0.10  # real ~1-5%; old bug printed 47%


def test_iv_tenor_gradient(med_panel):
    c10 = float(np.isfinite(med_panel.fields["implied_volatility_call_10"]).mean())
    c60 = float(np.isfinite(med_panel.fields["implied_volatility_call_60"]).mean())
    assert c10 < c60  # 10d strictly sparser subset of 60d


def test_panic_gate_fires_in_2020_not_always(big_panel):
    # big_panel build itself refreshes LAST_PANIC_RATE for seed 11 / n=1000
    # (f_m path is N-dependent; this is the verified acceptance config).
    assert LAST_PANIC_RATE["overall"] is not None
    assert 0.0 < LAST_PANIC_RATE["overall"] < 0.35
    assert LAST_PANIC_RATE["y2020"] > 0.0


def test_leverage_sign_favors_unsigned(med_panel, med_refs):
    # Sign reconciliation (item 14a): unsigned debt/assets must beat negated.
    from simulator import simulate
    up = simulate("rank(ts_backfill(debt, 32) / ts_backfill(assets, 32))", med_panel, med_refs, (), settings=None)
    dn = simulate("-rank(ts_backfill(debt, 32) / ts_backfill(assets, 32))", med_panel, med_refs, (), settings=None)
    assert up["metrics"]["sharpe"] > dn["metrics"]["sharpe"]

def test_delistings_present_but_bounded(big_panel):
    cl = big_panel.fields["close"]
    dead = (~np.isfinite(cl)).sum(axis=0) > 0
    frac = float(dead.mean())
    assert 0.0 < frac < 0.15  # ~2%/yr hazard, not a massacre
    assert float(np.isfinite(cl).mean()) >= 0.95


def test_validate_panel_rejects_tampering():
    panel = generate(seed=11, n_stocks=50, start="2020-01-01", end="2020-06-30")
    validate_panel(panel)  # must pass on fresh output
    bad = generate(seed=11, n_stocks=50, start="2020-01-01", end="2020-06-30")
    bad.fields["close"][:] = np.nan
    with pytest.raises(ValueError):
        validate_panel(bad)

def test_l1_fat_tails_present(med_panel):
    r = med_panel.fields["returns"]
    m = np.isfinite(r)
    mu = np.where(m, r, 0.0).sum() / m.sum()
    sd = (np.where(m, (r - mu) ** 2, 0.0).sum() / m.sum()) ** 0.5
    zk = ((r - mu) / sd)[m]
    assert float((zk ** 4).mean() - 3.0) > 1.5  # t-mixer fat tails, Gaussian ~= 0


def test_l1_stress_years_elevated_vol(med_panel):
    # Dated episodes elevate vol above the i.i.d. baseline (~0.3); the
    # stochastic crisis chain may additionally elevate any year, so no
    # cross-year ordering is asserted (ordering would overfit one stream).
    r = med_panel.fields["returns"]
    yrs = med_panel.dates.astype("datetime64[Y]").astype(int) + 1970
    v2020 = float(r[yrs == 2020].std() * (252.0 ** 0.5))
    assert v2020 > 0.4


def test_pit_fundamentals_knowledge_bounded(med_panel):
    eb = med_panel.fields["ebitda"]
    assert float(np.isnan(eb[:63]).mean()) > 0.9  # nothing knowable at inception
    assert float(np.isnan(eb[200:]).mean()) < 0.05  # published thereafter

def test_rr21_tail_correlations_spike(med_panel):
    # RR-21: in liquidations correlations spike toward 1.0 (systematic
    # dominates). Magnitude-based stress: top-5pct |mkt| days must correlate
    # above the all-days baseline. (Signed downside-only slices are
    # dispersion-confounded; magnitude is the liquidation-relevant proxy.)
    r = med_panel.fields["returns"]
    m = np.isfinite(r)
    mkt = np.where(m, r, 0.0).sum(axis=1) / m.sum(axis=1)
    def meancorr(x):
        xf = np.where(np.isfinite(x), x, 0.0)
        c = np.corrcoef(xf[:, :50].T)
        iu = np.triu_indices(c.shape[0], 1)
        return float(np.nanmean(c[iu]))
    am = np.abs(mkt)
    assert meancorr(r[am >= np.nanquantile(am, 0.95)]) > meancorr(r)
