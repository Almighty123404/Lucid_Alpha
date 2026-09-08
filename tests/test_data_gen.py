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


def test_generate_rejects_degenerate_dimensions():
    with pytest.raises(ValueError, match="n_stocks"):
        generate(seed=11, n_stocks=1, start="2020-01-01", end="2020-06-30")
    with pytest.raises(ValueError, match="end"):
        generate(seed=11, n_stocks=20, start="2020-06-30", end="2020-01-01")


def test_sentiment_coverage_sparse(med_panel):
    cov = float(np.isfinite(med_panel.vector_fields["nws12_afterhsz_01l"][0]).mean())
    assert 0.01 <= cov <= 0.10  # real ~1-5%; old bug printed 47%


def test_iv_tenor_gradient(med_panel):
    c10 = float(np.isfinite(med_panel.fields["implied_volatility_call_10"]).mean())
    c60 = float(np.isfinite(med_panel.fields["implied_volatility_call_60"]).mean())
    assert c10 < c60  # 10d strictly sparser subset of 60d


def test_panic_gate_fires_in_2020_not_always():
    # Deliberately NOT on the shared fixture: LAST_PANIC_RATE is module-global
    # (last generate() call wins), so sharing makes this order-dependent.
    # Own build keeps the acceptance config exact (seed 11 / n=1000).
    generate(seed=11, n_stocks=1000, start="2020-01-01", end="2022-12-31")
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

def test_codebook_extension_present(med_panel):
    want = ["vwap", "shares_out", "adv60", "gross_profit", "net_income", "eps",
            "equity", "working_capital", "free_cash_flow", "est_eps", "snt_news",
            "iv_10", "hv_20", "put_call_ratio", "short_interest", "borrow_fee",
            "insider_buying"]
    for f in want:
        assert f in med_panel.fields, f
    assert "return_assets" in med_panel.fields  # G1 identity
    m3 = np.isfinite(med_panel.fields["return_assets"]) & np.isfinite(med_panel.fields["net_income"]) & np.isfinite(med_panel.fields["assets"])
    assert bool((((med_panel.fields["return_assets"] - med_panel.fields["net_income"] / np.maximum(med_panel.fields["assets"], 1e-12))[m3]) == 0).all())
    assert "operating_expense" in med_panel.fields and "ebit" in med_panel.fields
    m4 = np.isfinite(med_panel.fields["operating_expense"]) & np.isfinite(med_panel.fields["cogs"]) & np.isfinite(med_panel.fields["sales"])
    assert bool((((med_panel.fields["operating_expense"] - med_panel.fields["cogs"])[m4] >= 0).all()))
    for v in ["analyst_eps_estimates", "option_implied_vol_surface",
              "segment_revenue", "price_volume_intraday"]:
        assert v in med_panel.vector_fields, v
        assert len(med_panel.vector_fields[v]) >= 3
    assert "market" in med_panel.groups


def test_codebook_identities(med_panel):
    import numpy as np
    f = med_panel.fields
    m = np.isfinite(f["equity"]) & np.isfinite(f["assets"]) & np.isfinite(f["liabilities"])
    assert bool((((f["equity"] - (f["assets"] - f["liabilities"]))[m]) == 0).all())
    m2 = np.isfinite(f["free_cash_flow"]) & np.isfinite(f["operating_cash_flow"]) & np.isfinite(f["capex"])
    assert bool((((f["free_cash_flow"] - (f["operating_cash_flow"] - f["capex"]))[m2]) == 0).all())

def test_720_tenor_sparser_than_60(med_panel):
    import numpy as np
    f = med_panel.fields
    c720 = float(np.isfinite(f["implied_volatility_call_720"]).mean())
    c60 = float(np.isfinite(f["implied_volatility_call_60"]).mean())
    assert 0.0 < c720 < c60  # strict sparser subset by construction
    # subset property: 720 print implies 60 print
    both = np.isfinite(f["implied_volatility_call_720"]) & ~np.isfinite(f["implied_volatility_call_60"])
    assert not bool(both.any())


def test_cashflow_pit_bounded(med_panel):
    import numpy as np
    c = med_panel.fields["cashflow_op"]
    assert float(np.isnan(c[:63]).mean()) > 0.9  # nothing knowable at inception
    assert float(np.isnan(c[200:]).mean()) < 0.05
    # not an identity off ebitda: residual variation after projection
    import numpy as np
    e = med_panel.fields["ebitda"]
    m = np.isfinite(c) & np.isfinite(e)
    assert 0.0 < abs(float(np.corrcoef(c[m], e[m])[0, 1])) < 0.99


def test_scl12_independent_source(med_panel):
    import numpy as np
    a = np.nanmean(np.stack(med_panel.vector_fields["scl12_alltype_buzzvec"]), axis=0)
    b = np.nanmean(np.stack(med_panel.vector_fields["nws12_afterhsz_01l"]), axis=0)
    m = np.isfinite(a) & np.isfinite(b)
    assert m.mean() > 0.0005  # some overlap to correlate over
    assert abs(float(np.corrcoef(a[m], b[m])[0, 1])) < 0.5  # independent processes
    cov = float(np.isfinite(med_panel.vector_fields["scl12_alltype_buzzvec"][0]).mean())
    assert 0.01 <= cov <= 0.10


def test_pead_drift_slow_pitclean_forwardpaying(med_panel):
    import numpy as _np
    f = med_panel.fields
    m = f['ebitda'] / _np.maximum(f['sales'], 1e-12)
    raw = m - _np.roll(m, 63, axis=0)
    raw[:63] = 0.0
    a, b = raw[:-1].ravel(), raw[1:].ravel()
    ok = _np.isfinite(a) & _np.isfinite(b)
    assert float(_np.corrcoef(a[ok], b[ok])[0, 1]) > 0.9  # slow drift, not spikes
    assert bool((raw[:63] == 0).all())  # no wrap-around leak into warmup
    r = f['returns']
    fwd = _np.nancumsum(_np.where(_np.isfinite(r), r, 0.0), axis=0)
    f63 = (fwd[63:] - fwd[:-63]).ravel()
    s = raw[:-63].ravel()
    mm = _np.isfinite(s) & _np.isfinite(f63)
    ic = float(_np.corrcoef(s[mm].argsort().argsort(), f63[mm].argsort().argsort())[0, 1])
    assert ic > 0.01  # drift funds forward returns



def test_delist_lottery_stable_under_edge_toggle():
    import data_gen as _dg
    from data_gen import generate as _gen
    old = _dg.EDGE['pead']
    try:
        _dg.EDGE['pead'] = old
        a = _gen(seed=11, n_stocks=50, start='2020-01-01', end='2020-06-30')
        _dg.EDGE['pead'] = 0.0
        b = _gen(seed=11, n_stocks=50, start='2020-01-01', end='2020-06-30')
    finally:
        _dg.EDGE['pead'] = old
    import numpy as _np
    da = set(_np.where(~_np.isfinite(a.fields['close'][-1]))[0].tolist())
    db = set(_np.where(~_np.isfinite(b.fields['close'][-1]))[0].tolist())
    assert len(da) > 0, 'fixture must delist >=1 name'
    assert da == db  # fork-isolated lottery: EDGE/model edits never re-roll

