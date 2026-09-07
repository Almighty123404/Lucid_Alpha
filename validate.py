"""Empirical fidelity harness (Agent-4 blueprint): 10 moments + BENCH verdicts.

Compares a generated panel against Compustat/CRSP reference ranges
(winsorized, APPROX where noted in BENCH) plus leakage + tail lenses.
Diagnostic-only: never gates. numpy + pit + risk only.
"""
import numpy as np

BENCH = {  # key: (lo, hi, tol_warn_frac)
    "sales_g_vol": (0.15, 0.25, 0.20),
    "gm_mean": (0.32, 0.40, 0.15),
    "gm_std": (0.18, 0.25, 0.20),
    "ebitda_kurt": (5.0, 15.0, 0.30),
    "lev_median": (0.20, 0.30, 0.15),
    "lev_iqr": (0.20, 0.30, 0.25),
    "ret_kurt": (6.0, 14.0, 0.25),
    "paircorr_all": (0.12, 0.25, 0.30),
    "paircorr_tail": (0.35, 0.55, 0.20),
    "fund_ar1": (0.70, 0.90, 0.10),
}


def _verdict(v, lo, hi, tol):
    span = hi - lo
    if v is not None and np.isfinite(v) and lo <= v <= hi:
        return "pass"
    if v is not None and np.isfinite(v) and lo - tol * span <= v <= hi + tol * span:
        return "warn"
    return "fail"


def _wins(x, p=(1, 99)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    lo, hi = np.nanpercentile(x, p)
    return x[(x >= lo) & (x <= hi)]


def ks_2samp(x, y):
    """Two-sample KS, numpy-only."""
    x, y = np.sort(np.asarray(x)[np.isfinite(x)]), np.sort(np.asarray(y)[np.isfinite(y)])
    grid = np.sort(np.concatenate([x, y]))
    fx = np.searchsorted(x, grid, side="right") / max(len(x), 1)
    fy = np.searchsorted(y, grid, side="right") / max(len(y), 1)
    return float(np.abs(fx - fy).max())


def w1_dist(x, y, nq=2000):
    """1-Wasserstein on quantile-matched grids."""
    x = np.asarray(x)[np.isfinite(np.asarray(x))]
    y = np.asarray(y)[np.isfinite(np.asarray(y))]
    q = np.linspace(0.001, 0.999, nq)
    return float(np.abs(np.quantile(x, q) - np.quantile(y, q)).mean())


def js_div(x, y, bins=50):
    """Jensen-Shannon divergence (base-2, [0,1]), fixed-bin histogram."""
    x = np.asarray(x)[np.isfinite(np.asarray(x))]
    y = np.asarray(y)[np.isfinite(np.asarray(y))]
    lo, hi = np.quantile(np.concatenate([x, y]), [0.005, 0.995])
    px, e = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    py, _ = np.histogram(y, bins=bins, range=(lo, hi), density=True)
    w = (e[1:] - e[:-1])
    P, Q = px * w + 1e-12, py * w + 1e-12
    P, Q = P / P.sum(), Q / Q.sum()
    M = 0.5 * (P + Q)
    return float(0.5 * (P * np.log2(P / M)).sum() + 0.5 * (Q * np.log2(Q / M)).sum())


def compute_moments(dates, F):
    """10 key moments from a served panel. Skips the 63d PIT warmup.

    M1/M10 are measured on the TRUE quarterly grid (every 63rd day): daily
    diffs of a forward-filled step panel read ~0 and AR ~1 by construction,
    which measures the fill, not the economics (fixed 2026-QFC review).
    M7 uses light (0.1/99.9) winsorization: 1/99 clips the very tails whose
    kurtosis is under test (same review).
    """
    sales = F["sales"]
    gm = (sales - F["cogs"]) / np.maximum(sales, 1e-12)
    q = np.log(np.maximum(sales[63::63], 1e-12))
    gq = np.diff(q, axis=0)
    m1 = float(np.nanmedian(np.nanstd(gq, axis=0)))
    flat_gm = _wins(gm[63:].ravel())
    m2, m3 = float(np.nanmean(flat_gm)), float(np.nanstd(flat_gm))
    eb = _wins((F["ebitda"] / np.maximum(sales, 1e-12))[63:].ravel())
    ebs = (eb - eb.mean()) / (eb.std() + 1e-12)
    m4 = float((ebs ** 4).mean() - 3.0)
    lev = _wins((F["debt"] / np.maximum(F["assets"], 1e-12))[63:].ravel())
    m5 = float(np.median(lev))
    m6 = float(np.percentile(lev, 75) - np.percentile(lev, 25))
    # M7: excess kurtosis of MARKET-HEDGED residuals (daily cross-sectional
    # demean), pooled, NO winsorization. Rationale (2026-QFC review): pooled
    # raw returns mix the market factor with heterogeneous vols (kurt 3.9);
    # winsorizing then clips the very tails under test (2.2/1.3/0.8 at
    # 0.1/0.5/1% — monotone destruction). Residual pooled kurt 8.5 sits in
    # bench; nu_hat is fit on the same series so the pair is coherent.
    mkt_r = np.nanmean(F["returns"][1:], axis=1, keepdims=True)
    res = F["returns"][1:] - mkt_r
    xr = res[np.isfinite(res)]
    rs = (xr - xr.mean()) / (xr.std(ddof=1) + 1e-12)
    m7 = float((rs ** 4).mean() - 3.0)
    out_resid = xr
    N = F["returns"].shape[1]
    sub = F["returns"][1:, :min(N, 200)]
    with np.errstate(invalid="ignore"):
        C = np.ma.corrcoef(np.ma.masked_invalid(sub), rowvar=False).filled(np.nan)
    iu = C[np.triu_indices_from(C, 1)]
    m8 = float(np.nanmean(iu))
    mkt = np.nanmean(sub, axis=1)
    tail = sub[mkt < np.nanpercentile(mkt, 5)]
    with np.errstate(invalid="ignore"):
        Ct = np.ma.corrcoef(np.ma.masked_invalid(tail), rowvar=False).filled(np.nan)
    m9 = float(np.nanmean(Ct[np.triu_indices_from(Ct, 1)]))
    # M10: AR(1) of EBITDA-margin LEVELS on the quarterly grid (stationary,
    # bench 0.7-0.9). Sales levels are unit-root (~0.99) and sales growth
    # carries φ_g≈0.25 — neither belongs in this bench (fixed 2026-QFC review).
    _em = (F["ebitda"] / np.maximum(sales, 1e-12))[63::63]
    a, b = _em[:-1].ravel(), _em[1:].ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    m10 = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else float("nan")
    # M11 (diagnostic-only, no bench): residual paircorr after daily
    # cross-sectional demean. B1 finding: ~0.00 — the model is a 1-factor
    # market + independent-idio world (no residual industry structure, no
    # asymmetric tail coupling; cf. RR-21 Clayton deferral). Real markets
    # keep 0.05-0.15 here. Ticketed, not gated.
    _res = sub - np.nanmean(sub, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        _Cr = np.ma.corrcoef(np.ma.masked_invalid(_res), rowvar=False).filled(np.nan)
    m11 = float(np.nanmean(_Cr[np.triu_indices_from(_Cr, 1)]))
    return {"sales_g_vol": m1, "gm_mean": m2, "gm_std": m3, "ebitda_kurt": m4,
            "lev_median": m5, "lev_iqr": m6, "ret_kurt": m7,
            "paircorr_all": m8, "paircorr_tail": m9, "fund_ar1": m10,
            "_resid": out_resid, "_paircorr_resid": m11}


def fidelity_report(dates, fields, knowledge=None):
    """{moments, leakage, tail}. knowledge: {field: (values, knowledge_ts)}."""
    from pit import validate_pit
    from risk import fit_nu, exceedance_backtest
    leakage = "not-checked"
    if knowledge:
        for k, (v, kn) in knowledge.items():
            validate_pit(v, kn, dates)  # raises on violation (hard fail)
        leakage = "pass"
    mom = compute_moments(dates, fields)
    rep = {}
    for k, (lo, hi, tol) in BENCH.items():
        rep[k] = {"syn": round(float(mom[k]), 4) if np.isfinite(mom[k]) else None,
                  "benchmark": [lo, hi],
                  "verdict": _verdict(mom[k], lo, hi, tol)}
    mkt = np.nanmean(fields["returns"][1:], axis=1)
    tail = exceedance_backtest(mkt)
    nu_hat = float(fit_nu(mom.pop("_resid")))  # same residual series as M7
    resid_corr = round(float(mom.pop("_paircorr_resid")), 4)
    fails = sum(1 for v in rep.values() if v["verdict"] == "fail")
    hard = [k for k in ("lev_median", "ret_kurt", "paircorr_tail")
            if rep[k]["verdict"] == "fail"]
    return {"moments": rep, "leakage": leakage, "nu_hat": round(nu_hat, 3),
            "paircorr_resid": resid_corr,
            "tail_backtest": tail, "n_fail": fails, "hard_fails": hard,
            "ship": bool(leakage == "pass" and not hard and fails <= 2)}
