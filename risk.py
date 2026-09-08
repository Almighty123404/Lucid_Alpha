"""Risk lens (Haugh notes Ch.2 McNeil/Frey/Embrechts): VaR/ES reporting.

DIAGNOSTIC ONLY — never enters gates (Brain speaks Sharpe/Fitness).
Conventions follow the notes exactly:
- Loss L is POSITIVE for losses: L = -pnl (our pnl>0 means profit).
- VaR_a = q_a(L), the a-quantile of the loss distribution (Def. 4).
- ES_a = mean loss beyond VaR (Lemma 2, continuous case).
- Historical simulation (§3.1) reads quantiles off realized daily losses
  (unconditional; inaccurate in stress — documented, hence conditional below).
- Parametric normal (eq.12) and t (eq.13, nu=6 per Remark 1: nu~5-6 fits
  stock returns best). The normal-vs-t ES gap IS the light-vs-heavy-tail
  correction our Sharpe framing omits.
- Conditional trailing-window VaR/ES = risk given F_t (notes §1.2: the right
  object for short horizons, especially in stress).
- Exceedance backtest (§3.4): exceedance count vs Binomial(n, 1-a).
"""
import math
from functools import lru_cache

import numpy as np

NORM_PPF_95 = 1.6448536269514722
NORM_PPF_99 = 2.3263478740408408


def _losses(pnl_daily):
    x = np.asarray(pnl_daily[1:], dtype=np.float64)
    x = x[np.isfinite(x)]
    return -x  # loss positive


def var_es_historical(pnl_daily, alphas=(0.95, 0.99)):
    """Empirical quantiles + tail means of daily losses. No distribution."""
    L = _losses(pnl_daily)
    out = {}
    if len(L) < 30:
        return {a: {"var": float("nan"), "es": float("nan")} for a in alphas}
    s = np.sort(L)
    n = len(s)
    for a in alphas:
        k = min(int(math.ceil(a * n)) - 1, n - 1)  #Def.3 inf{x:F>=a}
        k = max(k, 0)
        var = float(s[k])
        tail = s[k:]
        out[a] = {"var": var, "es": float(tail.mean())}
    return out


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_ppf(p):
    # Acklam approximation (matches fastexpr._norm_ppf family; fine here).
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    if 0.02425 <= p <= 1 - 0.02425:
        q = p - 0.5
        r = q * q
        num = ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        return q * num / den
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    x = num / den
    return x if p < 0.02425 else -x


def _t_pdf(x, nu):
    c = math.gamma((nu + 1.0) / 2.0) / (math.sqrt(nu * math.pi) * math.gamma(nu / 2.0))
    return c * (1.0 + x * x / nu) ** (-(nu + 1.0) / 2.0)


def _t_cdf(x, nu):
    # Simpson integration of the pdf from -B to x (B far in the tail).
    if x <= -12.0:
        return 0.0
    if x >= 12.0:
        return 1.0
    n = 2000
    a, b = -12.0, x
    h = (b - a) / n
    s = _t_pdf(a, nu) + _t_pdf(b, nu)
    for i in range(1, n):
        s += (4.0 if i % 2 else 2.0) * _t_pdf(a + i * h, nu)
    return s * h / 3.0


@lru_cache(maxsize=8)
def _t_ppf(p, nu):
    # Cached: only (0.95, 6) and (0.99, 6) are ever queried in production,
    # so the Simpson+bisection cost is paid once per process, not per sim.
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def var_es_normal(mu, sigma, alpha):
    """Closed form, notes eq.12. mu/sigma of LOSS (daily, decimal)."""
    q = _norm_ppf(alpha)
    return mu + sigma * q, mu + sigma * _norm_pdf(q) / (1.0 - alpha)


def var_es_t(mu, sigma, nu, alpha):
    """Closed form, notes eq.13 (standardized t, scaled)."""
    q = _t_ppf(alpha, nu)
    es_std = _t_pdf(q, nu) / (1.0 - alpha) * ((nu + q * q) / (nu - 1.0))
    return mu + sigma * q, mu + sigma * es_std


def fit_nu(losses, lo=4.0, hi=30.0):
    """Per-alpha t degrees-of-freedom via kurtosis matching (item 3).

    Standardized-t kurtosis is 3(nu-2)/(nu-4); inverting gives
    nu = (4k-6)/(k-3) for sample kurtosis k. Clamped to [lo, hi]. Falls back
    to 6.0 (Remark 1) when n < 100 or degenerate; k <= 3 (Gaussian-or-thinner
    tail) maps to the hi cap since t -> normal as nu -> inf (the inversion is
    singular at k~=3, so it is never evaluated there). Approximate
    (method-of-moments, not MLE) — documented, lens-grade.
    """
    x = np.asarray(losses, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return 6.0
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    if sd < 1e-12:
        return 6.0
    k = float((((x - mu) / sd) ** 4).mean())
    if k <= 3.0:
        # Gaussian-or-thinner tail: t tends to normal as nu -> inf; the
        # 30.0 cap is that limit for reporting. (Near k~=3 the inversion is
        # singular, so do NOT evaluate the formula there — fall through to
        # the cap only via the k<=3 branch or the clamp below.)
        return 30.0
    nu = (4.0 * k - 6.0) / (k - 3.0)
    return round(min(max(nu, lo), hi), 2)


def risk_report(pnl_daily, nu=None):
    """Full lens for one alpha's daily PnL series. All losses positive.

    nu=None (default) fits per-alpha df via fit_nu(); pass an explicit nu
    (e.g. 6.0) to reproduce the Remark-1 reference. The normal-vs-t ES gap
    uses whichever nu is reported in the output.
    """
    L = _losses(pnl_daily)
    out = {"n_days": int(len(L))}
    if len(L) < 30:
        return out
    if nu is None:
        nu = fit_nu(L)
    mu = float(L.mean())
    sd = float(L.std(ddof=1)) if len(L) > 1 else 0.0
    out["hist"] = {str(a): v for a, v in var_es_historical(pnl_daily).items()}
    out["normal"] = {}
    out["t"] = {}
    for a in (0.95, 0.99):
        vn, en = var_es_normal(mu, sd, a)
        vt, et = var_es_t(mu, sd, nu, a)
        out["normal"][str(a)] = {"var": vn, "es": en}
        out["t"][str(a)] = {"var": vt, "es": et}
    out["es_gap_t_vs_normal_99"] = out["t"]["0.99"]["es"] - out["normal"]["0.99"]["es"]
    out["nu"] = nu
    return out


def trade_lens(pnl_daily, annualization=252.0):
    """Firm-style trade lens (IIT competition backtester convention), diagnostic-only.

    Sortino (mean/downside-std, annualized), Calmar (CAGR/max-drawdown) and
    daily hit rate on one alpha's daily book-PnL series. Mirrors the
    round-trip report fields (Sortino/Calmar/WinRate) so sim output reads
    side-by-side with firm-style reports — with one documented difference:
    ours are computed on daily BOOK PnL, theirs on per-trade round trips, so
    compare direction/rank, never levels. Never enters gates.
    """
    x = np.asarray(pnl_daily, dtype=np.float64)
    x = x[np.isfinite(x)]
    out = {"n_days": int(len(x))}
    if len(x) < 30:
        return out
    mu = float(x.mean())
    down = x[x < 0]
    dsd = float(down.std(ddof=1)) if len(down) > 1 else 0.0
    out["sortino"] = round(float(mu / dsd * math.sqrt(annualization)), 4) if dsd > 1e-12 else 0.0
    curve = np.cumsum(x)
    peak = np.maximum.accumulate(curve)
    dd = float((peak - curve).max())
    out["max_drawdown"] = round(dd, 4)
    cagr = float(curve[-1] / max(len(x) / annualization, 1e-12))
    out["calmar"] = round(float(cagr / dd), 4) if dd > 1e-12 else float("nan")
    out["hit_rate"] = round(float((x > 0).mean()), 4)
    return out


def conditional_risk(pnl_daily, window=252, alpha=0.95, nu=6):
    """Trailing-window VaR/ES series (risk given F_t, notes §1.2).

    Returns dict of arrays (NaN before window fills): hist_var, hist_es,
    t_es. The conditional t-ES path is the stress lens: it moves with
    trailing volatility where full-sample statistics sit still.
    """
    x = np.asarray(pnl_daily, dtype=np.float64)
    T = len(x)
    hv = np.full(T, np.nan)
    he = np.full(T, np.nan)
    te = np.full(T, np.nan)
    for t in range(window, T):
        L = -x[max(1, t - window + 1):t + 1]
        L = L[np.isfinite(L)]
        if len(L) < 30:
            continue
        s = np.sort(L)
        k = min(int(math.ceil(alpha * len(s))) - 1, len(s) - 1)
        hv[t] = s[max(k, 0)]
        he[t] = s[max(k, 0):].mean()
        mu, sd = float(L.mean()), float(L.std(ddof=1))
        if sd > 1e-12:
            _, et = var_es_t(mu, sd, nu, alpha)
            te[t] = et
    return {"hist_var": hv, "hist_es": he, "t_es": te,
            "window": window, "alpha": alpha, "nu": nu}


def exceedance_backtest(pnl_daily, var_series=None, alpha=0.95):
    """Notes §3.4: exceedance count vs Binomial(n, 1-a).

    var_series: trailing VaR path (from conditional_risk) aligned to pnl;
    defaults to the constant historical VaR (weaker test, unconditional).
    Returns counts, expected rate, two-sided binomial p-value, verdict.
    """
    L = _losses(pnl_daily)
    n = len(L)
    if n < 30:
        return {"n": n, "exceed": 0, "expected_rate": 1.0 - alpha,
                "p_value": float("nan"), "verdict": "insufficient data"}
    if var_series is None:
        v = float(np.sort(L)[min(int(math.ceil(alpha * n)) - 1, n - 1)])
        exceed = int((L >= v).sum())
    else:
        v = np.asarray(var_series[1:1 + n], dtype=np.float64)
        ok = np.isfinite(v)
        L, v = L[ok], v[ok]
        n = len(L)
        exceed = int((L >= v).sum())
    p = 1.0 - alpha
    # Two-sided exact binomial tail in log space (math.comb overflows float
    # for n in the thousands; lgamma path is exact to ~1e-12 here).
    lp, lq = math.log(p), math.log(1.0 - p)
    dev = abs(exceed - n * p) - 1e-12
    pv = 0.0
    for k in range(n + 1):
        if abs(k - n * p) >= dev:
            pv += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1)
                           - math.lgamma(n - k + 1) + k * lp + (n - k) * lq)
    pv = min(max(pv, 0.0), 1.0)
    return {"n": n, "exceed": exceed, "expected": round(n * p, 2),
            "expected_rate": p, "p_value": round(pv, 4),
            "verdict": "calibrated" if pv >= 0.05 else "MISCALIBRATED"}
