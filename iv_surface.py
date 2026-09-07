"""SSVI arbitrage-free IV surface engine (Agent-3 blueprint + PDF gap-fills).

Gatheral-Jacquier power-law SSVI:
  w(k,theta) = theta/2 * (1 + rho*phi*k + sqrt((phi*k+rho)^2 + 1-rho^2))
  phi(theta) = eta / theta^gamma, gamma=0.5

PDF deltas incorporated:
  - Forward variance: w(T)=sig^2*T additive; wF=w2-w1 (Forward Volatility.pdf);
    calendar assert is exactly wF >= 0.
  - Breeden-Litzenberger: f(K)=e^{rT}*d2C/dK2 >= 0 (Breeden-Litzenberger Eq.pdf);
    discrete call-price convexity check on the strike grid.
  - Empirical ingest: bid-ask midpoint, forward log-moneyness k=ln(K/F)
    (Volatility Surface Modelling.pdf) — documented in mask/spec functions.

Pure NumPy. No simulator imports. Noise is post-surface by contract; asserts
re-check after noise (caller halves noise or NaNs the print on breach).
"""
import numpy as np

TENOR_DAYS = np.array([10.0, 30.0, 60.0, 720.0])
TENOR_Y = TENOR_DAYS / 365.0
K_GRID = np.array([-0.35, -0.15, 0.0, 0.15, 0.35])
GAMMA = 0.5


def term_mult(T_d):
    """Term multiplier: 10d .928 / 30d .972 / 60d 1.0 / 720d 1.08 (kept)."""
    return np.clip(1.0 + 0.08 * np.log(np.asarray(T_d) / 60.0) / np.log(720.0 / 60.0),
                   0.85, 1.15)


def regime_mult(crisis=0.0, panic=0.0, stress=0.0, bear=0.0):
    """Crisis linkage, capped at 1.9 (SSVI bound headroom)."""
    m = ((1.35 if crisis else 1.0) * (1.25 if panic else 1.0)
         * (1.20 if stress else 1.0) * (1.10 if bear else 1.0))
    return min(m, 1.9)


def ssvi_params(liq, skew0, crisis):
    """rho in (-0.85, 0]; eta <= 0.75. Broadcastable over (T,N)."""
    rho = np.clip(-np.tanh(2.5 * np.asarray(skew0) + 0.35 + 0.10 * np.asarray(crisis)),
                  -0.85, 0.0)
    eta = np.minimum(0.35 + 0.25 * (1.0 - np.asarray(liq)) + 0.15 * np.asarray(crisis),
                     0.75)
    return rho, eta


def ssvi_total_var(k, theta, rho, eta, gamma=GAMMA):
    phi = np.asarray(eta) / np.maximum(np.asarray(theta), 1e-8) ** gamma
    s = phi * k + rho
    return 0.5 * np.asarray(theta) * (1.0 + rho * phi * k + np.sqrt(s * s + (1.0 - rho ** 2)))


def forward_variance(sig1, T1, sig2, T2):
    """Forward variance block wF=w2-w1; raises on calendar breach (wF<0)."""
    w1 = np.asarray(sig1) ** 2 * T1
    w2 = np.asarray(sig2) ** 2 * T2
    wF = w2 - w1
    if np.nanmin(wF) < -1e-8:
        raise ValueError(f"calendar arb: min forward variance {np.nanmin(wF)} < 0")
    return wF


def forward_vol(sig1, T1, sig2, T2):
    """Forward vol over (T1,T2]: sqrt(wF/(T2-T1))."""
    return np.sqrt(np.maximum(forward_variance(sig1, T1, sig2, T2), 0.0) / (T2 - T1))


def generate_iv_surface(iv_base, liq, skew0, crisis, tenors_d=TENOR_DAYS, k_grid=K_GRID):
    """SSVI surface from ATM anchor. iv_base:(T,N) vol-pts; liq/skew0:(N,);
    crisis:(T,) {0,1}. Returns {tenor_d: (T,N,K) vol-pts}, thetas (T,N,J)."""
    iv_base = np.asarray(iv_base, dtype=float)
    T_, N = iv_base.shape
    liq = np.asarray(liq, dtype=float)
    skew0 = np.asarray(skew0, dtype=float)
    crisis = np.asarray(crisis, dtype=float)
    J, K = len(tenors_d), len(k_grid)
    Ty = np.asarray(tenors_d, dtype=float) / 365.0
    out, thetas = {}, np.empty((T_, N, J))
    for j, Td in enumerate(tenors_d):
        sig_atm = (iv_base / 100.0) * term_mult(float(Td))
        theta = np.maximum(sig_atm ** 2 * Ty[j], 1e-8)
        thetas[:, :, j] = theta
        surf = np.empty((T_, N, K))
        for a, k in enumerate(k_grid):
            rho, eta = ssvi_params(liq[None, :], skew0[None, :], crisis[:, None])
            w = ssvi_total_var(k, theta, rho, eta)
            surf[:, :, a] = np.sqrt(w / Ty[j]) * 100.0
        out[float(Td)] = surf
    return out, thetas


def assert_no_calendar_arb(thetas, tenors_d=TENOR_DAYS):
    """Total variance non-decreasing in T (forward variance >= 0)."""
    o = np.argsort(np.asarray(tenors_d))
    d = np.diff(np.asarray(thetas)[:, :, o], axis=2)
    if np.nanmin(d) < -1e-8:
        raise ValueError(f"calendar arb: min d_theta={np.nanmin(d)}")


def _durrleman_g(k, w, dw, d2w):
    w = np.maximum(w, 1e-12)
    return (1.0 - k * dw / (2.0 * w)) ** 2 - (dw ** 2 / 4.0) * (1.0 / w + 0.25) + d2w / 2.0


def assert_no_butterfly_arb(k_grid, surf_volpts, tenors_y=TENOR_Y, nk_fine=200):
    """Durrleman g>=0 + Lee slope<=2 + Fukasawa d1/d2 monotone, per tenor."""
    surf_volpts = np.asarray(surf_volpts, dtype=float)
    for j, Ty in enumerate(tenors_y):
        v = surf_volpts[j] if surf_volpts.ndim == 2 else surf_volpts[0, j]
        kf = np.linspace(k_grid[0], k_grid[-1], nk_fine)
        vv = np.interp(kf, np.asarray(k_grid), v)
        w = (vv / 100.0) ** 2 * float(Ty)
        h = kf[1] - kf[0]
        dw = np.gradient(w, h)
        d2w = np.gradient(dw, h)
        g = _durrleman_g(kf, w, dw, d2w)
        if np.nanmin(g) < -1e-3:
            raise ValueError(f"butterfly arb T={Ty}: min g={np.nanmin(g):.4f}")
        if np.abs(dw[np.isfinite(dw)]).max() > 2.0 + 1e-6:
            raise ValueError("Lee bound breach")
        s = vv / 100.0 * np.sqrt(float(Ty))
        d1 = -kf / np.maximum(s, 1e-8) + s / 2
        d2 = d1 - s
        if not (np.all(np.diff(d1) <= 1e-8) and np.all(np.diff(d2) <= 1e-8)):
            raise ValueError("Fukasawa d1/d2 non-monotone")


def assert_ssvi_bounds(thetas, rhos, etas, gamma=GAMMA):
    """Gatheral-Jacquier Thm 4.2 sufficient conditions."""
    phi = np.asarray(etas) / np.maximum(np.asarray(thetas), 1e-8) ** gamma
    if np.nanmax(thetas * phi * (1 + np.abs(rhos))) >= 4.0:
        raise ValueError("Thm4.2 cond1 breach")
    if np.nanmax(thetas * phi * phi * (1 + np.abs(rhos))) > 4.0 + 1e-9:
        raise ValueError("Thm4.2 cond2 breach")


def breeden_litzenberger_density(call_prices, strikes, r=0.0, T=1.0):
    """Risk-neutral density via BL finite difference; negative = arb signal.

    call_prices: (K,) arbitrage-free call values on uniform strike grid.
    Returns (density, min_density). Caller fails CI if min < 0 (beyond tol).
    """
    c = np.asarray(call_prices, dtype=float)
    k = np.asarray(strikes, dtype=float)
    h = k[1] - k[0]
    d2 = (c[2:] - 2 * c[1:-1] + c[:-2]) / h ** 2
    dens = np.exp(r * T) * d2
    return dens, float(np.min(dens))


def assert_coverage_nesting(iv_cov_60, iv_cov_10, iv_cov_720):
    """10d and 720d must be strict subsets of the 60d mask."""
    if bool(((np.asarray(iv_cov_10)) & (~np.asarray(iv_cov_60))).sum()):
        raise ValueError("10d must be subset of 60d")
    if bool(((np.asarray(iv_cov_720)) & (~np.asarray(iv_cov_60))).sum()):
        raise ValueError("720d must be subset of 60d")


def garman_klass_estimate(open_, high, low, close):
    """Range-based vol estimator (PDF gap-fill): more efficient than
    close-close; diagnostic lens for the iv_base anchor, not a gate."""
    o, h, l, c = (np.log(np.maximum(np.asarray(x), 1e-12)) for x in (open_, high, low, close))
    return np.sqrt(np.maximum(0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2, 0.0))
