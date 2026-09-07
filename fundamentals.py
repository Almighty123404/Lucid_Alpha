"""VAR(1)-GARCH quarterly fundamentals generator (Agent-2 blueprint).

Production generator for synthetic corporate fundamentals (Sales, EBITDA/CFO
margins, Assets, Leverage). Parametric econometric path — NOT a GAN:
  - sales growth AR(1)-GARCH with size-variance scaling (Gibrat violation)
  - EBITDA/CFO margins VAR(1)-GARCH with correlated-t innovations (rho~0.65:
    the accrual-wedge fix; rho=0 doubles wedge variance)
  - balance-sheet block conditional on margins (asset stickiness,
    counter-cyclical leverage)
  - hard accounting constraints C1-C6 by construction/projection (never
    rejection sampling)

RNG contract: all draws come from the caller-owned `rng` (data_gen forks
`rng_fund` so the f_m/panic/sent streams never shift). No simulator/pit
imports. numpy only.
"""
import numpy as np

FUND_DEFAULTS = {
    # growth block (Eq 9-11)
    "mu_g": 0.015, "phi_g": 0.25, "sig_g": 0.12,
    "garch_g": (0.08, 0.86, 0.06),  # alpha, beta, gamma(asym)
    "sig_g_min": 0.02, "psi": 0.20, "beta_size": -0.005, "lam_m": 0.4,
    # margin block (Eq 13-15)
    "mu_e": 0.12, "mu_c": 0.10,           # margin levels (pre-transform mean)
    "phi_ee": 0.78, "phi_ec": 0.08, "phi_ce": 0.18, "phi_cc": 0.52,
    "sig_e": 0.35, "sig_c": 0.45,          # logit-space innovation scales
    "garch_e": (0.08, 0.88), "garch_c": (0.08, 0.88),
    "sig_min": 0.05, "nu": 6, "rho_ec": 0.65,
    "accrual_beta": (0.10, -0.10),          # (delta_e, delta_c) loading on cycle
    # balance sheet (Eq 17-18)
    "mu_a": 0.0, "phi_a": 0.93, "sig_a": 0.08, "theta_ae": 0.3,
    "mu_l": -0.6, "phi_l": 0.96, "sig_l": 0.10,
    "theta_le": -0.6, "theta_lg": -0.3, "corr_al": 0.2,
    # bounds / constraints
    "lo_e": -0.2, "hi_e": 0.6, "lo_c": -0.3, "hi_c": 0.6, "kappa": 0.15,
    "sig_s0": 0.3,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _to_unit(v, lo, hi):
    return (v - lo) / (hi - lo)


def _from_unit(u, lo, hi):
    return lo + (hi - lo) * u


def _logit(u):
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    return np.log(u / (1.0 - u))


def _correlated_t(n, R, nu, rng):
    """(n,) draws of bivariate-t(nu) with corr R via Cholesky + chi-square."""
    L = np.linalg.cholesky(np.asarray(R, dtype=float))
    z = rng.normal(0.0, 1.0, (n, 2)) @ L.T
    w = rng.chisquare(nu, n) / nu
    return z / np.sqrt(w)[:, None]


def init_fundamentals_state(rng, size, params=FUND_DEFAULTS):
    """Draw stationary initial state. Returns dict of (N,) arrays."""
    N = int(size.shape[0])
    p = params
    # mu_e may be per-firm (data_gen anchors it to the static margin parent so
    # generator EBITDA margins live on the same scale as margin_pit/ebitda).
    mu_e = np.broadcast_to(np.asarray(p["mu_e"], dtype=float), (N,))
    # mu_c may be per-firm (data_gen: margin - 3.5pp accrual wedge, Eq 21).
    mu_c = np.broadcast_to(np.asarray(p["mu_c"], dtype=float), (N,))
    # mu_l may be per-firm (data_gen: logit of the static lev parent, so the
    # quarterly process mean-reverts to the firm's own level and the
    # unsigned-leverage alpha keeps trading the EDGE['lev'] leg, RR-11).
    mu_l = np.broadcast_to(np.asarray(p["mu_l"], dtype=float), (N,))
    s0 = np.log(0.8 * size) + rng.normal(0.0, p["sig_s0"], N)
    E0 = _logit(_to_unit(mu_e, p["lo_e"], p["hi_e"]))
    E0 = E0 + rng.normal(0.0, p["sig_e"], N)
    C0 = _logit(_to_unit(mu_c, p["lo_c"], p["hi_c"]))
    C0 = C0 + rng.normal(0.0, p["sig_c"], N)
    mu_l0 = np.broadcast_to(np.asarray(p["mu_l"], dtype=float), (N,))
    return {
        "s": s0, "g": np.full(N, p["mu_g"]),
        "E": E0, "C": C0,
        "a": np.full(N, p["mu_a"]) + rng.normal(0.0, p["sig_a"], N),
        "ls": mu_l0 + rng.normal(0.0, p["sig_l"], N),
        "vg": np.full(N, p["sig_g"] ** 2),
        "ve": np.full(N, p["sig_e"] ** 2),
        "vc": np.full(N, p["sig_c"] ** 2),
        "eps_g": np.zeros(N),
    }


def generate_fundamentals_q(rng, size, nq, f_mbar=None, params=FUND_DEFAULTS):
    """Quarterly (Q,N) fundamentals. rng: caller-forked Generator (stream-safe).

    f_mbar: (Q,) mean daily market factor per quarter (read-only linkage,
    market -> fundamentals, never feeds back). None = zeros.
    Returns dict of (Q,N) arrays {sales_q, assets_q, debt_q, ebitda_q, cfo_q,
    lev_q, distress} plus audit counts.
    """
    p = params
    size = np.asarray(size, dtype=float)
    N = size.shape[0]
    Q = int(nq)
    size_med = float(np.median(size))
    mu_e = np.broadcast_to(np.asarray(p["mu_e"], dtype=float), (N,))
    mu_c = np.broadcast_to(np.asarray(p["mu_c"], dtype=float), (N,))
    mu_l = np.broadcast_to(np.asarray(p["mu_l"], dtype=float), (N,))
    if f_mbar is None:
        f_mbar = np.zeros(Q)
    f_mbar = np.asarray(f_mbar, dtype=float)
    st = init_fundamentals_state(rng, size, p)
    a_g, b_g, gm_g = p["garch_g"]
    a_e, b_e = p["garch_e"]
    a_c, b_c = p["garch_c"]
    R = [[1.0, p["rho_ec"]], [p["rho_ec"], 1.0]]
    d_e, d_c = p["accrual_beta"]
    out = {k: np.empty((Q, N)) for k in
           ("sales_q", "assets_q", "debt_q", "ebitda_q", "cfo_q", "lev_q")}
    distress = np.zeros((Q, N), dtype=np.int8)
    n_c3 = 0
    n_c4 = 0
    for q in range(Q):
        # --- growth with GJR-GARCH + size scaling (Eq 9-11)
        vg = np.maximum(a_g * st["eps_g"] ** 2
                        + gm_g * st["eps_g"] ** 2 * (st["eps_g"] < 0)
                        + b_g * st["vg"], p["sig_g_min"] ** 2)
        st["vg"] = vg
        scale = (size_med / np.maximum(size, 1e-12)) ** p["psi"]
        eta_g = rng.normal(0.0, 1.0, N)
        g = (p["mu_g"] + p["phi_g"] * (st["g"] - p["mu_g"])
             + p["beta_size"] * np.log(size / size_med)
             + p["lam_m"] * f_mbar[q]
             + np.sqrt(vg) * scale * eta_g)
        st["eps_g"] = np.sqrt(vg) * scale * eta_g
        st["g"] = g
        st["s"] = st["s"] + g
        # --- margins VAR(1)-GARCH, correlated-t (Eq 13-15)
        cyc = f_mbar[q]
        eta = _correlated_t(N, R, p["nu"], rng)
        ve = np.maximum(a_e * (eta[:, 0] ** 2) + b_e * st["ve"], p["sig_min"] ** 2)
        vc = np.maximum(a_c * (eta[:, 1] ** 2) + b_c * st["vc"], p["sig_min"] ** 2)
        st["ve"], st["vc"] = ve, vc
        muE = _logit(_to_unit(mu_e, p["lo_e"], p["hi_e"]))
        muC = _logit(_to_unit(mu_c, p["lo_c"], p["hi_c"]))
        E = (muE + p["phi_ee"] * (st["E"] - muE) + p["phi_ec"] * (st["C"] - muC)
             + d_e * cyc + np.sqrt(ve) * eta[:, 0])
        C = (muC + p["phi_ce"] * (st["E"] - muE) + p["phi_cc"] * (st["C"] - muC)
             + d_c * cyc + np.sqrt(vc) * eta[:, 1])
        st["E"], st["C"] = E, C
        # --- balance sheet (Eq 17-19)
        e_m = _from_unit(_sigmoid(E), p["lo_e"], p["hi_e"])
        z = rng.normal(0.0, 1.0, (N, 2))
        z[:, 1] = p["corr_al"] * z[:, 0] + np.sqrt(1 - p["corr_al"] ** 2) * z[:, 1]
        a = (p["mu_a"] + p["phi_a"] * (st["a"] - p["mu_a"])
             + p["theta_ae"] * (e_m - mu_e) + p["sig_a"] * z[:, 0])
        ls = (mu_l + p["phi_l"] * (st["ls"] - mu_l)
              + p["theta_le"] * (e_m - mu_e) + p["theta_lg"] * g
              + p["sig_l"] * z[:, 1])
        st["a"], st["ls"] = a, ls
        # --- emission (Eq 22-25)
        sales = np.exp(st["s"])
        assets = sales * np.exp(a)
        lev = _sigmoid(ls)
        debt = assets * lev
        c_m = _from_unit(_sigmoid(C), p["lo_c"], p["hi_c"])
        ebitda = sales * e_m
        cfo = sales * c_m
        # --- C4 cash-flow bound (smooth projection)
        over = cfo - ebitda
        cap = p["kappa"] * np.abs(sales)
        viol = over > cap
        if viol.any():
            n_c4 += int(viol.sum())
            cfo = np.where(viol, ebitda + cap * np.tanh(over / np.maximum(cap, 1e-12)), cfo)
        out["sales_q"][q] = sales
        out["assets_q"][q] = assets
        out["debt_q"][q] = debt
        out["ebitda_q"][q] = ebitda
        out["cfo_q"][q] = cfo
        out["lev_q"][q] = lev
        distress[q] = 0
    # --- C3 equity floor is applied by the caller (needs curr-liab ratios);
    # distress flags for negative revision skew are set there. Record C4 count.
    out["distress"] = distress
    out["n_c4_proj"] = n_c4
    out["n_c3_rescale"] = n_c3
    return out


def apply_equity_floor(assets_q, debt_q, curr_liab_q, floor=0.05):
    """C2/C3: Liabilities = debt + curr; Equity = assets - liab; if
    Equity < floor*Assets, rescale debt down and flag distress. Returns
    (debt_adj, equity, distress). Pure function of quarterly grids."""
    liab = debt_q + curr_liab_q
    equity = assets_q - liab
    bad = equity < floor * assets_q
    debt_adj = np.where(bad, assets_q * (1.0 - floor) - curr_liab_q, debt_q)
    debt_adj = np.maximum(debt_adj, 0.0)
    equity = np.where(bad, assets_q - (debt_adj + curr_liab_q), equity)
    return debt_adj, equity, bad.astype(np.int8)


def check_accounting_identities(sales, assets, debt, ebitda, cfo,
                                liabilities, equity, kappa=0.15,
                                lo_e=-0.2, hi_e=0.6, lo_c=-0.3, hi_c=0.6):
    """Vectorized C1-C4 audit. Returns {violation: rate}. All must be 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        m_e = ebitda / np.maximum(sales, 1e-12)
        m_c = cfo / np.maximum(sales, 1e-12)
    return {
        "c1_margin_ebitda": float((~np.isfinite(m_e) | (m_e < lo_e) | (m_e > hi_e)).mean()),
        "c1_margin_cfo": float((~np.isfinite(m_c) | (m_c < lo_c) | (m_c > hi_c)).mean()),
        "c2_identity": float((np.abs(assets - liabilities - equity) > 1e-6).mean()),
        "c3_equity_floor": float((equity < 0).mean()),
        "c4_cashflow_bound": float((cfo > ebitda + kappa * np.abs(sales)).mean()),
    }
