import numpy as np
from dataclasses import dataclass

EDGE = {
    'rev5': -0.00045,
    'sent': 0.00011,
    'mom120': 0.00012,
    'skew': -0.00025,
    'quality': 0.00016,
    'lowvol': 0.00008,
    'fmom': 0.00013,
    'lev': 0.00020,
}

REGIME = {2020: 1.5, 2021: 0.5, 2022: 1.0}

# Base-process realism (audit 2026-09-06, items 10-12, one coherent pass).
# Replaces the static MOM_MULT/REV_MULT year dummies (deleted — they were an
# admitted interim): the EDGE crisis response is now a PORTABLE trailing gate,
# so any crash, synthetic or real, triggers it; no calendar lookup anywhere.
# (11) f_m is GJR-GARCH(1,1)-t, not i.i.d.: sig2 = w + a*e^2 + g*e^2*I(e<0) +
#      b*sig2, z ~ t(nu) scaled to unit variance. Textbook equity-daily
#      (a~0.07, b~0.87, g~0.08, nu=7; stationarity a+b+g/2<1). Rationale: an
#      i.i.d. Gaussian base never enters the state any portable panic gate
#      detects (both tested gates fired P=0.00 on synthetic) — fix base first.
#      NOTE: GARCH clusters amplitude, not rank churn; churn comes from (12).
# STRESS: one DATED stress episode (COVID crash 2020-02-24..2020-03-31: drift
#      + vol multiplier on f_m). Dated, not a year dummy on the edge: without
#      a dated stress no seed reliably puts a crash in 2020 and the falsifier
#      could not test the gate. 2019-style calm-market failures stay unmodeled.
# (10) Panic gate (trailing only): panic = (126d drawdown < -0.15) AND
#      (126d trailing vol > 2x trailing median), 5d-majority hysteresis.
#      mom_mult = -1 / rev_mult = 0.5 in panic (shootout-winning magnitudes:
#      full -1 flip robust across seeds, -0.5 not; reversal dampened per
#      crowded-reversal evidence, NOT strengthened). Calibrated so
#      P(panic|synthetic) ~= P(panic|real) ~ 1-5%.
# (12) Churn package — NEUTRALIZED 2026-09-06 (sweep_tmp.py): a 2x2 sensitivity
#      sweep (MICRO_K 0.15/0.4 x BETA_RW 0.008/0.02) moved E2 turnover <=0.1pp
#      while halving reversal Sharpe — base churn does NOT drive E2 turnover;
#      turnover is controlled by the decay setting (decay 2 = 25.0% vs real
#      27.6%), so comparisons must be DECAY-MATCHED. Static betas + no micro
#      noise (parsimony). Kept (mild, documented): idio-vol co-movement,
#      phi 0.92, jumps 0.006.
MU_M, SIG_BAR = 0.0003, 0.012
GARCH_A, GARCH_B, GJR_G, GARCH_NU = 0.07, 0.87, 0.08, 7
GARCH_SIG_CAP, GARCH_Z_CAP = 0.05, 5.0
STRESS_START, STRESS_END = np.datetime64('2020-02-24'), np.datetime64('2020-03-31')
STRESS_DRIFT, STRESS_VOL = -0.008, 2.0
# 2022 bear grind (S&P -19%, persistent): mild daily bleed + elevated vol for
# most of 2022. Real-anchored like STRESS (not an edge year dummy): it keeps
# the trailing gate firing through 2022 so momentum prints negative years the
# way real 2022 (-0.60) did. Without it the gate only ever fires in Mar-2020
# and no simulated year can go negative — the original Bug-2 complaint.
BEAR_START, BEAR_END = np.datetime64('2022-01-03'), np.datetime64('2022-10-31')
BEAR_DRIFT, BEAR_VOL = -0.0024, 1.75
PANIC_DD, PANIC_VR, PANIC_SMOOTH = -0.10, 2.5, 5
MOM_PANIC, MOM_CALM = -1.0, 1.0
REV_PANIC, REV_CALM = 0.5, 1.0
# L1 statistics (Phase 4, Agent 2 blueprint):
# (a) 2-state vol chain (SIMULATED, never fitted): calm mult 1.0 / crisis
#     mult 2.0 on market vol, sticky transitions P=[[0.985,0.015],[0.10,0.90]].
#     Exogenous stochastic crisis clustering beyond the dated episodes; the
#     panic gate keys off realized vol/drawdown, never the latent state.
# (b) Common tail mixer: one chi-square draw per day rescales ALL cross-
#     sectional innovations (t-copula-with-identity dependence: joint fat-tail
#     days, symmetric). Clayton-asymmetric lower-tail coupling explicitly
#     deferred (crashes are asymmetric; t is not) — logged limitation.
CHAIN_P = ((0.985, 0.015), (0.10, 0.90))
CHAIN_MULT = (1.0, 2.0)
TAIL_NU = 6
BETA_RW, MICRO_K = 0.0, 0.0
SENT_PHI, SENT_JUMP_P = 0.92, 0.006
# Calibration instrumentation (not a model param): last generate() call's
# panic rate overall + in-2020, for gate fire-rate checks. Read-only usage.
LAST_PANIC_RATE = {'overall': None, 'y2020': None}


@dataclass
class Panel:
    dates: np.ndarray
    fields: dict
    vector_fields: dict
    groups: dict
    subuniverse: np.ndarray
    # Core Operational Invariant 2 — STRICT LINEAGE. dataset_id is the content
    # fingerprint of the panel (sha256 over shapes + array bytes + build
    # params). Empty string = unlogged provenance: any alpha evaluated on such
    # a panel is assigned DSR = 0 and blocked from promotion (see selection.py).
    dataset_id: str = ""


def fingerprint_panel(dates, fields, vector_fields, **params):
    """Deterministic content fingerprint (hex sha256) for lineage logging."""
    import hashlib
    import json as _json
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(dates.astype('datetime64[D]').astype(np.int64)).tobytes())
    for k in sorted(fields):
        v = np.ascontiguousarray(fields[k])
        h.update(k.encode() + str(v.shape).encode() + str(v.dtype).encode())
        h.update(np.where(np.isfinite(v), v, 0.0).tobytes())
    for k in sorted(vector_fields):
        for p in vector_fields[k]:
            v = np.ascontiguousarray(p)
            h.update(k.encode() + str(v.shape).encode() + str(v.dtype).encode())
            h.update(np.where(np.isfinite(v), v, 0.0).tobytes())
    h.update(_json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _roll_mean(x, w):
    T, N = x.shape
    xm = np.where(np.isfinite(x), x, 0.0)
    vm = np.isfinite(x).astype(np.float64)
    cx = np.vstack([np.zeros((1, N)), np.cumsum(xm, 0)])
    cv = np.vstack([np.zeros((1, N)), np.cumsum(vm, 0)])
    S = cx[w:] - cx[:-w]
    C = cv[w:] - cv[:-w]
    out = np.full_like(x, np.nan)
    out[w - 1:] = np.where(C > 0, S / np.maximum(C, 1e-12), np.nan)
    return out


def _roll_std(x, w):
    T, N = x.shape
    xm = np.where(np.isfinite(x), x, 0.0)
    x2 = xm * xm
    vm = np.isfinite(x).astype(np.float64)
    cx = np.vstack([np.zeros((1, N)), np.cumsum(xm, 0)])
    cx2 = np.vstack([np.zeros((1, N)), np.cumsum(x2, 0)])
    cv = np.vstack([np.zeros((1, N)), np.cumsum(vm, 0)])
    S = cx[w:] - cx[:-w]
    S2 = cx2[w:] - cx2[:-w]
    C = cv[w:] - cv[:-w]
    out = np.full_like(x, np.nan)
    mean = np.where(C > 0, S / np.maximum(C, 1e-12), 0.0)
    var = np.maximum(S2 / np.maximum(C, 1e-12) - mean * mean, 0.0)
    out[w - 1:] = np.where(C > 0, np.sqrt(var), np.nan)
    return out


def _cs_z(x):
    m = np.isfinite(x)
    cnt = m.sum(1, keepdims=True)
    s = np.where(m, x, 0.0).sum(1, keepdims=True)
    mean = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    v = np.where(m, (x - mean) ** 2, 0.0).sum(1, keepdims=True)
    sd = np.sqrt(v / np.maximum(cnt - 1, 1))
    sd = np.where(sd > 0, sd, 1.0)
    z = (x - mean) / sd
    return np.where(m, z, np.nan)


def _rank1d(v):
    order = np.argsort(v, kind='stable')
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v))
    return ranks / (len(v) - 1)


def _blackout_masks(rng, T, N, day_prob, min_cov, max_dur):
    blackout = np.zeros(T, dtype=bool)
    t = 0
    while t < T:
        if rng.random() < day_prob:
            dur = int(rng.integers(1, max_dur + 1))
            blackout[t:t + dur] = True
            t += dur
        else:
            t += 1
    return blackout


def generate(seed=7, n_stocks=1000, start='2020-01-01', end='2022-12-31'):
    rng = np.random.default_rng(seed)
    all_days = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    dates = all_days[np.is_busday(all_days)]
    T = len(dates)
    N = n_stocks

    sector = rng.integers(0, 10, N)
    industry = sector * 3 + rng.integers(0, 3, N)
    beta_m0 = rng.normal(1.0, 0.25, N)
    beta_s = rng.normal(1.0, 0.35, N)
    beta_i = rng.normal(0.5, 0.25, N)
    sig_idio0 = np.exp(rng.normal(np.log(0.016), 0.35, N))

    # (11) GJR-GARCH(1,1)-t market factor + dated COVID stress (see header).
    omega = SIG_BAR ** 2 * max(1.0 - GARCH_A - GARCH_B - GJR_G / 2.0, 1e-6)
    t_scale = float(np.sqrt(GARCH_NU / (GARCH_NU - 2.0)))
    f_m = np.empty(T)
    sig2 = SIG_BAR ** 2
    e_prev = 0.0
    stress = (dates >= STRESS_START) & (dates <= STRESS_END)
    bear = (dates >= BEAR_START) & (dates <= BEAR_END)
    # L1(a): exogenous 2-state vol chain (calm/crisis), drawn before f_m.
    chain = np.zeros(T, dtype=int)
    for t in range(1, T):
        stay = CHAIN_P[chain[t - 1]][chain[t - 1]]
        chain[t] = chain[t - 1] if rng.random() < stay else 1 - chain[t - 1]
    chain_mult = np.array([CHAIN_MULT[s] for s in chain])
    # L1(b): common tail mixer — one chi-square per day rescales every
    # cross-sectional innovation (joint fat-tail days).
    tail_scale = 1.0 / np.sqrt(rng.chisquare(TAIL_NU, T) / TAIL_NU)
    for t in range(T):
        sig2 = omega + GARCH_A * e_prev ** 2 + GJR_G * e_prev ** 2 * (e_prev < 0) + GARCH_B * sig2
        sig2 = min(max(sig2, 1e-12), GARCH_SIG_CAP ** 2)
        vol = np.sqrt(sig2) * chain_mult[t]  # L1(a): chain scales market vol
        drift = MU_M
        if stress[t]:
            vol = min(vol * STRESS_VOL, GARCH_SIG_CAP)
            drift += STRESS_DRIFT
        if bear[t]:
            vol = min(vol * BEAR_VOL, GARCH_SIG_CAP)
            drift += BEAR_DRIFT
        z = max(min(rng.standard_t(GARCH_NU) / t_scale, GARCH_Z_CAP), -GARCH_Z_CAP)
        f = drift + vol * z
        f_m[t] = f
        e_prev = f - MU_M
    sig_m = np.sqrt(np.maximum(
        np.array([np.mean((f_m[max(0, t - 22):t + 1] - MU_M) ** 2) if t > 0 else SIG_BAR ** 2
                  for t in range(T)]), 1e-12))
    # (12) betas static + idio scale co-moving with market vol. A random-walk
    # beta / micro-noise churn package was built, swept, and NEUTRALIZED (it
    # moved E2 turnover <=0.1pp while damaging reversal Sharpe): turnover is
    # decay-setting-controlled, so comparisons must be DECAY-MATCHED.
    beta_m = beta_m0[None, :] + np.cumsum(rng.normal(0.0, BETA_RW, (T, N)), axis=0)
    beta_s_t = beta_s[None, :] + np.cumsum(rng.normal(0.0, BETA_RW, (T, 10)), axis=0)[:, sector]
    beta_i_t = beta_i[None, :] + np.cumsum(rng.normal(0.0, BETA_RW, (T, 30)), axis=0)[:, industry]
    # RR-21 (2026-09-06): cap 1.5, not 3.0. In liquidations correlations must
    # SPIKE (systematic dominates); letting idio triple in stress drowned the
    # common factor and tail-day paircorr DROPPED (0.27 vs 0.37 all-days).
    # Market chain (2.0x) + GARCH stress now outrank idio in crisis.
    idio_scale = np.clip(1.0 + 0.8 * (sig_m / SIG_BAR - 1.0), 0.5, 1.5)
    # L1(b): common tail mixer multiplies every cross-sectional innovation.
    ts = tail_scale[:, None]
    f_s = rng.normal(0.0, 0.0035, (T, 10))[:, sector] * ts
    f_i = rng.normal(0.0, 0.0015, (T, 30))[:, industry] * ts
    idio = rng.normal(0.0, 1.0, (T, N)) * sig_idio0[None, :] * idio_scale[:, None] * ts
    base = beta_m * f_m[:, None] + beta_s_t * f_s + beta_i_t * f_i + idio

    size = np.exp(rng.normal(10.0, 1.0, N))
    liq_rank = _rank1d(size)
    illiq = (1.8 - 1.3 * liq_rank)[None, :]
    liqf = (0.5 + liq_rank)[None, :]

    rev5 = np.nan_to_num(_cs_z(_roll_mean(base, 5)), nan=0.0)
    mom120 = np.nan_to_num(_cs_z(_roll_mean(base, 120)), nan=0.0)
    rv20 = _roll_std(base, 20)
    med_rv = np.nanmedian(rv20, axis=0)
    rv20 = np.where(np.isfinite(rv20), rv20, med_rv)
    lowvol_z = np.nan_to_num(_cs_z(rv20), nan=0.0)

    margin = rng.beta(3.0, 12.0, N) * 0.35
    lev = np.clip(rng.normal(0.35, 0.15, N), 0.02, 0.9)
    skew0 = rng.normal(0.03, 0.05, N)
    quality_z = (_rank1d(margin) - 0.5) * 2.0
    skew_z = (_rank1d(skew0) - 0.5) * 2.0
    # Item 14a (2026-09-06): leverage level effect, POSITIVE sign per real-Brain
    # outcome (unsigned rank(debt/assets) PASSES on real; simulator printed null
    # +0.37 FAIL because no leverage leg existed at all). Small coeff so it
    # shows up without dominating. Period caveat: 2020-22 low rates plausibly
    # rewarded levered firms; sign may not generalize — and it tensions the
    # classic leverage-aversion literature, so this stays flagged, not settled.
    lev_z = (_rank1d(lev) - 0.5) * 2.0

    phi = SENT_PHI
    sent_latent = np.zeros((T, N))
    s = rng.normal(0.0, 1.0, N)
    innov_scale = np.sqrt(1 - phi * phi)
    for t in range(T):
        s = phi * s + innov_scale * rng.normal(0.0, 1.0, N)
        sent_latent[t] = s
    sent_z = np.nan_to_num(_cs_z(sent_latent), nan=0.0)

    years = dates.astype('datetime64[Y]').astype(int) + 1970
    regime = np.array([REGIME.get(int(y), 1.0) for y in years])[:, None]
    # (10) portable panic gate on f_m (trailing only, causal: windows end t-1).
    # Probe-verified 2026-09-06 (gateprobe2): 21d-vol-ratio peaks 2.68-4.44 in
    # 2020 vs <=1.80 outside (seed42 2.74 outlier handled by the joint DD leg);
    # point-drawdown troughs -0.24..-0.49, all in 2020. A 126d-vol-ratio gate
    # was REJECTED (peaks 1.47-1.96, never 2x: slow GARCH absorbs stress into
    # both numerator and denominator), as was ratio-drawdown cum/max-1
    # (ill-scaled near zero, divide warnings). Short numerator + point DD it is.
    trail = np.full(T, np.nan)
    for t in range(1, T):
        lo = max(0, t - 21)
        w = f_m[lo:t]
        trail[t] = np.std(w, ddof=1) if t - lo >= 10 else np.nan
    trail_med = np.full(T, np.nan)
    for t in range(1, T):
        lo = max(0, t - 504)
        w = trail[lo:t]
        w = w[np.isfinite(w)]
        trail_med[t] = np.median(w) if len(w) >= 63 else np.nan
    cum = np.cumsum(f_m)
    runmax = np.maximum.accumulate(np.concatenate([[0.0], cum[:-1]]))
    ddpts = cum - runmax  # point drawdown in return units (no ratio scaling)
    panic_raw = (ddpts < PANIC_DD) & (trail > PANIC_VR * trail_med)
    panic_raw[:63] = False
    panic_raw = np.where(np.isfinite(trail) & np.isfinite(trail_med), panic_raw, False)
    # 5d-majority hysteresis against flicker.
    k = PANIC_SMOOTH
    panic = np.zeros(T, dtype=bool)
    for t in range(T):
        lo = max(0, t - k + 1)
        panic[t] = panic_raw[lo:t + 1].sum() >= (k // 2 + 1)
    mom_mult = np.where(panic, MOM_PANIC, MOM_CALM)[:, None]
    rev_mult = np.where(panic, REV_PANIC, REV_CALM)[:, None]
    LAST_PANIC_RATE['overall'] = float(panic.mean())
    LAST_PANIC_RATE['y2020'] = float(panic[years == 2020].mean())

    nq_pre = int(np.ceil(T / 63))
    margin_q = np.clip(margin[None, :] + rng.normal(0.0, 0.008, (N, nq_pre)).T, 0.01, 0.45)
    # Phase 4 PIT: margins served knowledge-bounded (publication lag 20-45d +
    # 2% restatements) instead of calendar forward-fill with zero lag. Early
    # days with nothing yet published contribute zero edge (neutral), never
    # NaN returns — see fmom_z nan_to_num below.
    from pit import build_revision_log as _brl, pit_asof as _asof, pit_asof_multi as _asof_multi, validate_pit as _vpit
    _pe = np.array([dates[min((q + 1) * 63 - 1, T - 1)] for q in range(nq_pre)])
    _mrows = _brl(_pe, margin_q, "margin", rng)
    margin_pit, _mkn = _asof(_mrows, dates)
    _vpit(margin_pit, _mkn, dates)
    dmargin = np.diff(margin_pit, axis=0, prepend=margin_pit[:1])
    dm_z = np.nan_to_num(_cs_z(dmargin), nan=0.0)
    fmom_z = dm_z

    edge = (EDGE['rev5'] * rev5 * illiq * rev_mult
            + EDGE['sent'] * sent_z * illiq
            + EDGE['mom120'] * mom120 * mom_mult
            + EDGE['skew'] * skew_z[None, :]
            + EDGE['quality'] * quality_z[None, :] * liqf
            + EDGE['lowvol'] * (-lowvol_z)
            + EDGE['lev'] * lev_z[None, :] * liqf
            + EDGE['fmom'] * fmom_z) * regime

    r = base + edge
    # (12) microstructure noise: bid-ask bounce proxy, scaled by name idio.
    r = r + rng.normal(0.0, 1.0, (T, N)) * (MICRO_K * sig_idio0[None, :])

    p0 = np.exp(rng.uniform(np.log(5.0), np.log(400.0), N))
    close = p0[None, :] * np.cumprod(1.0 + r, axis=0)
    open_ = np.vstack([close[:1], close[:-1]]) * (1.0 + rng.normal(0.0, 0.004, (T, N)))
    rng_hl = np.abs(rng.normal(0.0, 1.0, (T, N))) * sig_idio0[None, :] * 0.7
    high = np.maximum(open_, close) * (1.0 + rng_hl)
    low = np.minimum(open_, close) * (1.0 - rng_hl)

    dollar_vol = size[None, :] * (1.0 + 0.3 * np.abs(np.nan_to_num(_cs_z(r)))) * np.exp(rng.normal(0.0, 0.35, (T, N)))
    volume = dollar_vol / close
    adv20 = _roll_mean(dollar_vol, 20)
    shares_out = size / np.exp(rng.normal(3.0, 0.5, N))
    cap = shares_out[None, :] * close

    nq = int(np.ceil(T / 63))
    sales_q = (size * 0.8)[None, :] * np.exp(rng.normal(0.0, 0.06, (N, nq))).T
    turnover_a = rng.uniform(0.4, 1.8, N)
    assets_q = sales_q / turnover_a[None, :]
    # Item 14a: quarterly leverage innovations (random walk, sigma 0.015/q).
    # Static lev made debt/assets near-frozen (signal TO 0.1%, below the 1%
    # floor no matter the edge sign); real leverage ratios move quarterly.
    lev_q = np.clip(lev[None, :] + np.cumsum(rng.normal(0.0, 0.015, (nq, N)), axis=0), 0.02, 0.9)
    debt_q = assets_q * lev_q
    curr_a = rng.uniform(0.25, 0.5, N)
    liab_c = rng.uniform(0.08, 0.3, N)

    # Phase 4 PIT: quarterly levels served knowledge-bounded (same 20-45d
    # publication lag + 2% restatements as margin above) instead of zero-lag
    # calendar forward-fill. ebitda reuses the pre-edge PIT margin so the
    # numerator and the fmom signal agree on what was knowable when. The
    # three post-edge fields share one DuckDB query (pit_asof_multi).
    _pe2 = np.array([dates[min((q + 1) * 63 - 1, T - 1)] for q in range(nq)])
    _pit3 = _asof_multi({"sales": _brl(_pe2, sales_q, "sales", rng),
                         "assets": _brl(_pe2, assets_q, "assets", rng),
                         "debt": _brl(_pe2, debt_q, "debt", rng)}, dates)
    sales, _skn = _pit3["sales"]
    assets, _akn = _pit3["assets"]
    debt, _dkn = _pit3["debt"]
    _vpit(sales, _skn, dates)
    _vpit(assets, _akn, dates)
    _vpit(debt, _dkn, dates)
    ebitda = sales * margin_pit
    assets_curr = assets * curr_a[None, :]
    liabilities_curr = assets * liab_c[None, :]

    iv_base = rv20 * np.sqrt(252.0) * 100.0 * (1.05 + rng.normal(0.0, 0.08, (T, N))) + 8.0
    # Sec 3.3: options coverage shrinks with shorter tenor + smaller cap.
    # 60-day IV prints broadly; 10-day IV is a strict sparser subset —
    # near-full for TOP500-equivalent names, sharp drop for illiquid names.
    p_iv = 0.10 + 0.85 * liq_rank[None, :] ** 2
    iv_cov = rng.random((T, N)) < p_iv
    iv_black = _blackout_masks(rng, T, 1, 0.008, 0.1, 5)
    iv_cov &= ~iv_black[:, None]
    iv_cov_60 = iv_cov
    short_keep = (0.35 + 0.60 * liq_rank[None, :])  # tenor gradient
    iv_cov_10 = iv_cov_60 & (rng.random((T, N)) < short_keep)

    def _tiny_scale(mask):
        hit = mask & (rng.random((T, N)) < 0.004) & (liq_rank[None, :] < 0.35)
        return np.where(hit, rng.uniform(0.02, 0.1, (T, N)), 1.0)

    call_10 = iv_base * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    call_60 = iv_base * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    put_10 = iv_base * (1.0 + skew0[None, :]) * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    put_60 = iv_base * (1.0 + skew0[None, :]) * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    call_10 = np.where(iv_cov_10, call_10, np.nan)
    call_60 = np.where(iv_cov_60, call_60, np.nan)
    put_10 = np.where(iv_cov_10, put_10, np.nan)
    put_60 = np.where(iv_cov_60, put_60, np.nan)

    # Sec 3.3 (recalibrated 2026-09-06 vs 4 real-Brain records):
    # real nws12-style sentiment prints ~1-5% of cells (2-72 long/short per
    # side on TOP3000), not ~47%. Prior p_cov=0.25+0.5*liq (mean ~0.50) gave
    # false confidence on sentiment+fundamental combos (syn S 2.9-3.4/F 2.0-2.5
    # -> real S 0.6-0.9/F 0.6-0.8 with 47-50% concentration). Sparse gradient
    # + longer blackouts: mean cov ~0.04-0.06, top names ~0.12, tail ~0.02.
    p_cov = 0.02 + 0.10 * liq_rank[None, :] ** 2
    sent_black = _blackout_masks(rng, T, 1, 0.03, 0.01, 30)
    in_black = sent_black[:, None]
    sent_cov = (rng.random((T, N)) < p_cov) & (~in_black | (rng.random((T, N)) < 0.01))
    jumps = np.where(rng.random((T, N)) < SENT_JUMP_P, rng.choice([-1.0, 1.0], (T, N)) * rng.uniform(3.0, 7.0, (T, N)), 0.0)
    sent_obs = np.where(sent_cov, sent_z * 0.9 + rng.normal(0.0, 0.5, (T, N)) + jumps, np.nan)
    vec_parts = [np.where(sent_cov, sent_obs + rng.normal(0.0, 0.35, (T, N)), np.nan) for _ in range(3)]
    buzz = np.where(sent_cov, 1.5 + np.abs(sent_z) * 1.5 + rng.normal(0.0, 0.7, (T, N)), np.nan)

    # Phase 2 delistings (see note above): applied to price/volume/returns;
    # adv20 trailing mean decays naturally (realistic); cap follows close.
    if T > 63:
        n_delist = int(round(N * DELIST_P * (T / 252.0)))
        if n_delist > 0:
            who = rng.choice(N, size=min(n_delist, N), replace=False)
            for j in sorted(who.tolist()):
                dday = int(rng.integers(63, T))
                for arr in (close, open_, high, low, volume):
                    arr[dday:, j] = np.nan
                r[dday, j] = -0.5
                if dday + 1 < T:
                    r[dday + 1:, j] = 0.0
    fields = {
        'close': close,
        'open': open_,
        'high': high,
        'low': low,
        'volume': volume,
        'returns': r,
        'adv20': adv20,
        'cap': cap,
        'ebitda': ebitda,
        'sales': sales,
        'debt': debt,
        'assets': assets,
        'liabilities_curr': liabilities_curr,
        'assets_curr': assets_curr,
        'implied_volatility_call_10': call_10,
        'implied_volatility_call_60': call_60,
        'implied_volatility_put_10': put_10,
        'implied_volatility_put_60': put_60,
        'buzz': buzz,
    }
    vector_fields = {'nws12_afterhsz_01l': vec_parts}
    groups = {'sector': sector, 'industry': industry}
    n_sub = max(1, int(N * 0.5))
    subuniverse = liq_rank >= np.sort(liq_rank)[-n_sub]
    panel = Panel(dates=dates, fields=fields, vector_fields=vector_fields,
                  groups=groups, subuniverse=subuniverse,
                  dataset_id=fingerprint_panel(dates, fields, vector_fields,
                                               seed=seed, n_stocks=N,
                                               start=str(start), end=str(end)))
    validate_panel(panel)
    return panel
# Field staleness (institutional data-integrity note, Phase 2): daily =
# close/open/high/low/volume/returns/buzz/IV (IV + sentiment gated by
# coverage masks); quarterly-step = ebitda/sales/debt/assets/currents (63d
# forward-fill, innovations quarterly); static characteristic = margin/lev/
# skew0/quality_z/skew_z/lev_z/sector/industry/size (no time variation).
# adv20 trails 20d. Consumers must not treat quarterly/static fields as news.
#
# Phase 2: delistings. ~2%/yr of names delist on a random post-2020 date:
# -50% print on the delist day, then NaN close/volume (pasteurization drops
# them; point-in-time universe + flat thin-days handle the rest). Without
# this the panel is survivorship-free by construction and concentration/
# turnover never face the dropout churn real books do.
DELIST_P = 0.02


def validate_panel(panel):
    """Fail loudly on malformed panels (Phase 2 data-integrity gate).

    Checks: required fields present, shapes agree, dates monotonic business
    days, close essentially complete, adv20 mostly finite, no fully-NaN day
    past the 20d warmup, subuniverse mask sane. Called by generate() and
    generate_real_panel(); raises ValueError naming the violation.
    """
    req = {'close', 'open', 'high', 'low', 'volume', 'returns', 'adv20', 'cap',
           'ebitda', 'sales', 'debt', 'assets',
           'implied_volatility_call_10', 'implied_volatility_call_60',
           'implied_volatility_put_10', 'implied_volatility_put_60', 'buzz'}
    missing = req - set(panel.fields)
    if missing:
        raise ValueError(f"panel missing fields: {sorted(missing)}")
    T, N = panel.fields['returns'].shape
    for k, v in panel.fields.items():
        if v.shape != (T, N):
            raise ValueError(f"field {k} shape {v.shape} != ({T},{N})")
    if not bool((np.diff(panel.dates.astype('datetime64[D]').astype(int)) > 0).all()):
        raise ValueError("panel dates not strictly increasing")
    if float(np.isfinite(panel.fields['close']).mean()) < 0.95:
        raise ValueError("close coverage below 95% (delistings account for ~3%)")
    # Raw-vs-log large-move guard (2026-09-06): cumprod(1+r) breaks at r<=-1.
    # The deliberate -50% delist print is the floor; anything at/below -100%
    # is a generator bug, not a market event.
    if bool((panel.fields['returns'][np.isfinite(panel.fields['returns'])] <= -1.0).any()):
        raise ValueError("returns at/below -100% would break price compounding")
    if float(np.isfinite(panel.fields['adv20'][20:]).mean()) < 0.90:
        raise ValueError("adv20 coverage below 90% past 20d warmup")
    KEL = np.isfinite(panel.fields['close']).sum(axis=1)
    if int((KEL == 0).sum()) > 0:
        bad = np.where(KEL == 0)[0]
        if int((bad >= 20).sum()) > 0:
            raise ValueError(f"fully-NaN close days past warmup: {bad[bad >= 20][:5]}")
    sub = np.asarray(panel.subuniverse, dtype=bool)
    if sub.shape != (N,) or not (0 < sub.sum() < N + 1) or sub.sum() < 1:
        raise ValueError("subuniverse mask degenerate")
    return True
