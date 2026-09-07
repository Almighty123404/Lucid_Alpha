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
    # Cashflow-yield level edge (user alpha #3). Calibrated once (2026-09-07):
    # 0.00012 puts the ts_rank(cashflow/cap) leg at clearly positive Sharpe
    # without dominating quality/momentum legs. Do NOT retune per-alpha.
    # PEAD drift edge (C5, added 2026-QFC review): post-announcement drift in
    # the direction of the revision surprise. Direction verified by A/B
    # (EDGE on vs off: revision-drift alpha +0.71 -> +1.63); magnitude
    # UNCALIBRATED (no real event-return shapes yet) — flagged like RR-11,
    # do NOT retune per-alpha.
    'pead': 0.00010,
    # Cashflow-yield level edge (user alpha #3). Calibrated once (2026-09-07):
    # 0.00012 puts the ts_rank(cashflow/cap) leg at clearly positive Sharpe
    # without dominating quality/momentum legs. Do NOT retune per-alpha.
    'cfo': 0.00012,
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
    # Deterministic nesting (no RNG): subindustry must not consume draws —
    # anything drawn here shifts every downstream stream (f_m path, panic
    # gate). Verified: an rng.integers call at this spot zeroed y2020 panic.
    subindustry = industry * 2 + (np.arange(N) % 2)
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
    # Second sentiment latent (scl12 source) lives HERE, pre-edge, mirroring
    # sent_z — not post-edge with the observables. Rationale: the sentiment
    # EDGE leg must see both sources (same mechanism, same coefficient), and
    # edge inputs must exist before r = base + edge. This consumes RNG here,
    # so all downstream streams shift vs pre-scl12 builds (re-verified below);
    # upstream (base, rev5, mom120, f_m gate inputs) is untouched.
    _phi2, _s2 = 0.90, rng.normal(0.0, 1.0, N)
    _lat2 = np.zeros((T, N))
    _is2 = np.sqrt(1 - _phi2 * _phi2)
    for _t in range(T):
        _s2 = _phi2 * _s2 + _is2 * rng.normal(0.0, 1.0, N)
        _lat2[_t] = _s2
    sent2_z = np.nan_to_num(_cs_z(_lat2), nan=0.0)

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
    # Parent unification (2026-QFC review): margin_q comes from the generator's
    # E states (VAR(1) φ≈0.78, real left tail), NOT beta-jitter. Served ebitda
    # keeps its formula (sales × margin_pit) and fmom_z its code — only the
    # margin VALUES change, so C4/fund_ar1/fmom share one margin process.
    # The generator runs FIRST on the forked stream (zero main-rng draws);
    # margin PIT serving stays on main rng. The removed beta-jitter draw
    # shifts downstream main draws (accepted, panic-safe: gate inputs upstream).
    from fundamentals import generate_fundamentals_q as _genfund, apply_equity_floor as _eqfloor
    from fundamentals import FUND_DEFAULTS as _FUND_DEFAULTS
    _rng_fund = np.random.default_rng([seed, 0xF17D])
    _qb = np.array_split(np.arange(T), nq_pre)
    _fbar = np.array([float(np.mean(f_m[b])) for b in _qb])
    _lev_clip = np.clip(lev, 0.03, 0.89)
    _fparams = dict(_FUND_DEFAULTS,
                    mu_e=np.clip(margin, 0.02, 0.40),
                    mu_c=np.clip(margin - 0.035, -0.25, 0.40),
                    mu_l=np.log(_lev_clip / (1.0 - _lev_clip)))
    _fund = _genfund(_rng_fund, size, nq_pre, _fbar, _fparams)
    margin_q = _fund["ebitda_q"] / np.maximum(_fund["sales_q"], 1e-12)
    # Phase 4 PIT: margins served knowledge-bounded (publication lag 20-45d +
    # 2% restatements) instead of calendar forward-fill with zero lag. Early
    # days with nothing yet published contribute zero edge (neutral), never
    # NaN returns — see fmom_z nan_to_num below.
    from pit import build_revision_log_v2 as _brl, pit_asof as _asof, pit_asof_multi as _asof_multi, validate_pit as _vpit
    _pe = np.array([dates[min((q + 1) * 63 - 1, T - 1)] for q in range(nq_pre)])
    _is_q4 = np.array([(q % 4 == 3) for q in range(nq_pre)])
    _mrows = _brl(_pe, margin_q, "margin", rng, liq_rank=liq_rank, is_q4=_is_q4)
    margin_pit, _mkn = _asof(_mrows, dates)
    _vpit(margin_pit, _mkn, dates)
    dmargin = np.diff(margin_pit, axis=0, prepend=margin_pit[:1])
    dm_z = np.nan_to_num(_cs_z(dmargin), nan=0.0)
    fmom_z = dm_z
    # PEAD (C5): revision surprise = served margin vs its 63d-ago published
    # self. A step level persisting ~one quarter (not a 1-day spike), so
    # delay-1 traders harvest the drift it funds (unlike the contemporaneous
    # fmom leg). Roll wrap-around zeroed: no future leak into warmup.
    # NaN-safe like fmom (unknown early days contribute zero, never NaN).
    _pead_raw = margin_pit - np.roll(margin_pit, 63, axis=0)
    _pead_raw[:63] = 0.0
    pead_z = np.nan_to_num(_cs_z(_pead_raw), nan=0.0)

    # Quarterly VALUES from the generator above (fork stream already consumed;
    # PIT revision logs below stay on main rng in fixed field order).
    sales_q, assets_q = _fund["sales_q"], _fund["assets_q"]
    cfo_q = _fund["cfo_q"]
    # WC positivity (corr-leg review #40): curr strictly above liab so working
    # capital never flips sign (old iid uniforms crossed 0, injecting rank
    # noise that nulled debt/working_capital exactly where real passes).
    liab_c = rng.uniform(0.08, 0.3, N)
    curr_a = liab_c + rng.uniform(0.05, 0.30, N)
    # C2/C3 at quarterly level: exact identity, 5% equity floor w/ rescale.
    _curr_lq = assets_q * liab_c[None, :]
    debt_q, _eq_q, _distress_q = _eqfloor(assets_q, assets_q * _fund["lev_q"], _curr_lq)
    _pit4 = _asof_multi({"sales": _brl(_pe, sales_q, "sales", rng, liq_rank=liq_rank, is_q4=_is_q4),
                         "assets": _brl(_pe, assets_q, "assets", rng, liq_rank=liq_rank, is_q4=_is_q4),
                         "debt": _brl(_pe, debt_q, "debt", rng, liq_rank=liq_rank, is_q4=_is_q4, distress=_distress_q),
                         "cashflow_op": _brl(_pe, cfo_q, "cashflow_op", rng, liq_rank=liq_rank, is_q4=_is_q4, distress=_distress_q)}, dates)
    sales, _skn = _pit4["sales"]
    assets, _akn = _pit4["assets"]
    debt, _dkn = _pit4["debt"]
    cashflow_op, _ckn = _pit4["cashflow_op"]
    _vpit(sales, _skn, dates)
    _vpit(assets, _akn, dates)
    _vpit(debt, _dkn, dates)
    _vpit(cashflow_op, _ckn, dates)
    # Cashflow-yield level exposure (user alpha #3 trades ts_rank(cashflow/cap)):
    # NaN-safe like fmom (unknown early days contribute zero, never NaN).
    _cfy = np.nan_to_num(_cs_z(np.where(np.isfinite(assets) & (assets != 0),
                                        cashflow_op / np.maximum(assets, 1e-12), np.nan)), nan=0.0)
    ebitda = sales * margin_pit
    assets_curr = assets * curr_a[None, :]
    liabilities_curr = assets * liab_c[None, :]

    edge = (EDGE['rev5'] * rev5 * illiq * rev_mult
            + EDGE['sent'] * sent_z * illiq
            + EDGE['sent'] * sent2_z * illiq
            + EDGE['mom120'] * mom120 * mom_mult
            + EDGE['skew'] * skew_z[None, :]
            + EDGE['quality'] * quality_z[None, :] * liqf
            + EDGE['lowvol'] * (-lowvol_z)
            + EDGE['lev'] * lev_z[None, :] * liqf
            + EDGE['cfo'] * _cfy * liqf
            + EDGE['fmom'] * fmom_z
            + EDGE['pead'] * pead_z) * regime

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

    iv_base = rv20 * np.sqrt(252.0) * 100.0 * (1.05 + rng.normal(0.0, 0.08, (T, N))) + 8.0
    # C6 SSVI rewire (sticky-ratio puts, fixed 2026-QFC review): calls are
    # arbitrage-free ATM slices; puts = ATM slice x (1 + s_i) with STATIC
    # per-name relative wing markup s_i from the SSVI surface at mean theta.
    # Why static: real desks quote skew (RR/fly) as persistent characteristics
    # and 25d put/call ratios are ~level-proportional; evaluating the wing at
    # daily theta made ratios flap with level (rank chasing -> killed real
    # Brain skew legs #2/#6). Spread in vol points still widens with level
    # (s_i x ATM(t)) and in crisis. Surface asserts below still run on SSVI.
    from iv_surface import (generate_iv_surface as _geniv, term_mult as _tm,
                            ssvi_total_var as _ssviw, ssvi_params as _ssvip,
                            assert_no_calendar_arb as _calarb,
                            assert_coverage_nesting as _nest)
    _crisis = ((chain == 1) | panic | stress | bear).astype(float)
    _mreg = ((np.where(chain == 1, 1.35, 1.0)) * (np.where(panic, 1.25, 1.0))
             * (np.where(stress, 1.20, 1.0)) * (np.where(bear, 1.10, 1.0)))
    _mreg = np.minimum(_mreg, 1.9)[:, None]
    _surf, _thetas = _geniv(iv_base * _mreg, liq_rank, skew0, _crisis)
    _calarb(_thetas)
    _rho_s, _eta_s = _ssvip(liq_rank, skew0, np.zeros(N))
    # Reference-vol evaluation (fixed 2026-QFC review): s_i is a STATIC
    # characteristic evaluated at 25 vol, NOT at each name's mean theta (which
    # collapses dispersion 10x: measured cs-std 0.008 << print noise 0.029).
    # Wing depth per tenor (deeper protection at longer tenors): persistent
    # ranks (cs-std >> noise) with realistic steep-short/flat-long shape.
    _thm = 0.25 ** 2
    _s, _sk = {}, {10.0: -0.15, 60.0: -0.25, 720.0: -0.35}
    for _Td, _Ty in ((10.0, 10.0 / 365.0), (60.0, 60.0 / 365.0), (720.0, 720.0 / 365.0)):
        _th = np.maximum(_thm * _tm(float(_Td)) ** 2, 1e-8)
        _wW = _ssviw(_sk[float(_Td)], _th, _rho_s, _eta_s)
        _wA = _ssviw(0.0, _th, _rho_s, _eta_s)
        _s[float(_Td)] = np.sqrt(_wW / np.maximum(_wA, 1e-12)) - 1.0
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

    call_10 = _surf[10.0][:, :, 2] * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    call_60 = _surf[60.0][:, :, 2] * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    put_10 = _surf[10.0][:, :, 2] * (1.0 + _s[10.0])[None, :] * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    put_60 = _surf[60.0][:, :, 2] * (1.0 + _s[60.0])[None, :] * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    call_10 = np.where(iv_cov_10, call_10, np.nan)
    call_60 = np.where(iv_cov_60, call_60, np.nan)
    put_10 = np.where(iv_cov_10, put_10, np.nan)
    put_60 = np.where(iv_cov_60, put_60, np.nan)
    # 720-day (2Y) tenor: strict sparser subset of the 60d mask — long-dated
    # options print mostly on liquid names. ATM slice (term premium x1.08 via
    # term_mult) with 0.02 noise; put = ATM x static 720 markup (wide skew).
    long_keep = 0.15 + 0.45 * liq_rank[None, :]
    iv_cov_720 = iv_cov_60 & (rng.random((T, N)) < long_keep)
    call_720 = np.where(iv_cov_720, _surf[720.0][:, :, 2] * (1.0 + rng.normal(0.0, 0.02, (T, N))), np.nan)
    put_720 = np.where(iv_cov_720, _surf[720.0][:, :, 2] * (1.0 + _s[720.0])[None, :] * (1.0 + rng.normal(0.0, 0.02, (T, N))), np.nan)
    _nest(iv_cov_60, iv_cov_10, iv_cov_720)

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
    # Second sentiment observables reuse the pre-edge sent2_z latent (computed
    # next to sent_z so the EDGE leg sees it). Only mask/obs noise draws here.
    _p2 = 0.015 + 0.08 * liq_rank[None, :] ** 2
    _blk2 = _blackout_masks(rng, T, 1, 0.02, 0.01, 12)
    _cov2 = (rng.random((T, N)) < _p2) & (~_blk2[:, None] | (rng.random((T, N)) < 0.01))
    _obs2 = np.where(_cov2, sent2_z * 0.9 + rng.normal(0.0, 0.5, (T, N)), np.nan)
    scl12_parts = [np.where(_cov2, _obs2 + rng.normal(0.0, 0.35, (T, N)), np.nan) for _ in range(3)]

    # Phase 2 delistings (see note above): applied to price/volume/returns;
    # adv20 trailing mean decays naturally (realistic); cap follows close.
    # LOTTERY IS FORK-ISOLATED (2026-QFC review): names/dates drawn from
    # (seed, 0xDE17), never the main stream. Rationale: the block sits at the
    # END of generate(), so every upstream draw-count change re-rolled the
    # dropout set (~60 names), moving ALL alpha Sharpes ±0.5 while panic
    # stayed frozen — discovered bisecting a C6 Sharpe gap (a586->2eb5 dsid
    # with identical pre-IV code). One-time re-roll now, stable forever.
    if T > 63:
        _rng_del = np.random.default_rng([seed, 0xDE17])
        n_delist = int(round(N * DELIST_P * (T / 252.0)))
        if n_delist > 0:
            who = _rng_del.choice(N, size=min(n_delist, N), replace=False)
            for j in sorted(who.tolist()):
                dday = int(_rng_del.integers(63, T))
                for arr in (close, open_, high, low, volume):
                    arr[dday:, j] = np.nan
                r[dday, j] = -0.5
                if dday + 1 < T:
                    r[dday + 1:, j] = 0.0
    # Codebook extension (Brain reference mapping): every new field below is
    # either (a) an accounting identity over already-PIT-served parents, so it
    # inherits knowledge timing for free, or (b) a documented synthetic proxy
    # with fresh trailing draws (appended AFTER all existing draws, so no
    # upstream RNG stream shifts). NaN propagates from PIT parents naturally.
    _fin = np.isfinite(close)
    vwap = np.where(_fin, (high + low + close) / 3.0, np.nan)
    shares_out_f = np.where(_fin, shares_out[None, :], np.nan)
    adv60 = _roll_mean(dollar_vol, 60)
    _gm = rng.uniform(0.25, 0.6, N)[None, :]
    cogs = sales * (1.0 - _gm)
    gross_profit = sales - cogs
    operating_income = ebitda - 0.15 * ebitda
    _interest = debt * (0.04 / 252.0)
    _ebt = ebitda - 0.15 * ebitda - _interest
    tax_expense = 0.21 * np.maximum(_ebt, 0.0)
    net_income = _ebt - tax_expense
    eps = net_income / np.maximum(shares_out[None, :], 1e-12)
    liabilities = debt + liabilities_curr
    # C2/C3 daily mirror of the quarterly floor: exact identity, rescale debt
    # where equity would breach 5% of assets (distress handled quarterly).
    _eq_pre = assets - liabilities
    _bad = _eq_pre < 0.05 * assets
    debt = np.where(_bad, assets * 0.95 - liabilities_curr, debt)
    liabilities = debt + liabilities_curr
    equity = assets - liabilities
    cash_and_equiv = assets_curr * 0.3
    retained_earnings = equity * 0.4
    goodwill = assets * 0.1
    working_capital = assets_curr - liabilities_curr
    # C5 single cash-flow truth: the PIT-served cashflow_op IS operating cash
    # flow (killed the parallel ebitda*0.9 series that alpha #3 never traded).
    operating_cash_flow = cashflow_op
    capex = sales * 0.05
    free_cash_flow = operating_cash_flow - capex
    dividends_paid = np.maximum(net_income, 0.0) * 0.3
    # G1 return-on-assets identity (codebook request #32): pure function of
    # PIT-served parents, zero RNG, inherits knowledge timing for free.
    return_assets = net_income / np.maximum(assets, 1e-12)
    # Analyst overlays are synthetic consensus proxies (APPROX, daily noise —
    # not revision-logged like fundamentals; flagged in staleness register).
    est_eps = eps * 4.0 * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    est_revenue = sales * 4.0 * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    est_eps_std = np.abs(est_eps) * 0.15
    recommendation = np.clip(3.0 + rng.normal(0.0, 0.8, (T, N)), 1.0, 5.0)
    eps_surprise = rng.normal(0.0, 0.05, (T, N))
    # Sentiment matrix legs share the calibrated news coverage mask.
    snt_news = np.where(sent_cov, np.clip(sent_z * 0.3 + rng.normal(0.0, 0.2, (T, N)), -1.0, 1.0), np.nan)
    snt_social = np.where(sent_cov, np.clip(sent_z * 0.2 + rng.normal(0.0, 0.3, (T, N)), -1.0, 1.0), np.nan)
    news_volume = np.where(sent_cov, np.maximum(np.round(np.where(np.isfinite(buzz), buzz, 0.0) * 2.0), 0.0), np.nan)
    # Options summary legs from the tenor-graded IV fields (same masks).
    iv_10 = np.where(iv_cov_10, (call_10 + put_10) / 2.0, np.nan)
    iv_30 = np.where(iv_cov_60, (call_60 + put_60) / 2.0, np.nan)
    hv_20 = rv20 * np.sqrt(252.0) * 100.0
    # put_call_ratio is a liquidity-demand proxy (APPROX): no option-volume
    # model exists, so skew + noise stands in. Documented, not structural.
    put_call_ratio = 0.7 + 2.0 * np.clip(skew0[None, :], -0.1, 0.2) + rng.normal(0.0, 0.1, (T, N))
    opt_open_interest = (liq_rank[None, :] * 1e6) * (1.0 + rng.normal(0.0, 0.2, (T, N)))
    # Short/insider overlays: persistent short fraction + sparse insider prints.
    _short_frac = np.clip(np.abs(rng.normal(0.02, 0.02, N))[None, :] + rng.normal(0.0, 0.002, (T, N)), 0.0, 0.2)
    short_interest = np.where(_fin, shares_out[None, :] * _short_frac, np.nan)
    days_to_cover = np.where(_fin, (short_interest * close) / np.maximum(adv20, 1e-12), np.nan)
    borrow_fee = np.clip(0.0025 + _short_frac * 0.5 + rng.normal(0.0, 0.002, (T, N)), 0.0025, None)
    _ins = rng.random((T, N)) < 0.02
    insider_buying = np.where(_ins & _fin, np.abs(rng.normal(0.0, 1.0, (T, N))) * size[None, :] * 1e-4, 0.0)
    insider_selling = np.where(_ins & _fin, -np.abs(rng.normal(0.0, 1.0, (T, N))) * size[None, :] * 1e-4, 0.0)
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
        'cashflow_op': cashflow_op,
        'liabilities_curr': liabilities_curr,
        'assets_curr': assets_curr,
        'implied_volatility_call_10': call_10,
        'implied_volatility_call_60': call_60,
        'implied_volatility_put_10': put_10,
        'implied_volatility_put_60': put_60,
        'implied_volatility_call_720': call_720,
        'implied_volatility_put_720': put_720,
        'buzz': buzz,
        'vwap': vwap,
        'shares_out': shares_out_f,
        'adv60': adv60,
        'cogs': cogs,
        'gross_profit': gross_profit,
        'operating_income': operating_income,
        'net_income': net_income,
        'return_assets': return_assets,
        'eps': eps,
        'tax_expense': tax_expense,
        'liabilities': liabilities,
        'equity': equity,
        'cash_and_equiv': cash_and_equiv,
        'retained_earnings': retained_earnings,
        'goodwill': goodwill,
        'working_capital': working_capital,
        'operating_cash_flow': operating_cash_flow,
        'capex': capex,
        'free_cash_flow': free_cash_flow,
        'dividends_paid': dividends_paid,
        'est_eps': est_eps,
        'est_revenue': est_revenue,
        'est_eps_std': est_eps_std,
        'recommendation': recommendation,
        'eps_surprise': eps_surprise,
        'snt_news': snt_news,
        'snt_social': snt_social,
        'news_volume': news_volume,
        'iv_10': iv_10,
        'iv_30': iv_30,
        'hv_20': hv_20,
        'put_call_ratio': put_call_ratio,
        'opt_open_interest': opt_open_interest,
        'short_interest': short_interest,
        'days_to_cover': days_to_cover,
        'borrow_fee': borrow_fee,
        'insider_buying': insider_buying,
        'insider_selling': insider_selling,
        # Brain-reference aliases (same arrays, no copies): Brain codebooks
        # list multiple names for one series (revenue/sales, ni/net_income,
        # total_debt/debt, ocf/fcf, pcr, dated IV tenors). Aliases keep
        # user-pasted Brain expressions working verbatim.
        'revenue': sales,
        'op_income': operating_income,
        'ni': net_income,
        'total_debt': debt,
        'ocf': operating_cash_flow,
        'fcf': free_cash_flow,
        'pcr': put_call_ratio,
        'implied_volatility_10': iv_10,
        'implied_volatility_30': iv_30,
        'historical_volatility_20': hv_20,
    }
    # Vector (3D tensor) codebook: parts lists match the existing VectorVal
    # convention (one matrix per element). Estimates across forward quarters,
    # IV surface across moneyness, revenue by segment, intraday buckets.
    _est_q = [est_eps * (1.0 + (q + 1) * 0.02 + rng.normal(0.0, 0.02, (T, N))) for q in range(4)]
    _skew_mid = np.where(np.isfinite(call_60) & np.isfinite(put_60), (call_60 + put_60) / 2.0, np.nan)
    _surf = [put_60 * 1.15, put_60 * 1.05, _skew_mid, call_60 * 1.05, call_60 * 1.15]
    _seg_w = rng.dirichlet([1.0, 1.0, 1.0, 1.0], N).T  # (4, N) static shares
    _seg_rev = [sales * _seg_w[q][None, :] for q in range(4)]
    _intra = [volume / 6.0 * (1.0 + rng.normal(0.0, 0.1, (T, N))) for _ in range(6)]
    vector_fields = {'nws12_afterhsz_01l': vec_parts,
                     'scl12_alltype_buzzvec': scl12_parts,
                     # Alias (Brain codebook short name; same parts, no copy).
                     'scl12_buzzvec': scl12_parts,
                     'analyst_eps_estimates': _est_q,
                     'option_implied_vol_surface': _surf,
                     'segment_revenue': _seg_rev,
                     'price_volume_intraday': _intra}
    groups = {'sector': sector, 'industry': industry, 'subindustry': subindustry,
              'market': np.zeros(N, dtype=int)}
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
           'ebitda', 'sales', 'debt', 'assets', 'cashflow_op',
           'implied_volatility_call_10', 'implied_volatility_call_60',
           'implied_volatility_put_10', 'implied_volatility_put_60',
           'implied_volatility_call_720', 'implied_volatility_put_720', 'buzz',
           'vwap', 'shares_out', 'adv60',
           'cogs', 'gross_profit', 'operating_income', 'net_income', 'eps',
           'tax_expense', 'liabilities', 'equity', 'cash_and_equiv',
           'retained_earnings', 'goodwill', 'working_capital',
           'operating_cash_flow', 'capex', 'free_cash_flow', 'dividends_paid',
           'return_assets',
           'est_eps', 'est_revenue', 'est_eps_std', 'recommendation',
           'eps_surprise', 'snt_news', 'snt_social', 'news_volume',
           'iv_10', 'iv_30', 'hv_20', 'put_call_ratio', 'opt_open_interest',
           'short_interest', 'days_to_cover', 'borrow_fee', 'insider_buying',
           'insider_selling', 'revenue', 'op_income', 'ni', 'total_debt', 'ocf',
           'fcf', 'pcr', 'implied_volatility_10', 'implied_volatility_30',
           'historical_volatility_20'}
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
