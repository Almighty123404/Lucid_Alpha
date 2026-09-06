"""WorldQuant Brain-compatible simulation pipeline (Build Spec Sec 5-7).

Ordered stages: parse/type-check -> evaluate -> universe/pasteurization ->
neutralization -> decay -> truncation -> delay -> book normalize -> PnL ->
metrics (Sec 6) + submission gates (Sec 7) + diagnostics (Sec 8).

Backwards-compatible entry point: simulate(expression, panel, refs, own_refs,
settings=None, cutoffs=None). settings=None uses DEFAULT_SETTINGS.
"""
import copy
import json
import warnings

import numpy as np

from fastexpr import parse, parse_program, eval_node, eval_program, Env, WQError
from config import (SimulationSettings, DEFAULT_SETTINGS, DEFAULT_EXECUTION, CUTOFFS as _CFG_CUTOFFS,
                    get_cutoff, subuniverse_cutoff, UNIVERSES)

warnings.filterwarnings('ignore')

# Legacy float map kept for `from simulator import CUTOFFS` consumers.
CUTOFFS = {k: v["value"] for k, v in _CFG_CUTOFFS.items()
           if isinstance(v, dict) and isinstance(v.get("value"), (int, float))}

MIN_NAMES = 5


def _coerce_settings(settings):
    if settings is None:
        return SimulationSettings()
    if isinstance(settings, SimulationSettings):
        return settings
    if isinstance(settings, dict):
        return SimulationSettings(**{k: v for k, v in settings.items()
                                     if k in SimulationSettings.FIELDS})
    raise ValueError(f"settings must be SimulationSettings/dict/None, got {type(settings)}")


def _shift_rows(W, k=1):
    out = np.zeros_like(W)
    if k < W.shape[0]:
        out[k:] = W[:-k]
    return out


# ---------------------------------------------------------------- stage 2-3
def _signal_from_expr(expr_str, panel, verify=True):
    env = Env(panel, verify_units=verify)
    try:
        sig = eval_program(expr_str, env)
    except WQError:
        raise
    except Exception as e:
        raise WQError(str(e))
    if not (isinstance(sig, np.ndarray) and sig.ndim == 2):
        raise WQError("Expression does not produce a cross-sectional matrix")
    return np.where(np.isinf(sig), np.nan, sig).astype(np.float64)


def _universe_mask(panel, universe, window=252, min_obs=20):
    """Top-N by trailing average dollar volume, point-in-time (Sec 2.1, APPROX).

    Audit 2026-09-06 (item 13): membership for day t comes ONLY from data
    before t (trailing `window`-day mean adv20, or |close*volume|, ending at
    t-1; expanding mean while t < min_obs; day-0 snapshot at t=0). Replaces
    the static full-period-mean mask, which had lookahead bias (future
    liquidity selecting the past portfolio) and misclassified 13-18/500 names
    daily vs the point-in-time sort. Returns a (T, N) bool array; tiers with
    size >= N are all-eligible (TOP3000/TOP1000 are no-ops on N=1000 panels).
    Unknown names -> all eligible. Still APPROX (real membership = dated
    index constituents, not a liquidity sort).
    """
    T, N = panel.fields['returns'].shape
    size = UNIVERSES.get(str(universe).upper(), {}).get("size", N)
    if size is None or size >= N:
        return np.ones((T, N), dtype=bool)
    adv = panel.fields.get('adv20', None)
    if adv is None:
        dv = np.abs(panel.fields['close'] * panel.fields.get('volume', 1.0))
        liq = np.where(np.isfinite(dv), dv, np.nan)
    else:
        liq = np.where(np.isfinite(adv), adv, np.nan)
    # Phase 2: vectorized trailing means (cumsum) + Numba top-N kernel.
    # Same contract as the old per-day loop: t=0 uses the day-0 snapshot,
    # t<20 uses expanding mean, else trailing `window` ending t-1 (causal).
    filled = np.where(np.isfinite(liq), liq, 0.0)
    valid = np.isfinite(liq).astype(np.float64)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(filled, axis=0)])
    cc = np.vstack([np.zeros((1, N)), np.cumsum(valid, axis=0)])
    t = np.arange(T)
    lo = np.maximum(t - window, 0)
    den = cc[t] - cc[lo]
    scores = np.where(den > 0, (cs[t] - cs[lo]) / np.maximum(den, 1e-12), np.nan)
    scores[0] = np.where(np.isfinite(liq[0]), liq[0], np.nan)
    for t in range(1, min(min_obs, T)):
        with np.errstate(invalid='ignore'):
            scores[t] = np.nanmean(liq[:t], axis=0)
    from kernels import topn_rows
    return topn_rows(scores, size)


def _eligibility_mask(panel):
    """Pasteurization proxy (Sec 2.6): tradable iff price + liquidity print.

    Requires finite close and finite adv20 (or dollar volume). Delisted /
    halted / insufficient-history names print NaN and are excluded.
    """
    close = panel.fields.get('close', None)
    adv = panel.fields.get('adv20', None)
    T, N = panel.fields['returns'].shape
    elig = np.ones((T, N), dtype=bool)
    if close is not None:
        elig &= np.isfinite(close)
    if adv is not None:
        elig &= np.isfinite(adv)
    else:
        vol = panel.fields.get('volume', None)
        if vol is not None and close is not None:
            elig &= np.isfinite(vol)
    return elig


# ---------------------------------------------------------------- stage 4-5
def _apply_neutralization(sig, panel, level):
    lvl = str(level).upper().replace('-', '')
    if lvl in ("NONE", ""):
        return sig
    if lvl == "MARKET":
        m = np.isfinite(sig)
        cnt = m.sum(1, keepdims=True)
        mean = np.where(m, sig, 0.0).sum(1, keepdims=True) / np.maximum(cnt, 1)
        return np.where(m, sig - mean, np.nan)
    key = {"SECTOR": "sector", "INDUSTRY": "industry",
           "SUBINDUSTRY": "subindustry", "COUNTRY": "country",
           "EXCHANGE": "exchange"}.get(lvl, None)
    groups = getattr(panel, 'groups', {}) or {}
    arr = None
    if key and key in groups:
        arr = np.asarray(groups[key])
    elif lvl == "SUBINDUSTRY" and "industry" in groups:
        arr = np.asarray(groups["industry"])  # fallback: no sub-industry map
    elif lvl in ("COUNTRY", "EXCHANGE") and "sector" in groups:
        arr = np.asarray(groups["sector"])  # fallback: synthetic has no country
    elif "sector" in groups:
        arr = np.asarray(groups["sector"])
    if arr is None:
        return _apply_neutralization(sig, panel, "MARKET")
    out = sig.copy()
    for val in np.unique(arr):
        cols = arr == val
        sub = out[:, cols]
        valid = np.isfinite(sub)
        cnt = valid.sum(1)
        ssum = np.where(valid, sub, 0.0).sum(1)
        mean = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0)
        out[:, cols] = sub - mean[:, None]
    return out


def _apply_decay(sig, n):
    """Global ts_decay_linear post-processing (Sec 2.3). n<=1 is a no-op."""
    if n is None or n <= 1:
        return sig
    # Audit Phase B4: closed-form triangular MA via cumsum (was a per-lag
    # Python loop). num[t] = Σ_{i<w} (w-i)·x[t-i] = w·S0 − S1 with rolling
    # sums S0 (of x) and S1 (of i·x); same NaN-aware zero-fill convention.
    # Verified vs loop form (max-abs-diff ~1e-11 floating-point, NaN-identical).
    w = int(n)
    T, N = sig.shape
    xm = np.where(np.isfinite(sig), sig, 0.0)
    vm = np.isfinite(sig).astype(np.float64)
    cx = np.vstack([np.zeros((1, N)), np.cumsum(xm, axis=0)])
    cv = np.vstack([np.zeros((1, N)), np.cumsum(vm, axis=0)])
    t = np.arange(T)
    cyx = np.vstack([np.zeros((1, N)), np.cumsum(t[:, None] * xm, axis=0)])
    cyv = np.vstack([np.zeros((1, N)), np.cumsum(t[:, None] * vm, axis=0)])
    lo = np.maximum(t + 1 - w, 0)
    S0x = cx[t + 1] - cx[lo]
    S0v = cv[t + 1] - cv[lo]
    S1x = t[:, None] * S0x - (cyx[t + 1] - cyx[lo])
    S1v = t[:, None] * S0v - (cyv[t + 1] - cyv[lo])
    num = w * S0x - S1x
    den = w * S0v - S1v
    out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    out[:w - 1] = np.nan
    return out


# ---------------------------------------------------------------- stage 6-9
def _weights_from_signal(sig, min_names=MIN_NAMES, book_size=1.0,
                         truncation=0.0, nan_handling="ON"):
    """Demean + scale to gross book, with plain-clip truncation (cap-strict).

    Audit 2026-09-06 (item 4): an earlier water-filling loop here was dead
    code (pre-clip guaranteed its re-check never triggered), so this is
    honestly a plain clip: capped names stay at +/-cap and the book runs
    under-invested rather than violating the cap. Deliberately NOT
    ffn-style redistribution (which refills the book but can cascade past
    the cap); matches documented truncation-as-cap semantics.
    """
    T, N = sig.shape
    # Audit Phase B4: vectorized in place (same semantics as the old per-day
    # loop; exact because thin-day handling is stateless since item 3 — no
    # prev-carry to serialize). Verified bit-identical vs loop form.
    # nan_handling=ON remains a documented no-op (group/CS mean imputation
    # ~= demean after centering); missing names get zero active weight.
    m = np.isfinite(sig)
    counts = m.sum(axis=1)
    ok = counts >= min_names
    x = np.where(m, sig, 0.0)
    means = np.where(counts > 0, x.sum(axis=1) / np.maximum(counts, 1), 0.0)
    xd = np.where(m, sig - means[:, None], 0.0)
    s = np.abs(xd).sum(axis=1)
    good = ok & (s >= 1e-12)
    W = np.where(good[:, None] & m, xd / np.maximum(s, 1e-12)[:, None] * book_size, 0.0)
    cap = abs(truncation) * book_size if truncation else 0.0
    if cap > 0:
        W = np.clip(W, -cap, cap)
    return W


def _apply_delay(W, delay):
    """Strict lag on the weight vector (Sec 2.2).

    Tradable weights Wd[t] = W[t-delay]. PnL matches Wd[t-1] against R[t],
    so delay=1 earns R[T+2] from signal T (execute T+1), delay=0 earns
    R[T+1] (execute at T close). Legacy behavior == delay 0 mechanics.
    """
    d = int(delay)
    if d <= 0:
        return W.copy()
    return _shift_rows(W, d)


def _pnl_series(Wd, R):
    # INVARIANT (audit 2026-09-06, item 5): this returns a DAILY PnL series
    # (pnl[t] = day-t profit, pnl[0] = 0), NOT a cumulative curve. Self-corr
    # downstream must consume daily series only — never pass cumsum(pnl).
    # Enforced by construction (both producers below emit daily); there is no
    # robust runtime assert distinguishing daily vs cumulative, so this comment
    # plus the _rolling_self_corr note is the guard. See QuantML
    # wq-alpha-research skill README (daily-changes rule) and xiegengcai
    # DeepWiki self-corr (pnl-ffill-shift convention).
    T = Wd.shape[0]
    pnl = np.zeros(T)
    pnl[1:] = (Wd[:-1] * R[1:]).sum(1)
    return pnl


def _sharpe(pnl, annualization=252.0):
    x = pnl[1:]
    sd = x.std()
    return float(x.mean() / sd * np.sqrt(annualization)) if sd > 1e-12 else 0.0


def _turnover_series(Wd):
    # Calibrated 2026-09-06 (Bug 1a): Brain reports gross traded weight
    # sum|dW| (E1 syn 33.14 vs real 33.33, 0.6% error); the 0.5 half-turnover
    # convention under-predicted by exactly 2x on price-only legs.
    return np.abs(np.diff(Wd, axis=0)).sum(1)


def _roll_std_plain(x, w, min_obs=5):
    """NaN-aware rolling std for the High-Fidelity impact model (Phase 2).

    Falls back to 0.0 where fewer than min_obs valid obs (documented leniency:
    no impact charged where no volatility can be measured).
    """
    T, N = x.shape
    xm = np.where(np.isfinite(x), x, 0.0)
    x2 = xm * xm
    vm = np.isfinite(x).astype(np.float64)
    cx = np.vstack([np.zeros((1, N)), np.cumsum(xm, axis=0)])
    cx2 = np.vstack([np.zeros((1, N)), np.cumsum(x2, axis=0)])
    cv = np.vstack([np.zeros((1, N)), np.cumsum(vm, axis=0)])
    t = np.arange(T)
    lo = np.maximum(t + 1 - w, 0)
    C = cv[t + 1] - cv[lo]
    S = cx[t + 1] - cx[lo]
    S2 = cx2[t + 1] - cx2[lo]
    mean = np.where(C > 0, S / np.maximum(C, 1e-12), 0.0)
    var = np.maximum(S2 / np.maximum(C, 1e-12) - mean * mean, 0.0)
    return np.where(C >= min_obs, np.sqrt(var), 0.0)


def _high_fidelity_report(Wd, R, adv, exec_cfg, book_size=1.0, annualization=252.0):
    """High-Fidelity execution lens (Phase 2, Agent 1 blueprint). DIAGNOSTIC
    ONLY — never enters gates until calibrated against real shortfall.

    Mechanics (all elementwise, vectorized): participation = |dW|*book/ADV;
    FIFO back-of-queue proxy fill = min(1, max_participation/participation)
    (names with no liquidity print are unfillable unless dW == 0); filled
    trades rebuild an effective weight path; costs = spread*urgency +
    commission (bps on filled notional) + Almgren-Chriss permanent
    (sqrt-law) + temporary (linear-in-rate) from 20d rolling sigma.
    Unfilled weight is dropped and counted as shortfall (working-order
    persistence is out of scope for the daily-bar model).
    """
    T, N = Wd.shape
    dW = np.diff(Wd, axis=0, prepend=np.zeros((1, N)))
    adv_ok = (np.isfinite(adv) & (adv > 0)) if adv is not None else np.zeros_like(dW, dtype=bool)
    need = np.abs(dW) > 1e-12
    part = np.where(need & adv_ok, np.abs(dW) * book_size / np.maximum(adv, 1e-12), 0.0)
    part = np.where(need & ~adv_ok, np.inf, part)
    fill = np.where(need, np.minimum(1.0, float(exec_cfg["max_participation"]) / np.maximum(part, 1e-12)), 1.0)
    dWf = dW * fill
    Wfd = np.cumsum(dWf, axis=0)
    sigma = _roll_std_plain(R, 20)
    pf = np.where(need & adv_ok, np.abs(dWf) * book_size / np.maximum(adv, 1e-12), 0.0)
    bps = (float(exec_cfg["spread_bps"]) * float(exec_cfg["urgency"])
           + float(exec_cfg["commission_bps"])) / 1e4
    imp = (float(exec_cfg["lambda_perm"]) * sigma * np.power(np.maximum(pf, 0.0), float(exec_cfg["alpha"]))
           + float(exec_cfg["eta_temp"]) * sigma * pf)
    cost_cell = np.abs(dWf) * (bps + imp)
    cost = cost_cell.sum(axis=1)
    gross = np.zeros(T)
    gross[1:] = (Wfd[:-1] * R[1:]).sum(axis=1)
    net = gross - cost
    sd = net[1:].std()
    sharpe = float(net[1:].mean() / sd * np.sqrt(annualization)) if sd > 1e-12 else 0.0
    to_f = float(np.abs(dWf[1:]).sum(axis=1).mean()) if T > 1 else 0.0
    invested = book_size / 2.0
    ret_ann = float(net[1:].mean() * annualization / invested) if invested > 0 else 0.0
    fit_floor = 0.125
    fitness = float(sharpe * np.sqrt(abs(ret_ann) / max(to_f, fit_floor)))
    intended = float(np.abs(dW[1:]).sum())
    filled = float(np.abs(dWf[1:]).sum())
    wsum = float(np.abs(Wd).sum())
    wfsum = float(np.abs(Wfd).sum())
    return {
        'net_sharpe': round(sharpe, 4),
        'net_fitness': round(fitness, 4),
        'net_returns_pct': round(ret_ann * 100.0, 4),
        'filled_turnover_pct': round(to_f * 100.0, 4),
        'fill_rate': round(filled / intended, 4) if intended > 1e-12 else 1.0,
        'capped_cell_frac': round(float(((fill < 1.0 - 1e-9) & need).sum() / max(need.sum(), 1)), 4),
        'shortfall_frac': round(1.0 - wfsum / wsum, 4) if wsum > 1e-12 else 0.0,
        'realized_cost_bps': round(float(cost[1:].sum() / filled * 1e4), 4) if filled > 1e-12 else 0.0,
    }


def _fitness(sharpe, ret_ann, turnover, floor=0.125):
    return float(sharpe * np.sqrt(abs(ret_ann) / max(turnover, floor)))


def _metrics_from_weights(Wd, R, book_size=1.0, floor=0.125, annualization=252.0,
                          cost_bps=10.0):
    pnl = _pnl_series(Wd, R)
    x = pnl[1:]
    sharpe = _sharpe(pnl, annualization)
    turns = _turnover_series(Wd)
    turnover = float(turns[1:].mean()) if len(turns) > 1 else 0.0
    invested = book_size / 2.0  # Sec 6 SOURCED: invested == half of book
    ret_ann = float(x.mean() * annualization / invested) if invested > 0 else 0.0
    fitness = _fitness(sharpe, ret_ann, turnover, floor)
    cum = np.cumsum(x)
    dd = float((np.maximum.accumulate(cum) - cum).max() / book_size * 100.0) if len(cum) else 0.0
    total_traded = float(turns[1:].sum() * book_size) if len(turns) > 1 else 0.0
    total_pnl = float(x.sum())
    margin_bps = float(total_pnl / total_traded * 1e4) if total_traded > 1e-12 else 0.0
    # Phase 3 cost lens (reporting ONLY — never enters gates/fitness): linear
    # slippage drag = daily turnover (book fraction) * cost_bps, annualized.
    # Default 10 bps one-way; override via cutoffs {"cost_bps": N}.
    cost_drag = float(turnover * annualization * (cost_bps / 1e4) * 100.0)
    ret_net = float(ret_ann * 100.0 - cost_drag)
    aw = np.abs(Wd)
    mi = np.unravel_index(np.argmax(aw), aw.shape)
    order = np.argsort(aw.ravel())[::-1]
    top = []
    for k in order[:3]:
        tt, nn = int(k // aw.shape[1]), int(k % aw.shape[1])
        if aw[tt, nn] <= 0:
            break
        top.append({'date_idx': tt, 'stock': nn,
                    'weight_pct': round(float(aw[tt, nn]) / book_size * 100.0, 3)})
    return {
        'pnl': pnl,
        'sharpe': sharpe,
        'fitness': fitness,
        'turnover_pct': turnover * 100.0,
        'returns_pct': ret_ann * 100.0,
        'drawdown_pct': dd,
        'margin_bps': margin_bps,
        'weight_concentration_pct': float(aw[mi]) / book_size * 100.0,
        'max_weight_pos': (int(mi[0]), int(mi[1])),
        'top_weights': top,
        'cost_drag_pct': cost_drag,
        'returns_net_pct': ret_net,
    }


def _yearly(pnl, Wd, dates, book_size=1.0, floor=0.125, annualization=252.0):
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    turns = np.concatenate([[0.0], _turnover_series(Wd)])
    out = []
    nuniv = Wd.shape[1]
    for y in sorted(set(years[1:].tolist())):
        idx = np.where(years == y)[0]
        x = pnl[idx]
        sd = x.std()
        sharpe = float(x.mean() / sd * np.sqrt(annualization)) if sd > 1e-12 else 0.0
        turnover = float(turns[idx[1:]].mean()) if len(idx) > 1 else 0.0
        invested = book_size / 2.0
        ret = float(x.mean() * annualization / invested) if invested > 0 else 0.0
        longs, shorts = [], []
        for t in idx:
            if np.abs(Wd[t]).sum() <= 1e-12:
                continue  # warmup / holiday: no book, not thin coverage
            longs.append(int((Wd[t] > 0).sum()))
            shorts.append(int((Wd[t] < 0).sum()))
        out.append({
            'year': int(y),
            'sharpe': round(sharpe, 3),
            'fitness': round(_fitness(sharpe, ret, turnover, floor), 3),
            'turnover_pct': round(turnover * 100.0, 2),
            'returns_pct': round(ret * 100.0, 2),
            'mean_long': round(float(np.mean(longs)), 1) if longs else 0.0,
            'min_long': int(np.min(longs)) if longs else 0,
            'mean_short': round(float(np.mean(shorts)), 1) if shorts else 0.0,
            'min_short': int(np.min(shorts)) if shorts else 0,
            'thin_coverage': bool((np.min(longs) if longs else 0) < max(5, nuniv * 0.01)
                                  or (np.min(shorts) if shorts else 0) < max(5, nuniv * 0.01)),
        })
    return out


def _period_metrics(pnl, Wd, dates, book_size=1.0, floor=0.125, annualization=252.0):
    """IS / TEST / OS splits (Sec 7 SOURCED windows, proportional fallback).

    Published structure: IS ~= 7y->2y ago, most recent year OS. Short panels
    fall back to OS=last 252 sessions, TEST=prior 252, IS=rest.
    """
    T = len(dates)
    if T >= 3 * 252:
        splits = {'IS': (0, T - 2 * 252), 'TEST': (T - 2 * 252, T - 252),
                  'OS': (T - 252, T)}
    else:
        i1, i2 = int(T * 0.6), int(T * 0.8)
        splits = {'IS': (0, i1), 'TEST': (i1, i2), 'OS': (i2, T)}
    out = {}
    for name, (a, b) in splits.items():
        if b - a < 10:
            continue
        sub = pnl[a:b]
        sd = sub[1:].std() if len(sub) > 2 else 0.0
        sharpe = float(sub[1:].mean() / sd * np.sqrt(annualization)) if sd > 1e-12 else 0.0
        turns = _turnover_series(Wd[a:b])
        turnover = float(turns[1:].mean()) if len(turns) > 1 else 0.0
        invested = book_size / 2.0
        ret = float(sub[1:].mean() * annualization / invested) if (len(sub) > 1 and invested > 0) else 0.0
        out[name] = {'sharpe': round(sharpe, 3),
                     'fitness': round(_fitness(sharpe, ret, turnover, floor), 3),
                     'turnover_pct': round(turnover * 100.0, 2),
                     'returns_pct': round(ret * 100.0, 2),
                     'n_days': int(b - a)}
    return out


def _safe_corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else 0.0


def _rolling_self_corr(pnl, rpnl, window_years=2, annualization=252.0):
    """Rolling-window correlation on daily PnL changes (Sec 7).

    Never on cumulative levels. Uses the intersection of overlapping periods
    and the trailing `window_years` window (or full overlap if shorter).

    INVARIANT (audit 2026-09-06, item 5): `pnl`/`rpnl` MUST be daily series
    as produced by _pnl_series (day-t profit per element). Correlating
    cumulative curves would inflate |corr| toward 1 via shared drift; every
    external reimplementation surveyed (xiegengcai DeepWiki ffill-shift,
    QuantML skill README) uses daily changes. Guarded by construction +
    this note; see _pnl_series invariant.
    """
    w = int(window_years * annualization)
    n = min(len(pnl), len(rpnl))
    if n < 10:
        return 0.0, n
    a, b = pnl[-n:], rpnl[-n:]
    if n > w:
        a, b = a[-w:], b[-w:]
    return _safe_corr(a[1:], b[1:]), len(a)


# ---------------------------------------------------------------- entry point
def simulate(expr_str, panel, refs=(), own_refs=(), settings=None, cutoffs=None,
             mode="fast_gate", exec_settings=None):
    """Brain-shaped contract: simulate(expression, settings) -> results (Sec 9).

    `panel` carries market data; `settings` is a SimulationSettings/dict.
    `cutoffs` optionally overrides gate thresholds.
    `mode` selects the execution back-end (Phase 2 dual-mode): "fast_gate"
    runs the frozen Brain-proxy math (bit-parity target across upgrades);
    "high_fidelity" additionally computes fills + AC impact + spread costs as
    a DIAGNOSTIC-ONLY lens (rep['high_fidelity']) that never enters gates.
    `exec_settings` is an ExecutionSettings/dict for High-Fidelity mode.
    """
    st = _coerce_settings(settings)
    rep = {'expression': expr_str, 'settings': st.to_dict()}
    rep['dataset_id'] = getattr(panel, 'dataset_id', '') or ''
    rep['trial_logged'] = False
    if str(mode) not in ("fast_gate", "high_fidelity"):
        raise ValueError(f"mode must be fast_gate|high_fidelity, got {mode!r}")
    rep['mode'] = str(mode)
    exec_cfg = dict(DEFAULT_EXECUTION)
    if exec_settings is not None:
        if hasattr(exec_settings, 'to_dict'):
            exec_cfg.update(exec_settings.to_dict())
        else:
            exec_cfg.update({k: v for k, v in dict(exec_settings).items()})
    verify = str(st.unitHandling).upper() == "VERIFY"
    try:
        sig = _signal_from_expr(expr_str, panel, verify=verify)
    except WQError as e:
        rep.update(error=str(e), passed=False)
        # Errored trials still count toward N_trials (excluding them would
        # understate N and inflate DSR — the documented abuse mode).
        if rep['dataset_id']:
            try:
                import selection as _sel
                if _sel.AUTO_LOG:
                    _sel.TrialRegistry().log(expr_str[:120], rep['dataset_id'],
                                             st.to_dict(), 0.0, 0.0, False)
                    rep['trial_logged'] = True
            except Exception:
                pass
        return rep

    R = panel.fields['returns']
    book = float(get_cutoff("book_size", cutoffs))
    floor = float(get_cutoff("fitness_turnover_floor", cutoffs))
    ann = float(get_cutoff("annualization", cutoffs))

    # stage 3: universe + pasteurization
    umask = _universe_mask(panel, st.universe)
    sig[~umask] = np.nan  # umask is (T, N) point-in-time (item 13)
    if str(st.pasteurization).upper() == "ON":
        sig[~_eligibility_mask(panel)] = np.nan

    # stages 4-5
    sig = _apply_neutralization(sig, panel, st.neutralization)
    sig = _apply_decay(sig, st.decay)

    # stages 6-9
    W = _weights_from_signal(sig, book_size=book, truncation=st.truncation,
                             nan_handling=st.nanHandling)
    Wd = _apply_delay(W, st.delay)
    m = _metrics_from_weights(Wd, R, book_size=book, floor=floor, annualization=ann,
                              cost_bps=float(get_cutoff("cost_bps", cutoffs)))

    # sub-universe check (S1 method, audit 2026-09-06 item 1): pasteurize the
    # full-pipeline signal down to sub-universe names, MARKET-neutralize,
    # rescale to book via the standard weight fn, PnL vs sub-universe returns.
    # (Replaces re-running the full pipeline with its own neutralization on
    # restricted columns.) Cutoff is S1-relative: scales with candidate Sharpe.
    subcols = np.asarray(panel.subuniverse, dtype=bool)
    if subcols.shape[0] != sig.shape[1]:
        subcols = np.ones(sig.shape[1], dtype=bool)
    sig_sub = sig.copy()
    sig_sub[:, ~subcols] = np.nan
    sig_sub = _apply_neutralization(sig_sub, panel, "MARKET")
    Wsub = _weights_from_signal(sig_sub, book_size=book,
                                truncation=st.truncation, nan_handling=st.nanHandling)
    Wsub_d = _apply_delay(Wsub, st.delay)
    subsh = _sharpe(_pnl_series(Wsub_d, R), ann)
    sub_cut = subuniverse_cutoff(int(subcols.sum()),
                                 alpha_size=UNIVERSES.get(str(st.universe).upper(), {}).get("size", sig.shape[1]),
                                 alpha_sharpe=m['sharpe'], overrides=cutoffs)

    # self-correlation (audit 2026-09-06 item 8): gate on rolling 4Y window on
    # daily changes vs reference pool (S1 official-doc scrape + DeepWiki +
    # skill README outrank the older 2Y seminar-notes value); 2Y reported as
    # transition info only. Item 9: escape must beat EVERY ref above cutoff.
    allrefs = [(r['name'], r['pnl'], r['sharpe']) for r in refs]
    allrefs += [(f"{r.get('team', 'own')}:{r['name']}", r['pnl'], r['sharpe']) for r in own_refs]
    maxcorr4, best4, best4_n = 0.0, None, 0
    maxcorr2, best2, best2_n = 0.0, None, 0
    for name, rpnl, rsh in allrefs:
        c4, n4 = _rolling_self_corr(m['pnl'], np.asarray(rpnl), 4.0, ann)
        c2, n2 = _rolling_self_corr(m['pnl'], np.asarray(rpnl), 2.0, ann)
        if abs(c4) > maxcorr4:
            maxcorr4, best4, best4_n = abs(c4), (name, rsh, c4), n4
        if abs(c2) > maxcorr2:
            maxcorr2, best2, best2_n = abs(c2), (name, rsh, c2), n2

    corr_max = float(get_cutoff("self_corr_max", cutoffs))
    improve = float(get_cutoff("corr_sharpe_improve", cutoffs))
    corr_pass = True
    corr_detail = None
    if best4 is not None and maxcorr4 >= corr_max:
        blockers = []
        for name, rpnl, rsh in allrefs:
            c4, n4 = _rolling_self_corr(m['pnl'], np.asarray(rpnl), 4.0, ann)
            if abs(c4) >= corr_max:
                ratio = (m['sharpe'] / rsh) if rsh else 0.0
                ok = bool(m['sharpe'] >= improve * rsh)
                blockers.append({'ref': name, 'ref_sharpe': round(float(rsh), 4),
                                 'corr': round(float(c4), 4), 'ratio': round(float(ratio), 4),
                                 'pass': ok, 'window_days': int(n4)})
                corr_pass = corr_pass and ok
        corr_detail = {'required_ratio': improve, 'window': '4Y',
                       'candidate_sharpe': round(float(m['sharpe']), 4),
                       'blockers': blockers}

    mi = m['max_weight_pos']
    maxw = m['weight_concentration_pct']
    maxw_date = str(panel.dates[mi[0]])
    t = m['turnover_pct'] / 100.0
    sharpe_min = float(get_cutoff("sharpe_min", cutoffs))
    fitness_min = float(get_cutoff("fitness_min", cutoffs))
    to_min = float(get_cutoff("turnover_min", cutoffs))
    to_max = float(get_cutoff("turnover_max", cutoffs))
    conc_max = float(get_cutoff("weight_conc_max", cutoffs))

    criteria = {
        'sharpe': {'value': round(m['sharpe'], 4), 'requirement': f"> {sharpe_min}", 'pass': bool(m['sharpe'] > sharpe_min)},
        'fitness': {'value': round(m['fitness'], 4), 'requirement': f"> {fitness_min}", 'pass': bool(m['fitness'] > fitness_min)},
        'turnover': {'value': round(m['turnover_pct'], 2), 'requirement': f"{to_min * 100:.0f}%-{to_max * 100:.0f}%", 'pass': bool(to_min <= t <= to_max)},
        'weight_concentration': {'value': round(maxw, 2), 'requirement': f"<= {conc_max * 100:.0f}%", 'pass': bool(maxw / 100.0 <= conc_max)},
        'subuniverse_sharpe': {'value': round(subsh, 4), 'requirement': f"> {round(sub_cut, 4)} (0.75*sqrt(sub/alpha)*alpha_Sharpe)", 'pass': bool(subsh > sub_cut)},
        'self_correlation': {'value': round(maxcorr4, 4), 'requirement': f"< {corr_max} (4Y) or Sharpe >= {improve:.2f}x EVERY correlated ref", 'pass': bool(corr_pass)},
    }
    if best4 is not None:
        rep['top_corr_ref'] = best4[0]
        rep['top_corr_ref_sharpe'] = round(float(best4[1]), 4)
        rep['top_corr_signed'] = round(float(best4[2]), 4)
        rep['top_corr_window_days'] = int(best4_n)
        rep['top_corr_window'] = '4Y'
    if best2 is not None:
        rep['self_corr_2y'] = round(maxcorr2, 4)
        rep['self_corr_2y_ref'] = best2[0]
        rep['self_corr_2y_signed'] = round(float(best2[2]), 4)
    if corr_detail is not None:
        rep['self_corr_detail'] = corr_detail

    top_w = []
    for tw in m['top_weights']:
        top_w.append({'date': str(panel.dates[tw['date_idx']]), 'stock': tw['stock'],
                      'weight_pct': tw['weight_pct']})

    rep.update({
        'metrics': {k: round(v, 4) for k, v in m.items()
                    if k not in ('pnl', 'max_weight_pos', 'top_weights')},
        'criteria': criteria,
        'yearly': _yearly(m['pnl'], Wd, panel.dates, book, floor, ann),
        'periods': _period_metrics(m['pnl'], Wd, panel.dates, book, floor, ann),
        'subuniverse_cutoff': round(sub_cut, 4),
        'max_weight_date': maxw_date,
        'max_weight_stock': mi[1],
        'max_weight_dates': top_w,
        'passed': bool(all(c['pass'] for c in criteria.values())),
        'pnl': m['pnl'],
    })
    # Core Operational Invariant 2 — STRICT LINEAGE: log this completed trial
    # (error trials were already logged above; log here exactly once).
    dsid = rep['dataset_id']
    if dsid and not rep['trial_logged']:
        try:
            import selection as _sel
            if _sel.AUTO_LOG:
                _sel.TrialRegistry().log(expr_str[:120], dsid, st.to_dict(),
                                         m['sharpe'], m['fitness'], rep['passed'])
                rep['trial_logged'] = True
        except Exception:
            pass
    # Phase 2 dual-mode: High-Fidelity lens shares the Fast-Gate front-end
    # (same Wd order intent) and adds fills + impact + spread diagnostics.
    # Diagnostic-only: criteria/gates above are untouched in both modes.
    if str(mode) == "high_fidelity":
        rep['high_fidelity'] = _high_fidelity_report(
            Wd, R, panel.fields.get('adv20', None), exec_cfg,
            book_size=book, annualization=ann)
        rep['exec_settings'] = {k: exec_cfg[k] for k in sorted(exec_cfg)}
    return rep


def failed_criteria(rep):
    if rep.get('error'):
        return ['parse_error']
    return [k for k, v in rep['criteria'].items() if not v['pass']]


def build_reference_pool(panel, named_exprs, settings=None):
    pool = []
    for name, expr in named_exprs:
        try:
            rep = simulate(expr, panel, (), (), settings=settings)
            if rep.get('error'):
                continue
            pool.append({'name': name, 'expr': expr, 'pnl': rep['pnl'],
                         'sharpe': rep['metrics']['sharpe']})
        except WQError:
            continue
    return pool


def save_report(rep, path):
    slim = {k: v for k, v in rep.items() if k != 'pnl'}
    with open(path, 'w') as f:
        json.dump(slim, f, indent=2, default=str)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
