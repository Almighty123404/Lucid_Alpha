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
from config import (SimulationSettings, DEFAULT_SETTINGS, CUTOFFS as _CFG_CUTOFFS,
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


def _universe_mask(panel, universe):
    """Top-N by trailing average dollar volume (Sec 2.1, APPROX).

    Static membership from full-period mean adv20 (or |close*volume|);
    smaller tiers nest inside larger ones. Unknown names -> all eligible.
    """
    N = panel.fields['returns'].shape[1]
    size = UNIVERSES.get(str(universe).upper(), {}).get("size", N)
    if size is None or size >= N:
        return np.ones(N, dtype=bool)
    adv = panel.fields.get('adv20', None)
    if adv is None:
        dv = np.abs(panel.fields['close'] * panel.fields.get('volume', 1.0))
        liq = np.nanmean(np.where(np.isfinite(dv), dv, np.nan), axis=0)
    else:
        liq = np.nanmean(np.where(np.isfinite(adv), adv, np.nan), axis=0)
    liq = np.where(np.isfinite(liq), liq, -1.0)
    thresh = np.sort(liq)[-size]
    return liq >= thresh


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
    w = int(n)
    xm = np.where(np.isfinite(sig), sig, 0.0)
    vm = np.isfinite(sig).astype(np.float64)
    num = np.zeros_like(xm)
    den = np.zeros_like(vm)
    for i in range(w):
        wt = w - i
        src_n = np.zeros_like(xm)
        src_d = np.zeros_like(vm)
        if i < xm.shape[0]:
            src_n[i:] = xm[:xm.shape[0] - i]
            src_d[i:] = vm[:vm.shape[0] - i]
        num += wt * src_n
        den += wt * src_d
    out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    out[:w - 1] = np.nan
    return out


# ---------------------------------------------------------------- stage 6-9
def _weights_from_signal(sig, min_names=MIN_NAMES, book_size=1.0,
                         truncation=0.0, nan_handling="ON"):
    """Demean + scale to gross book, with truncation clip + renormalize."""
    T, N = sig.shape
    W = np.zeros_like(sig)
    prev = np.zeros(N)
    cap = abs(truncation) * book_size if truncation else 0.0
    for t in range(T):
        row = sig[t]
        m = np.isfinite(row)
        if m.sum() < min_names:
            W[t] = prev
            continue
        x = row[m].astype(np.float64)
        if str(nan_handling).upper() == "ON":
            # APPROX: group/cross-sectional mean imputation ~= demean (no-op
            # after centering); missing names get zero active weight via mask.
            pass
        x = x - x.mean()
        s = np.abs(x).sum()
        if s < 1e-12:
            W[t] = prev
            continue
        w = np.zeros(N)
        w[m] = x / s * book_size
        if cap > 0:
            w = np.clip(w, -cap, cap)
            g = np.abs(w).sum()
            if g > 1e-12:
                w *= book_size / g  # renormalize so gross exposure preserved
        W[t] = w
        prev = w
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
    T = Wd.shape[0]
    pnl = np.zeros(T)
    pnl[1:] = (Wd[:-1] * R[1:]).sum(1)
    return pnl


def _sharpe(pnl, annualization=252.0):
    x = pnl[1:]
    sd = x.std()
    return float(x.mean() / sd * np.sqrt(annualization)) if sd > 1e-12 else 0.0


def _turnover_series(Wd):
    return 0.5 * np.abs(np.diff(Wd, axis=0)).sum(1)


def _fitness(sharpe, ret_ann, turnover, floor=0.125):
    return float(sharpe * np.sqrt(abs(ret_ann) / max(turnover, floor)))


def _metrics_from_weights(Wd, R, book_size=1.0, floor=0.125, annualization=252.0):
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
def simulate(expr_str, panel, refs=(), own_refs=(), settings=None, cutoffs=None):
    """Brain-shaped contract: simulate(expression, settings) -> results (Sec 9).

    `panel` carries market data; `settings` is a SimulationSettings/dict.
    `cutoffs` optionally overrides gate thresholds.
    """
    st = _coerce_settings(settings)
    rep = {'expression': expr_str, 'settings': st.to_dict()}
    verify = str(st.unitHandling).upper() == "VERIFY"
    try:
        sig = _signal_from_expr(expr_str, panel, verify=verify)
    except WQError as e:
        rep.update(error=str(e), passed=False)
        return rep

    R = panel.fields['returns']
    book = float(get_cutoff("book_size", cutoffs))
    floor = float(get_cutoff("fitness_turnover_floor", cutoffs))
    ann = float(get_cutoff("annualization", cutoffs))

    # stage 3: universe + pasteurization
    umask = _universe_mask(panel, st.universe)
    sig[:, ~umask] = np.nan
    if str(st.pasteurization).upper() == "ON":
        sig[~_eligibility_mask(panel)] = np.nan

    # stages 4-5
    sig = _apply_neutralization(sig, panel, st.neutralization)
    sig = _apply_decay(sig, st.decay)

    # stages 6-9
    W = _weights_from_signal(sig, book_size=book, truncation=st.truncation,
                             nan_handling=st.nanHandling)
    Wd = _apply_delay(W, st.delay)
    m = _metrics_from_weights(Wd, R, book_size=book, floor=floor, annualization=ann)

    # sub-universe re-evaluation (restricted columns, same pipeline)
    subcols = np.asarray(panel.subuniverse, dtype=bool)
    if subcols.shape[0] != sig.shape[1]:
        subcols = np.ones(sig.shape[1], dtype=bool)
    Wsub = _weights_from_signal(sig[:, subcols], book_size=book,
                                truncation=st.truncation, nan_handling=st.nanHandling)
    Wsub_d = _apply_delay(Wsub, st.delay)
    subsh = _sharpe(_pnl_series(Wsub_d, R[:, subcols]), ann)
    sub_cut = subuniverse_cutoff(int(subcols.sum()),
                                 largest_size=int(get_cutoff("largest_universe_size", cutoffs)),
                                 delay=int(st.delay), overrides=cutoffs)

    # self-correlation: rolling 2Y window on daily changes vs reference pool
    win_yrs = float(get_cutoff("self_corr_window_years", cutoffs))
    allrefs = [(r['name'], r['pnl'], r['sharpe']) for r in refs]
    allrefs += [(f"{r.get('team', 'own')}:{r['name']}", r['pnl'], r['sharpe']) for r in own_refs]
    maxcorr, best, best_n = 0.0, None, 0
    for name, rpnl, rsh in allrefs:
        c, n = _rolling_self_corr(m['pnl'], np.asarray(rpnl), win_yrs, ann)
        if abs(c) > maxcorr:
            maxcorr, best, best_n = abs(c), (name, rsh, c), n

    corr_max = float(get_cutoff("self_corr_max", cutoffs))
    improve = float(get_cutoff("corr_sharpe_improve", cutoffs))
    corr_pass = True
    corr_detail = None
    if best is not None and maxcorr >= corr_max:
        ratio = (m['sharpe'] / best[1]) if best[1] else 0.0
        corr_pass = bool(m['sharpe'] >= improve * best[1])
        corr_detail = {'ref': best[0], 'ref_sharpe': round(float(best[1]), 4),
                       'corr': round(float(best[2]), 4),
                       'candidate_sharpe': round(float(m['sharpe']), 4),
                       'ratio': round(float(ratio), 4),
                       'required_ratio': improve,
                       'window_days': int(best_n)}

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
        'subuniverse_sharpe': {'value': round(subsh, 4), 'requirement': f"> {round(sub_cut, 4)}", 'pass': bool(subsh > sub_cut)},
        'self_correlation': {'value': round(maxcorr, 4), 'requirement': f"< {corr_max} or Sharpe >= {improve:.2f}x correlated ref", 'pass': bool(corr_pass)},
    }
    if best is not None:
        rep['top_corr_ref'] = best[0]
        rep['top_corr_ref_sharpe'] = round(float(best[1]), 4)
        rep['top_corr_signed'] = round(float(best[2]), 4)
        rep['top_corr_window_days'] = int(best_n)
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
