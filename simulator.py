import copy
import json
import warnings

import numpy as np

from fastexpr import parse, eval_node, Env, WQError

warnings.filterwarnings('ignore')

CUTOFFS = {
    'sharpe_min': 1.25,
    'fitness_min': 1.0,
    'turnover_min': 0.01,
    'turnover_max': 0.70,
    'weight_conc_max': 0.10,
    'subuniverse_sharpe_min': 0.80,
    'self_corr_max': 0.70,
    'corr_sharpe_improve': 1.10,
}

MIN_NAMES = 5


def _shift_rows(W):
    out = np.zeros_like(W)
    out[1:] = W[:-1]
    return out


def _weights_from_signal(sig, min_names=MIN_NAMES):
    T, N = sig.shape
    W = np.zeros_like(sig)
    prev = np.zeros(N)
    for t in range(T):
        row = sig[t]
        m = np.isfinite(row)
        if m.sum() < min_names:
            W[t] = prev
            continue
        x = row[m]
        x = x - x.mean()
        s = np.abs(x).sum()
        if s < 1e-12:
            W[t] = prev
            continue
        w = np.zeros(N)
        w[m] = x / s
        W[t] = w
        prev = w
    return W


def _pnl_series(W, R):
    T = W.shape[0]
    pnl = np.zeros(T)
    pnl[1:] = (W[:-1] * R[1:]).sum(1)
    return pnl


def _sharpe(pnl):
    x = pnl[1:]
    sd = x.std()
    return float(x.mean() / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0


def _turnover_series(W):
    return 0.5 * np.abs(np.diff(W, axis=0)).sum(1)


def _fitness(sharpe, ret_ann, turnover):
    return float(sharpe * np.sqrt(abs(ret_ann) / max(turnover, 0.125)))


def _metrics_from_weights(W, R):
    pnl = _pnl_series(W, R)
    x = pnl[1:]
    sd = x.std()
    sharpe = _sharpe(pnl)
    turns = _turnover_series(W)
    turnover = float(turns[1:].mean()) if len(turns) > 1 else 0.0
    ret_ann = float(x.mean() * 252.0)
    fitness = _fitness(sharpe, ret_ann, turnover)
    cum = np.cumsum(x)
    dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
    aw = np.abs(W)
    mi = np.unravel_index(np.argmax(aw), aw.shape)
    return {
        'pnl': pnl,
        'sharpe': sharpe,
        'fitness': fitness,
        'turnover_pct': turnover * 100.0,
        'returns_pct': ret_ann * 100.0,
        'drawdown_pct': dd * 100.0,
        'weight_concentration_pct': float(aw[mi]) * 100.0,
        'max_weight_pos': (int(mi[0]), int(mi[1])),
    }


def _yearly(pnl, W, dates):
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    turns = np.concatenate([[0.0], _turnover_series(W)])
    out = []
    for y in sorted(set(years[1:].tolist())):
        idx = np.where(years == y)[0]
        x = pnl[idx]
        sd = x.std()
        sharpe = float(x.mean() / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0
        turnover = float(turns[idx[1:]].mean()) if len(idx) > 1 else 0.0
        ret = float(x.mean() * 252.0)
        out.append({
            'year': int(y),
            'sharpe': round(sharpe, 3),
            'fitness': round(_fitness(sharpe, ret, turnover), 3),
            'turnover_pct': round(turnover * 100.0, 2),
            'returns_pct': round(ret * 100.0, 2),
        })
    return out


def _safe_corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else 0.0


def _signal_from_expr(expr_str, panel):
    ast = parse(expr_str)
    sig = eval_node(ast, Env(panel))
    if not (isinstance(sig, np.ndarray) and sig.ndim == 2):
        raise WQError("Expression does not produce a cross-sectional matrix")
    return np.where(np.isinf(sig), np.nan, sig).astype(np.float64)


def simulate(expr_str, panel, refs=(), own_refs=()):
    rep = {'expression': expr_str}
    try:
        sig = _signal_from_expr(expr_str, panel)
    except WQError as e:
        rep.update(error=str(e), passed=False)
        return rep

    R = panel.fields['returns']
    W = _weights_from_signal(sig)
    m = _metrics_from_weights(W, R)

    Wsub = _weights_from_signal(sig[:, panel.subuniverse])
    subsh = _sharpe(_pnl_series(Wsub, R[:, panel.subuniverse]))

    allrefs = [(r['name'], r['pnl'], r['sharpe']) for r in refs]
    allrefs += [(f"{r.get('team', 'own')}:{r['name']}", r['pnl'], r['sharpe']) for r in own_refs]
    maxcorr, best = 0.0, None
    for name, rpnl, rsh in allrefs:
        c = _safe_corr(m['pnl'][1:], rpnl[1:])
        if abs(c) > maxcorr:
            maxcorr, best = abs(c), (name, rsh)

    corr_pass = True
    if best is not None and maxcorr >= CUTOFFS['self_corr_max']:
        corr_pass = m['sharpe'] >= CUTOFFS['corr_sharpe_improve'] * best[1]

    mi = m['max_weight_pos']
    maxw = m['weight_concentration_pct']
    maxw_date = str(panel.dates[mi[0]])
    t = m['turnover_pct'] / 100.0

    criteria = {
        'sharpe': {'value': round(m['sharpe'], 4), 'requirement': f"> {CUTOFFS['sharpe_min']}", 'pass': bool(m['sharpe'] > CUTOFFS['sharpe_min'])},
        'fitness': {'value': round(m['fitness'], 4), 'requirement': f"> {CUTOFFS['fitness_min']}", 'pass': bool(m['fitness'] > CUTOFFS['fitness_min'])},
        'turnover': {'value': round(m['turnover_pct'], 2), 'requirement': f"{CUTOFFS['turnover_min'] * 100:.0f}%-{CUTOFFS['turnover_max'] * 100:.0f}%", 'pass': bool(CUTOFFS['turnover_min'] <= t <= CUTOFFS['turnover_max'])},
        'weight_concentration': {'value': round(maxw, 2), 'requirement': f"<= {CUTOFFS['weight_conc_max'] * 100:.0f}%", 'pass': bool(maxw / 100.0 <= CUTOFFS['weight_conc_max'])},
        'subuniverse_sharpe': {'value': round(subsh, 4), 'requirement': f"> {CUTOFFS['subuniverse_sharpe_min']}", 'pass': bool(subsh > CUTOFFS['subuniverse_sharpe_min'])},
        'self_correlation': {'value': round(maxcorr, 4), 'requirement': f"< {CUTOFFS['self_corr_max']} or Sharpe >= {CUTOFFS['corr_sharpe_improve']:.2f}x correlated ref", 'pass': bool(corr_pass)},
    }
    if best is not None:
        rep['top_corr_ref'] = best[0]
        rep['top_corr_ref_sharpe'] = round(float(best[1]), 4)

    rep.update({
        'metrics': {k: round(v, 4) for k, v in m.items() if k not in ('pnl', 'max_weight_pos')},
        'criteria': criteria,
        'yearly': _yearly(m['pnl'], W, panel.dates),
        'max_weight_date': maxw_date,
        'max_weight_stock': mi[1],
        'passed': bool(all(c['pass'] for c in criteria.values())),
        'pnl': m['pnl'],
    })
    return rep


def failed_criteria(rep):
    if rep.get('error'):
        return ['parse_error']
    return [k for k, v in rep['criteria'].items() if not v['pass']]


def build_reference_pool(panel, named_exprs):
    pool = []
    for name, expr in named_exprs:
        try:
            sig = _signal_from_expr(expr, panel)
            W = _weights_from_signal(sig)
            pnl = _pnl_series(W, panel.fields['returns'])
            pool.append({'name': name, 'expr': expr, 'pnl': pnl, 'sharpe': _sharpe(pnl)})
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
