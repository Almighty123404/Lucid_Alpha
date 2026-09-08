import json
import numpy as np
import os

from fastexpr import (parse, parse_program, to_expr, WQError, field_names, backfilled_fields, has_division,
                      bump_windows, add_backfill, swap_fields, wrap_ts_rank, wrap_decay,
                      wrap_group_neutralize, wrap_ts_delta, contains_group_neutralize,
                      wrap_trade_when, wrap_hump, wrap_truncate, wrap_regression_neut,
                      iter_nodes, Call, Bin, Neg, Field, Num, Assign, Env, eval_node, eval_program)
from simulator import simulate, failed_criteria

SPARSE_TOKENS = ('nws', 'snt', 'buzz', 'news', 'implied_volatility', 'est')

SIBLINGS = {
    'implied_volatility_call_10': 'implied_volatility_call_60',
    'implied_volatility_call_60': 'implied_volatility_put_60',
    'implied_volatility_put_10': 'implied_volatility_put_60',
    'implied_volatility_put_60': 'implied_volatility_call_60',
    'nws12_afterhsz_01l': 'buzz',
    'buzz': 'nws12_afterhsz_01l',
    'returns': 'close',
}


class Idea:
    def __init__(self, name, expr, rationale):
        self.name = name
        self.expr = expr
        self.rationale = rationale


# ---------------------------------------------------------------------------
# Audit 2026-09-06, Phase A (items 6/7/3 of the 14-remainder):
# A1 fingerprint/skeleton novelty gate (ports wq-alpha-agent fingerprint idea):
# A2 dimensional-consistency table (Alpha-squared-style pruning);
# A3 parsimony score (GP/AlphaAgent practice: prefer shallower on ties).
# ---------------------------------------------------------------------------

def _skel(node):
    if isinstance(node, Num):
        return '#'
    if isinstance(node, Field):
        return 'F'
    if isinstance(node, Neg):
        return '(-' + _skel(node.x) + ')'
    if isinstance(node, Bin):
        return '(' + _skel(node.l) + node.op + _skel(node.r) + ')'
    if isinstance(node, Call):
        return node.name + '(' + ','.join(_skel(a) for a in node.args) + ')'
    return '?'


def skeleton_of(expr_str):
    """Normalized structural skeleton: numbers -> #, fields -> F.

    Catches param-only duplicates (same skeleton, different literals/fields)
    that exact-string tried-sets miss. Returns None if unparseable.
    """
    try:
        return _skel(parse(expr_str))
    except WQError:
        return None


def window_tuple(expr_str):
    """Sorted int literals in the expression (param vector for dup detection)."""
    try:
        node = parse(expr_str)
    except WQError:
        return ()
    out = []
    for n in iter_nodes(node):
        if isinstance(n, Num) and isinstance(n.v, int):
            out.append(n.v)
    return tuple(sorted(out))


def node_count(expr_str):
    """AST size (parsimony score: fewer nodes = simpler). -1 if unparseable."""
    try:
        return sum(1 for _ in iter_nodes(parse(expr_str)))
    except WQError:
        return -1


# Phase B7: hypothesis-leg alignment scoring (AlphaAgent-style, rule-based v1).
# Advisory only: fraction of identifiable legs whose data family is named in
# the hypothesis text. Families inferred from field tokens per leg.
ALIGN_KEYWORDS = {
    'sentiment': ('sentiment', 'news', 'buzz', 'media', 'attention', 'nws', 'story', 'headline'),
    'fundamental': ('earning', 'ebitda', 'margin', 'profit', 'sales', 'revenue',
                    'leverage', 'debt', 'quality', 'fundamental', 'cash', 'asset',
                    'estimate', 'revision', 'analyst'),
    'microstructure': ('reversal', 'volume', 'liquidity', 'microstructure', 'turnover',
                       'mean-reversion', 'mean reversion', 'short-term', 'loser', 'bounce'),
    'derivatives': ('option', 'iv ', 'iv-', 'skew', 'put', 'call', 'fear', 'crash',
                    'volatility', 'vol ', 'implied', 'derivative'),
    'momentum': ('momentum', 'trend', 'drift', 'persistence', 'continuation'),
}
FIELD_FAMILY_TOKENS = {
    'sentiment': ('nws', 'snt', 'buzz', 'news'),
    'fundamental': ('ebitda', 'ebit', 'sales', 'debt', 'assets', 'liabilit', 'est',
                    'cogs', 'gross_profit', 'income', 'expense', 'eps', 'equity',
                    'cash', 'retained', 'goodwill', 'working_capital',
                    'capex', 'dividends', 'tax', 'margin', 'revenue', 'fcf',
                    'ocf', 'analyst', 'recommendation', 'surprise', 'segment',
                    'ni', 'total_debt'),
    'microstructure': ('returns', 'close', 'open', 'high', 'low', 'volume', 'adv20', 'cap',
                       'vwap', 'shares_out', 'short_interest', 'days_to_cover',
                       'borrow_fee', 'insider', 'intraday'),
    'derivatives': ('implied_volatility', 'iv_', 'hv_', 'historical', 'put_call', 'pcr',
                    'opt_open', 'skew', 'option'),
}


def _leg_family(leg_expr):
    # Longest-match voting: each field votes once, for the family owning its
    # longest matching token. Prevents substring collisions ('cap' ⊂ 'capex',
    # 'open' ⊂ 'opt_open_interest') from double-counting across families.
    try:
        names = {n.name for n in iter_nodes(parse(leg_expr)) if isinstance(n, Field)}
    except WQError:
        return None
    scores = {}
    for nm in names:
        best, bestlen = [], -1
        for fam, toks in FIELD_FAMILY_TOKENS.items():
            ml = max((len(t) for t in toks if t in nm), default=-1)
            if ml > bestlen:
                best, bestlen = [fam], ml
            elif ml == bestlen and ml >= 0:
                best.append(fam)
        for fam in best:
            scores[fam] = scores.get(fam, 0) + 1
    if not scores:
        return None
    return max(scores, key=lambda f: scores[f])


def alignment_score(hypothesis, leg_exprs):
    """Fraction of identifiable legs covered by the hypothesis (None if none).

    Advisory metric only — never gates. A low score means the story does not
    name the data it trades; a high score does not mean the story is true.
    """
    hy = (hypothesis or '').lower()
    fams = [_leg_family(le) for le in leg_exprs]
    fams = [f for f in fams if f is not None]
    if not fams:
        return None
    hit = sum(1 for f in fams
              if any(kw in hy for kw in ALIGN_KEYWORDS.get(f, (f,))))
    return round(hit / len(fams), 3)


def template_family_balance(templates, seed=0):
    """Count data families across one sampled instance per template.

    Advisory (ChenNachuan field-sampling balancer): when the pool exceeds ~20
    templates, price-volume families tend to dominate search; balance with
    Log+MinMax+Softmax sampling. Below that threshold the balancer stays
    dormant — reported, not enforced.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    counts = {}
    for name, builder, rationale in templates:
        try:
            expr = builder(rng)
        except TypeError:
            expr = builder
        fam = _leg_family(expr) or 'unknown'
        counts[fam] = counts.get(fam, 0) + 1
    total = sum(counts.values())
    return {'counts': counts, 'total': total,
            'balancer': 'active' if total > 20 else 'dormant (<=20 templates)'}


# Dimensional classes: CCY (money-like), COUNT (shares/volume), LESS
# (dimensionless: returns, scores, vols, ratios), GROUP (sector/industry maps),
# VECTOR (news-vector fields), BOOL (comparisons), DERIVED (products/ratios).
DIMENSIONS = {
    'open': 'CCY', 'high': 'CCY', 'low': 'CCY', 'close': 'CCY',
    'ebitda': 'CCY', 'sales': 'CCY', 'debt': 'CCY', 'assets': 'CCY',
    'cashflow_op': 'CCY',
    'liabilities_curr': 'CCY', 'assets_curr': 'CCY', 'cap': 'CCY', 'adv20': 'CCY',
    'volume': 'COUNT',
    'returns': 'LESS', 'buzz': 'LESS',
    'vwap': 'CCY', 'shares_out': 'COUNT', 'adv60': 'CCY',
    'cogs': 'CCY', 'gross_profit': 'CCY', 'operating_income': 'CCY',
    'net_income': 'CCY', 'eps': 'CCY', 'tax_expense': 'CCY', 'liabilities': 'CCY',
    'equity': 'CCY', 'cash_and_equiv': 'CCY', 'retained_earnings': 'CCY',
    'goodwill': 'CCY', 'working_capital': 'CCY', 'operating_cash_flow': 'CCY',
    'capex': 'CCY', 'free_cash_flow': 'CCY', 'dividends_paid': 'CCY',
    'return_assets': 'LESS',
    'operating_expense': 'CCY', 'ebit': 'CCY',
    'est_eps': 'CCY', 'est_revenue': 'CCY', 'est_eps_std': 'CCY',
    'recommendation': 'LESS', 'eps_surprise': 'LESS',
    'snt_news': 'LESS', 'snt_social': 'LESS', 'news_volume': 'COUNT',
    'iv_10': 'LESS', 'iv_30': 'LESS', 'hv_20': 'LESS', 'put_call_ratio': 'LESS',
    'opt_open_interest': 'COUNT', 'short_interest': 'COUNT',
    'days_to_cover': 'LESS', 'borrow_fee': 'LESS',
    'insider_buying': 'CCY', 'insider_selling': 'CCY',
    'revenue': 'CCY', 'op_income': 'CCY', 'ni': 'CCY', 'total_debt': 'CCY',
    'ocf': 'CCY', 'fcf': 'CCY', 'pcr': 'LESS',
    'implied_volatility_10': 'LESS', 'implied_volatility_30': 'LESS',
    'historical_volatility_20': 'LESS',
    'implied_volatility_call_10': 'LESS', 'implied_volatility_call_60': 'LESS',
    'implied_volatility_put_10': 'LESS', 'implied_volatility_put_60': 'LESS',
    'implied_volatility_call_720': 'LESS', 'implied_volatility_put_720': 'LESS',
    'nws12_afterhsz_01l': 'VECTOR', 'nan': 'LESS',
    'sector': 'GROUP', 'industry': 'GROUP', 'subindustry': 'GROUP',
    'country': 'GROUP', 'exchange': 'GROUP', 'market': 'GROUP',
    'nws12_afterhsz_01l': 'VECTOR', 'scl12_alltype_buzzvec': 'VECTOR', 'scl12_buzzvec': 'VECTOR', 'analyst_eps_estimates': 'VECTOR',
    'option_implied_vol_surface': 'VECTOR', 'segment_revenue': 'VECTOR',
    'price_volume_intraday': 'VECTOR',
}

_LESS_OUT = {'rank', 'quantile', 'zscore', 'normalize', 'winsorize', 'scale',
             'ts_zscore', 'ts_rank', 'ts_covariance', 'ts_corr', 'ts_regression',
             'group_rank', 'group_zscore', 'group_scale', 'bucket', 'densify',
             'ts_count_nans', 'ts_arg_max', 'ts_arg_min', 'days_from_last_change',
             'kth_element', 'last_diff_value', 'is_nan', 'not'}
_ADD_SUB = {'add', 'subtract'}


def _dim_of(node, scope=()):
    if isinstance(node, Num):
        return 'NUM'
    if isinstance(node, Field):
        if node.name in scope:
            return 'LESS'
        return DIMENSIONS.get(node.name, 'LESS')
    if isinstance(node, Neg):
        return _dim_of(node.x, scope)
    if isinstance(node, Bin):
        # Always validate children first: free operators (*, /, comparisons)
        # must not hide nested additive mismatches (e.g. rank(close+volume)).
        dl, dr = _dim_of(node.l, scope), _dim_of(node.r, scope)
        if node.op in ('+', '-'):
            if 'NUM' in (dl, dr):
                return dl if dr == 'NUM' else dr
            if dl != dr:
                raise WQError(f"dimension mismatch: {dl} {node.op} {dr}")
            return dl
        if node.op in ('<', '>', '<=', '>=', '==', '!=', '&', '|'):
            return 'BOOL'
        return 'DERIVED'
    if isinstance(node, Call):
        if node.name in _LESS_OUT:
            for a in node.args:  # still type-check args (catches rank(close+volume))
                _dim_of(a, scope)
            return 'LESS'
        if node.name in ('ts_mean', 'ts_std_dev', 'ts_sum', 'ts_product',
                         'ts_decay_linear', 'ts_delay', 'ts_backfill', 'ts_scale',
                         'ts_av_diff', 'ts_quantile', 'group_mean', 'group_backfill',
                         'abs', 'log', 'sign', 'sqrt', 'hump', 'truncate',
                         'inverse', 'reverse', 'signed_power', 'max', 'min',
                         'add', 'subtract', 'divide', 'multiply', 'power'):
            if node.name in _ADD_SUB or (node.name in ('max', 'min') and len(node.args) == 2):
                ds = [_dim_of(a, scope) for a in node.args[:2]]
                ds = [d for d in ds if d != 'NUM']
                if len(ds) == 2 and ds[0] != ds[1] and 'DERIVED' not in ds and 'BOOL' not in ds:
                    raise WQError(f"dimension mismatch in {node.name}: {ds[0]} vs {ds[1]}")
            if node.name in ('vec_avg', 'vec_mean', 'vec_sum', 'vec_max', 'vec_min', 'vec_std'):
                for a in node.args:
                    if _dim_of(a, scope) != 'VECTOR':
                        raise WQError(f"{node.name} expects a Vector field")
                return 'LESS'
            if node.name == 'group_neutralize' and len(node.args) >= 2:
                if _dim_of(node.args[1], scope) != 'GROUP':
                    raise WQError("group_neutralize expects a Group field")
            if node.name in ('trade_when', 'if_else'):
                if node.args and _dim_of(node.args[0], scope) not in ('BOOL', 'LESS', 'NUM'):
                    raise WQError(f"{node.name} condition must be boolean-like")
            inner = [_dim_of(a, scope) for a in node.args if not isinstance(a, Num)]
            inner = [d for d in inner if d not in ('GROUP', 'BOOL')]
            return inner[0] if inner else 'LESS'
        if node.name == 'group_neutralize' and len(node.args) >= 2:
            if _dim_of(node.args[1], scope) != 'GROUP':
                raise WQError("group_neutralize expects a Group field")
            return _dim_of(node.args[0], scope)
        if node.name in ('trade_when', 'if_else') and node.args:
            if _dim_of(node.args[0], scope) not in ('BOOL', 'LESS', 'NUM'):
                raise WQError(f"{node.name} condition must be boolean-like")
            if node.name == 'if_else' and len(node.args) == 3:
                branches = [_dim_of(node.args[1], scope), _dim_of(node.args[2], scope)]
                branches = [d for d in branches if d != 'NUM']
                if len(branches) == 2 and branches[0] != branches[1]:
                    raise WQError("if_else branches must have matching dimensions")
            return 'LESS'
        if node.name in ('vec_avg', 'vec_mean', 'vec_sum', 'vec_max', 'vec_min', 'vec_std'):
            for a in node.args:
                if _dim_of(a, scope) != 'VECTOR':
                    raise WQError(f"{node.name} expects a Vector field")
            return 'LESS'
        if node.name == 'vec_choose' and node.args:
            if _dim_of(node.args[0], scope) != 'VECTOR':
                raise WQError("vec_choose expects a Vector field as first arg")
            return 'LESS'
        for a in node.args:  # unknown ops: still validate args, result free
            _dim_of(a, scope)
        return 'DERIVED'
    if isinstance(node, Assign):
        return _dim_of(node.expr, scope)
    return 'LESS'


def dimension_check(expr_str):
    """Return list of dimensional violations ([] = clean, None = unchecked).

    Only additive (+/-) mismatches and group/vector misplacements are checked;
    products/ratios/comparisons are free (they produce derived/boolean dims).
    Lenient by design: unknown fields default to dimensionless.
    """
    try:
        bindings, final = parse_program(expr_str)
    except WQError:
        return None
    try:
        scope = []
        for binding in bindings:
            _dim_of(binding.expr, scope)
            scope.append(binding.name)
        _dim_of(final, scope)
    except WQError as e:
        return [str(e)]
    return []


# ---------------------------------------------------------------------------
# Per-component diagnostic discipline.
# Every candidate building block (a single field, or a single operator applied
# to a single field) must be simulated standalone BEFORE it is combined with
# anything else. Combinations must state a hypothesis, justify add vs multiply,
# and beat the best individual component on evidence.
# ---------------------------------------------------------------------------

def _signal_matrix(expr_str, panel):
    try:
        sig = eval_program(expr_str, Env(panel))
    except WQError:
        return None
    if not (isinstance(sig, np.ndarray) and sig.ndim == 2):
        return None
    return np.where(np.isinf(sig), np.nan, sig).astype(np.float64)


def _coverage_by_year(sig, dates):
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    out = {}
    for y in sorted(set(years.tolist())):
        idx = np.where(years == y)[0]
        longs, shorts = [], []
        for t in idx:
            row = sig[t]
            m = np.isfinite(row)
            if m.sum() < 5:
                continue  # warmup / holiday: no book, not thin coverage
            v = row[m]
            mu = v.mean()
            longs.append(int((v > mu).sum()))
            shorts.append(int((v < mu).sum()))
        if not longs:
            out[int(y)] = {'mean_long': 0.0, 'min_long': 0, 'mean_short': 0.0, 'min_short': 0}
        else:
            out[int(y)] = {
                'mean_long': round(float(np.mean(longs)), 1),
                'min_long': int(np.min(longs)),
                'mean_short': round(float(np.mean(shorts)), 1),
                'min_short': int(np.min(shorts)),
            }
    return out


def _curve_sentence(yearly, overall_sharpe):
    if not yearly:
        return "no yearly data available."
    sh = [y['sharpe'] for y in yearly]
    yrs = [y['year'] for y in yearly]
    if all(s > 0.5 for s in sh) and float(np.std(sh)) < 1.0:
        return f"steady contributor — positive every year ({min(yrs)}–{max(yrs)}), low dispersion."
    if sh[0] == max(sh) and sh[-1] == min(sh) and sh[0] - sh[-1] > 1.0:
        return "front-loaded/decaying — early years strong, fading later; edge may be arbitraged away."
    if all(sh[i] >= sh[i + 1] - 1e-9 for i in range(len(sh) - 1)) and sh[0] - sh[-1] > 0.5:
        return "decaying — monotonic decline year over year."
    if min(sh) < 0 and max(sh) > 1.0:
        return "regime-dependent — strong in some years, negative in others; works only in certain markets."
    if abs(overall_sharpe) < 0.5:
        return "choppy/flat — no persistent direction; noise dominates."
    if sum(1 for s in sh if s > 1.0) == 1 and overall_sharpe > 0.5:
        return "single-year-propped — one good year carries the aggregate; fragile."
    return "mixed — uneven across years without a clean trend."


def diagnose_component(expr_str, panel, refs=(), own_refs=(), settings=None):
    rep = simulate(expr_str, panel, refs, own_refs, settings=settings)
    if rep.get('error'):
        return {'expr': expr_str, 'error': rep['error'], 'keep': False,
                'reason': 'simulation error; discard.'}
    m = rep['metrics']
    yearly = rep.get('yearly', [])
    sig = _signal_matrix(expr_str, panel)
    cov = _coverage_by_year(sig, panel.dates) if sig is not None else {}
    curve = _curve_sentence(yearly, m['sharpe'])
    # Keep rule: discard flat, decaying, or single-year-driven blocks.
    sh = [y['sharpe'] for y in yearly]
    flat = abs(m['sharpe']) < 0.3
    decaying = len(sh) >= 2 and sh[-1] < sh[0] - 1.0 and sh[-1] < 0.5
    single = len(sh) >= 2 and sum(1 for s in sh if s > 1.0) == 1 and max(sh) - sorted(sh)[-2] > 1.5
    thin = False
    thin_year = None
    nuniv = sig.shape[1] if sig is not None else 0
    for y, c in cov.items():
        if c['min_long'] < max(5, nuniv * 0.01) or c['min_short'] < max(5, nuniv * 0.01):
            thin = True
            thin_year = y
            break
    if flat:
        keep, reason = False, "flat (|Sharpe|<0.3) — no edge; discard before combining."
    elif m['sharpe'] < -0.3:
        keep, reason = False, "negative edge as written — discard as-is (a negated version is a different component; diagnose it separately)."
    elif decaying:
        keep, reason = False, "decaying curve — edge fades; discard before combining."
    elif single:
        keep, reason = False, "single-year-propped — fragile; discard before combining."
    elif thin:
        keep = False
        reason = f"thin coverage in {thin_year} (min long/short near universe floor); needs backfill or discard."
    else:
        keep, reason = True, "survives standalone screen; eligible for combining."
    return {'expr': expr_str, 'sharpe': m['sharpe'], 'fitness': m['fitness'],
            'turnover_pct': m['turnover_pct'], 'yearly': yearly, 'coverage': cov,
            'curve': curve, 'keep': keep, 'reason': reason, 'rep': rep}


def _split_top_components(node):
    # Split a top-level combination into its building blocks so each can be
    # diagnosed standalone. Only splits additive/subtractive/multiplicative
    # combinations; anything else is a single component.
    if isinstance(node, Bin) and node.op in ('+', '-'):
        return [to_expr(node.l), to_expr(node.r)]
    if isinstance(node, Bin) and node.op == '*':
        return [to_expr(node.l), to_expr(node.r)]
    return [to_expr(node)]


def combine_with_justification(exprs, op, hypothesis):
    if op not in ('+', '-', '*'):
        raise ValueError("combination op must be one of '+', '-', '*'")
    if op == '*':
        why = ("multiply: hypothesis requires ALL conditions to hold simultaneously "
               "(e.g. good AND cheap AND under-owned); rejected add because either-leg-alone "
               "should not trigger the bet.")
    else:
        why = ("add/subtract: hypothesis is that each leg independently contributes and "
               "either alone still makes economic sense; rejected multiply because no "
               "'all-conditions-at-once' logic is claimed.")
    combo = exprs[0]
    for e in exprs[1:]:
        combo = f"({combo} {op} {e})" if op in ('+', '-') else f"(({combo}) * ({e}))"
    return combo, f"{hypothesis} Operator: {why}"


def compare_combined_vs_individual(combined_rep, individual_reps):
    rows = []
    for d in individual_reps:
        rows.append({'expr': d['expr'][:60], 'sharpe': d.get('sharpe'), 'fitness': d.get('fitness')})
    if combined_rep.get('error'):
        return rows, "combined expression errored; discard combination."
    m = combined_rep['metrics']
    rows.append({'expr': combined_rep['expression'][:60] + '  [COMBINED]',
                 'sharpe': m['sharpe'], 'fitness': m['fitness']})
    best_ind = max([d.get('sharpe', -99) for d in individual_reps] or [-99])
    if m['sharpe'] <= best_ind + 0.05 and m['fitness'] <= max([d.get('fitness', -99) for d in individual_reps] or [-99]) + 0.05:
        verdict = ("combined Sharpe/Fitness not meaningfully better than the best single "
                   "component — combination adds no value; discard, keep the best standalone.")
    else:
        verdict = "combined beats every individual component on evidence; keep combination."
    return rows, verdict


TEAM1_TEMPLATES = [
    ("Sentiment momentum",
     lambda r: f"ts_decay_linear(ts_mean(vec_avg(nws12_afterhsz_01l), {r.choice([3, 5, 8])}), {r.choice([8, 10, 15])})",
     "Persistent news sentiment predicts short-horizon drift; decay smooths the signal and cuts churn."),
    ("Earnings momentum",
     lambda r: f"rank(ts_delta(ebitda, {r.choice([42, 63, 126])}) / assets)",
     "EBITDA improvement relative to the asset base is fundamental momentum that the market underreacts to."),
    ("Leverage short",
     "-rank(ts_backfill(debt, 63) / ts_backfill(assets, 63))",
     "High-leverage firms earn lower risk-adjusted returns; short the top of the leverage distribution."),
    ("Sales momentum",
     lambda r: f"ts_delta(rank(ts_backfill(sales, 63)), {r.choice([42, 63, 126])})",
     "A rising sales rank captures revenue growth surprises before analysts fully revise."),
    ("Margin vs leverage spread",
     "rank(ebitda / sales) - rank(debt / assets)",
     "Quality spread: long profitable low-leverage names, short the reverse; both legs are classic quality factors."),
    ("Buzz intensity reversal",
     lambda r: f"-ts_zscore(vec_avg(nws12_afterhsz_01l), {r.choice([10, 20, 40])})",
     "Extreme spikes in news buzz mark attention-driven overreaction that mean-reverts."),
]

TEAM2_TEMPLATES = [
    ("IV skew premium",
     "-rank(implied_volatility_put_60 - implied_volatility_call_60)",
     "Steep put skew is a fear premium; short names whose skew is rich vs the cross-section."),
    ("Skew ratio short tenor",
     "-ts_backfill(implied_volatility_put_10, 10) / ts_backfill(implied_volatility_call_10, 10)",
     "Put/call IV ratio at short tenor measures crash demand; high ratio predicts underperformance."),
    ("Low volatility anomaly",
     lambda r: f"-ts_zscore(ts_std_dev(returns, {r.choice([10, 20])}), {r.choice([60, 120])})",
     "Low-idio-vol stocks earn positive risk-adjusted returns; short the high-vol tail."),
    ("Volume-weighted reversal",
     lambda r: f"-ts_zscore(returns * volume / ts_mean(volume, 20), {r.choice([3, 5, 10])})",
     "Down-moves on abnormal volume overreact and revert over a few days."),
    ("Short-term reversal",
     lambda r: f"-ts_zscore(returns, {r.choice([3, 5, 10])})",
     "Liquidity-provision edge: recent losers bounce, winners fade."),
    ("IV change reversal",
     lambda r: f"-ts_delta(ts_backfill(implied_volatility_call_10, 10), {r.choice([3, 5])})",
     "IV spikes overshoot realized risk; fading them captures the vol risk premium."),
    ("Price momentum",
     lambda r: f"ts_mean(returns, {r.choice([90, 120, 180])})",
     "Intermediate-horizon momentum drift; classic twelve-minus-one style exposure."),
]


class Ideator:
    def __init__(self, name, role, templates, rng):
        self.name = name
        self.role = role
        self.templates = templates
        self.rng = rng
        self.order = rng.permutation(len(templates))
        self.pointer = 0

    def propose(self, round_no):
        idx = int(self.order[self.pointer % len(self.templates)])
        self.pointer += 1
        name, builder, rationale = self.templates[idx]
        try:
            expr = builder(self.rng)
        except TypeError:
            expr = builder
        return Idea(name, expr, rationale)


def _mini(rep):
    if rep.get('error'):
        return f"ERROR: {rep['error']}"
    m = rep['metrics']
    return (f"sharpe={m['sharpe']:.2f} fitness={m['fitness']:.2f} turnover={m['turnover_pct']:.1f}% "
            f"conc={m['weight_concentration_pct']:.1f}% margin={m.get('margin_bps', 0.0):.1f}bps "
            f"subSharpe={rep['criteria']['subuniverse_sharpe']['value']:.2f} "
            f"corr={rep['criteria']['self_correlation']['value']:.2f}")


def _score(rep, expr=None):
    # Audit Phase A3: 4th element prefers shallower ASTs on exact ties
    # (parsimony pressure against bloat; GP/AlphaAgent practice).
    if rep.get('error'):
        return (-1, -1.0, -1.0, 0)
    nodes = -node_count(expr) if expr else 0
    return (sum(1 for c in rep['criteria'].values() if c['pass']),
            rep['metrics']['fitness'], rep['metrics']['sharpe'], nodes)


class Optimizer:
    def __init__(self, name, role, specialty):
        self.name = name
        self.role = role
        self.specialty = specialty

    @staticmethod
    def _load_lessons():
        """Phase B6: dead-end fix counts keyed (failed-gates, fix-action).

        Written by Competition from failed rounds (reports/lessons.jsonl).
        Missing/corrupt file -> {} (safe default: no filtering).
        """
        out = {}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'reports', 'lessons.jsonl')
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if not e.get('round_passed'):
                        key = (tuple(e.get('failed', [])), e.get('fix', ''))
                        out[key] = out.get(key, 0) + 1
        except OSError:
            pass
        return out

    def optimize(self, idea, panel, refs, own_refs, max_iter, settings=None):
        log = []
        try:
            node = parse(idea.expr)
        except WQError as e:
            log.append({'iteration': 0, 'action': f"could not parse idea: {e}", 'status': 'abandoned'})
            return idea.expr, {'error': str(e), 'passed': False}, log

        rep = simulate(to_expr(node), panel, refs, own_refs, settings=settings)
        log.append({'iteration': 0, 'action': f"evaluated raw idea '{idea.name}'", 'expr': to_expr(node),
                    'metrics': _mini(rep), 'failed': failed_criteria(rep),
                    'skeleton': skeleton_of(to_expr(node)), 'nodes': node_count(to_expr(node)),
                    'dimensions': dimension_check(to_expr(node)),
                    'alignment': alignment_score(idea.rationale, [to_expr(node)])})
        dim0 = dimension_check(to_expr(node))
        if dim0:
            log.append({'iteration': 0,
                        'action': f"dimension warning (advisory only, simulating anyway): {dim0}"})

        # Per-component diagnostic pass (mandatory before any combination is
        # kept). Decompose top-level add/sub/mul combos and diagnose each
        # block standalone: Sharpe/Fitness/Turnover, year-by-year Sharpe,
        # coverage, and curve shape.
        try:
            comp_exprs = _split_top_components(node)
        except Exception:
            comp_exprs = [to_expr(node)]
        component_diags = []
        if len(comp_exprs) > 1:
            for ce in comp_exprs:
                d = diagnose_component(ce, panel, refs, own_refs, settings)
                component_diags.append(d)
                log.append({'iteration': 0,
                            'action': f"component diagnostic: {ce[:80]}",
                            'metrics': (f"sharpe={d.get('sharpe', float('nan')):.2f} "
                                        f"fitness={d.get('fitness', float('nan')):.2f} "
                                        f"turnover={d.get('turnover_pct', float('nan')):.1f}% "
                                        f"curve=({d.get('curve', '?')}) "
                                        f"keep={d.get('keep')} :: {d.get('reason', '')}"),
                            'failed': [],
                            'component_expr': ce,
                            'component_yearly': d.get('yearly', []),
                            'component_coverage': d.get('coverage', {})})
            # If any leg is flat/decaying/single-year-propped, prefer
            # subtraction: fall back to the best surviving leg before adding
            # more machinery on top of a weak combination.
            survivors = [d for d in component_diags if d.get('keep')]
            if survivors and not rep.get('error'):
                best_leg = max(survivors, key=lambda d: (d.get('sharpe', -99), d.get('fitness', -99)))
                rows, verdict = compare_combined_vs_individual(rep, component_diags)
                log.append({'iteration': 0,
                            'action': f"combined-vs-individual: {verdict}",
                            'metrics': _mini(rep), 'failed': failed_criteria(rep),
                            'comparison': rows})
                if not best_leg['rep'].get('error') and _score(best_leg['rep'], best_leg['expr']) >= _score(rep, to_expr(node)):
                    log.append({'iteration': 0,
                                'action': (f"subtraction preferred: best leg '{best_leg['expr'][:60]}' "
                                           f"scores >= combined; reverting to smaller subset"),
                                'metrics': _mini(best_leg['rep']),
                                'failed': failed_criteria(best_leg['rep'])})
                    node = parse(best_leg['expr'])
                    rep = best_leg['rep']
        else:
            d = diagnose_component(comp_exprs[0], panel, refs, own_refs, settings)
            component_diags = [d]
            log.append({'iteration': 0,
                        'action': f"standalone diagnostic: {d.get('curve', '?')} {d.get('reason', '')}",
                        'metrics': _mini(rep), 'failed': failed_criteria(rep),
                        'component_yearly': d.get('yearly', []),
                        'component_coverage': d.get('coverage', {})})

        tried = {to_expr(node)}
        tried_skeletons = {(skeleton_of(to_expr(node)), window_tuple(to_expr(node)))}
        lessons = self._load_lessons()  # Phase B6: skip repeatedly dead-ended fixes
        best_expr, best_rep = to_expr(node), rep
        last_expr, last_rep = to_expr(node), rep
        diagnosis_counts = {}
        prev_sig = _mini(rep)

        for i in range(1, max_iter + 1):
            if rep.get('passed') or rep.get('error'):
                break
            failed_now = tuple(sorted(failed_criteria(rep)))
            chosen = None
            for f in self._fix_candidates(rep, node):
                if f.get('discouraged'):
                    log.append({'iteration': i, 'rejected_discouraged_pattern': f['action']})
                    continue
                if lessons.get((failed_now, f.get('action', '')), 0) >= 3:
                    log.append({'iteration': i,
                                'rejected_lesson': f"{f['action'][:80]} "
                                                   f"(dead-ended {lessons[(failed_now, f.get('action', ''))]}x "
                                                   f"on {list(failed_now)})"})
                    continue
                cand = f['node']
                sig = to_expr(cand)
                if sig in tried:
                    continue
                # Phase A1: skeleton gate — skip param-only re-sims of an
                # already-simulated (structure, window-vector) pair. Genuine
                # window changes always pass (new vector); only exact
                # structural repeats are blocked.
                skey = (skeleton_of(sig), window_tuple(sig))
                if skey in tried_skeletons:
                    log.append({'iteration': i, 'rejected_skeleton_duplicate': sig[:80]})
                    continue
                # Phase A2: dimensional gate — reject incoherent candidates
                # pre-simulate (Alpha-squared-style pruning). Verified clean
                # on all 13 team templates before enforcing.
                dimv = dimension_check(sig)
                if dimv:
                    log.append({'iteration': i, 'rejected_dimension_violation': dimv,
                                'expr': sig[:80]})
                    continue
                chosen = f
                break
            if chosen is None:
                log.append({'iteration': i, 'action': 'no viable fix remaining', 'status': 'abandoned',
                            'reason': 'all candidate fixes exhausted or previously tried'})
                break
            new_expr = to_expr(chosen['node'])
            tried.add(new_expr)
            tried_skeletons.add((skeleton_of(new_expr), window_tuple(new_expr)))
            node = chosen['node']
            rep = simulate(new_expr, panel, refs, own_refs, settings=settings)
            cur_sig = _mini(rep)
            entry = {'iteration': i, 'diagnosis': chosen['diagnosis'], 'hypothesis': chosen['hypothesis'],
                     'fix': chosen['action'], 'new_expr': new_expr, 'metrics': cur_sig,
                     'failed': failed_criteria(rep), 'nodes': node_count(new_expr)}
            log.append(entry)
            if _score(rep, new_expr) > _score(best_rep, best_expr):
                best_expr, best_rep = new_expr, rep
            last_expr, last_rep = new_expr, rep

            diag_key = chosen['diagnosis']
            if cur_sig == prev_sig:
                diagnosis_counts[diag_key] = diagnosis_counts.get(diag_key, 0) + 1
                if diagnosis_counts[diag_key] >= 1:
                    log.append({'iteration': i, 'action': f"stagnation: '{diag_key[:60]}...' produced unchanged metrics for 2 consecutive iterations — abandoning this fix path",
                                'status': 'stagnated'})
                    break
            else:
                diagnosis_counts.clear()
            prev_sig = cur_sig

        final_expr = last_expr
        final_rep = last_rep
        if _score(best_rep, best_expr) > _score(last_rep, last_expr) and best_rep.get('passed') and not last_rep.get('passed'):
            final_expr, final_rep = best_expr, best_rep

        return final_expr, final_rep, log

    def _fix_candidates(self, rep, node):
        failed = failed_criteria(rep)
        fixes = []
        conc_failed = 'weight_concentration' in failed
        t = rep['metrics']['turnover_pct'] if not rep.get('error') else 0.0

        # Subtraction before addition: if the signal is a combination, try
        # dropping each leg first. A smaller already-diagnosed subset is
        # preferred over new machinery on top of a weak combo.
        if isinstance(node, Bin) and node.op in ('+', '-', '*'):
            for side, label in ((node.l, 'left'), (node.r, 'right')):
                fixes.append({
                    'diagnosis': "Combined signal failing; testing whether a smaller subset already fixes it",
                    'hypothesis': (f"One leg of this '{node.op}' combination may be the source of the failure; "
                                   f"keeping only the {label} leg reverts to an already-diagnosed smaller subset"),
                    'action': f"subtractive fix: dropped one leg, kept {label} leg only",
                    'node': side})

        if conc_failed:
            sparse = {f for f in field_names(node) if any(tok in f for tok in SPARSE_TOKENS)} - backfilled_fields(node)
            if sparse:
                fixes.append({
                    'diagnosis': f"Max single-stock weight {rep['metrics']['weight_concentration_pct']:.1f}% of book on {rep['max_weight_date']} (stock #{rep['max_weight_stock']})",
                    'hypothesis': f"Sparse coverage in {sorted(sparse)[0]} leaves only a handful of covered names on some dates, so the normalized book piles into whatever is covered — {self.specialty}",
                    'action': f"added ts_backfill(..., 20) around sparse fields {sorted(sparse)}",
                    'node': add_backfill(node, 20)})
            if has_division(node):
                fixes.append({
                    'diagnosis': f"Max single-stock weight {rep['metrics']['weight_concentration_pct']:.1f}% of book on {rep['max_weight_date']}",
                    'hypothesis': f"Ratio constructions blow up when a denominator prints a near-zero value (e.g. option IV on an illiquid name), creating outlier weights — {self.specialty}",
                    'action': "wrapped signal in ts_rank(..., 120) to cap cross-sectional outliers",
                    'node': wrap_ts_rank(node, 120)})
            if not contains_group_neutralize(node):
                fixes.append({
                    'diagnosis': f"Max single-stock weight {rep['metrics']['weight_concentration_pct']:.1f}% of book on {rep['max_weight_date']}",
                    'hypothesis': "Sector tilts combined with thin coverage can pile weight into one industry; industry-neutralization re-spreads the book",
                    'action': "wrapped signal in group_neutralize(..., industry)",
                    'node': wrap_group_neutralize(node)})
            fixes.append({
                'diagnosis': f"Max single-stock weight {rep['metrics']['weight_concentration_pct']:.1f}% of book on {rep['max_weight_date']}",
                'hypothesis': "A single name dominates the book; truncate caps single-name signal to 5% so the demean/scale book cannot pile in",
                'action': "wrapped signal in truncate(..., 0.05)",
                'node': wrap_truncate(node, 0.05)})

        if 'turnover' in failed:
            if t > 70.0:
                fixes.append({
                    'diagnosis': f"Turnover {t:.1f}% exceeds the 70% cap",
                    'hypothesis': "Signal updates too quickly day-to-day; doubling smoothing windows roughly halves churn",
                    'action': "doubled all time-series windows",
                    'node': bump_windows(node, 2.0)})
                fixes.append({
                    'diagnosis': f"Turnover {t:.1f}% exceeds the 70% cap",
                    'hypothesis': "Linear decay spreads each day's trade over 15 days, directly cutting turnover without changing the thesis",
                    'action': "wrapped signal in ts_decay_linear(..., 15)",
                    'node': wrap_decay(node, 15)})
                fixes.append({
                    'diagnosis': f"Turnover {t:.1f}% exceeds the 70% cap",
                    'hypothesis': "Re-trading every open wastes turnover; trade_when holds the position and only updates on liquid days (volume > adv20), directly cutting trading days",
                    'action': "wrapped signal in trade_when(volume > adv20, ..., -1)",
                    'node': wrap_trade_when(node)})
                fixes.append({
                    'diagnosis': f"Turnover {t:.1f}% exceeds the 70% cap",
                    'hypothesis': "Small daily wiggles below 1% of book are not worth trading; hump filters them and only rebalances on large moves",
                    'action': "wrapped signal in hump(..., 0.01)",
                    'node': wrap_hump(node, 0.01)})
            else:
                fixes.append({
                    'diagnosis': f"Turnover {t:.1f}% below the 1% floor",
                    'hypothesis': "Signal is too slow to meet minimum participation; shortening windows raises responsiveness",
                    'action': "halved all time-series windows",
                    'node': bump_windows(node, 0.5)})

        if 'subuniverse_sharpe' in failed:
            if not contains_group_neutralize(node):
                fixes.append({
                    'diagnosis': f"Sub-universe Sharpe {rep['criteria']['subuniverse_sharpe']['value']:.2f} below cutoff {rep.get('subuniverse_cutoff', '?')} (S1 0.75-relative) while full-universe Sharpe is {rep['metrics']['sharpe']:.2f}",
                    'hypothesis': "Edge is concentrated in illiquid names or sector bets dominate the liquid subset; industry-neutralization isolates the stock-level signal",
                    'action': "wrapped signal in group_neutralize(..., industry)",
                    'node': wrap_group_neutralize(node)})
            fixes.append({
                    'diagnosis': f"Sub-universe Sharpe {rep['criteria']['subuniverse_sharpe']['value']:.2f} below cutoff {rep.get('subuniverse_cutoff', '?')} (S1 0.75-relative)",
                'hypothesis': "Fast noisy signals degrade in the liquid subset where pricing is efficient; longer windows de-noise",
                'action': "doubled all time-series windows",
                'node': bump_windows(node, 2.0)})

        if 'self_correlation' in failed:
            ref = rep.get('top_corr_ref', 'reference pool')
            refsh = rep.get('top_corr_ref_sharpe', 0.0)
            fixes.append({
                'discouraged': True,
                'action': f"considered wrapping in rank() to reduce correlation with '{ref}' — rejected: discouraged cosmetic pattern; surface-level re-wraps rarely reduce real correlation"})
            sib = swap_fields(node, SIBLINGS)
            if to_expr(sib) != to_expr(node):
                fixes.append({
                    'diagnosis': f"Self-correlation {rep['criteria']['self_correlation']['value']:.2f} with '{ref}' (ref Sharpe {refsh:.2f})",
                    'hypothesis': f"correlation with '{ref}' comes from sharing the same underlying data primitive; moving to a sibling field keeps the economic thesis but changes the data source",
                    'action': f"swapped primary data field to a sibling (e.g. different tenor/field) to decorrelate from '{ref}' while keeping the economic thesis",
                    'node': sib})
            fixes.append({
                'diagnosis': f"Self-correlation {rep['criteria']['self_correlation']['value']:.2f} with '{ref}' (ref Sharpe {refsh:.2f})",
                'hypothesis': "Level signals correlate with popular level factors; expressing the same thesis as change (delta) produces genuinely different positions",
                'action': "wrapped signal in ts_delta(..., 5)",
                'node': wrap_ts_delta(node, 5)})

        if 'sharpe' in failed or 'fitness' in failed:
            yearly = rep.get('yearly', [])
            worst = min(yearly, key=lambda y: y['sharpe']) if yearly else None
            worst_txt = f"weakest year is {worst['year']} (Sharpe {worst['sharpe']:.2f})" if worst else "performance uneven across years"
            sparse = {f for f in field_names(node) if any(tok in f for tok in SPARSE_TOKENS)} - backfilled_fields(node)
            if sparse:
                fixes.append({
                    'diagnosis': f"Sharpe {rep['metrics']['sharpe']:.2f} below 1.25; {worst_txt}",
                    'hypothesis': f"Sparse-coverage days in {sorted(sparse)[0]} inject stale or degenerate signal and drag returns; backfilling stabilizes the book through data outages",
                    'action': f"added ts_backfill(..., 20) around sparse fields {sorted(sparse)}",
                    'node': add_backfill(node, 20)})
            fixes.append({
                'diagnosis': f"Sharpe {rep['metrics']['sharpe']:.2f} below 1.25; {worst_txt}",
                'hypothesis': "Short windows fit noise; a longer lookback trades responsiveness for stability and usually lifts the worst year",
                'action': "doubled all time-series windows",
                'node': bump_windows(node, 2.0)})
            fixes.append({
                'diagnosis': f"Sharpe {rep['metrics']['sharpe']:.2f} below 1.25; {worst_txt}",
                'hypothesis': "Signal may be too slow for the effect; halving windows sharpens timing",
                'action': "halved all time-series windows",
                'node': bump_windows(node, 0.5)})
            sib = swap_fields(node, SIBLINGS)
            if to_expr(sib) != to_expr(node):
                fixes.append({
                    'diagnosis': f"Sharpe {rep['metrics']['sharpe']:.2f} below 1.25; {worst_txt}",
                    'hypothesis': "Primary data source may be weak; a sibling field (other tenor / other dataset) may carry the same thesis with a better signal-to-noise ratio",
                    'action': "swapped primary data field to a sibling",
                    'node': sib})

        return fixes


def make_teams(seed=5):
    rng = np.random.default_rng(seed)
    t1 = Ideator("1A", "Ideator — Sentiment & Fundamentals", TEAM1_TEMPLATES, np.random.default_rng(seed))
    o1 = Optimizer("1B", "Optimizer — Sentiment & Fundamentals",
                   "news and sentiment datasets have the thinnest coverage on low-liquidity dates")
    t2 = Ideator("2A", "Ideator — Microstructure & Volatility", TEAM2_TEMPLATES, np.random.default_rng(seed + 1))
    o2 = Optimizer("2B", "Optimizer — Microstructure & Volatility",
                   "options-implied fields blow up on near-zero IV prints and short-tenor sparsity")
    return [
        {'name': 'Team 1: Sentiment & Fundamentals', 'ideator': t1, 'optimizer': o1, 'id': 'T1'},
        {'name': 'Team 2: Microstructure & Volatility', 'ideator': t2, 'optimizer': o2, 'id': 'T2'},
    ]
