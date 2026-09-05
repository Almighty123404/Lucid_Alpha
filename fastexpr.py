import numpy as np


class WQError(Exception):
    pass


SYMBOLS = ['<=', '>=', '==', '!=', '<', '>', '+', '-', '*', '/', '^', '(', ')', ',', '&', '|', ';', '=']
_PREC = {'|': 1, '&': 2, '<': 3, '>': 3, '<=': 3, '>=': 3, '==': 3, '!=': 3, '+': 4, '-': 4, '*': 5, '/': 5, '^': 6}


def tokenize(s):
    toks = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or (c == '.' and i + 1 < n and s[i + 1].isdigit()):
            j = i
            while j < n and (s[j].isdigit() or s[j] == '.'):
                j += 1
            text = s[i:j]
            if text.count('.') > 1:
                raise WQError(f"Invalid number literal '{text}' at position {i}")
            toks.append(('NUM', float(text) if '.' in text else int(text), i))
            i = j
            continue
        if c.isalpha() or c == '_':
            j = i
            while j < n and (s[j].isalnum() or s[j] == '_'):
                j += 1
            toks.append(('ID', s[i:j], i))
            i = j
            continue
        for sym in SYMBOLS:
            if s.startswith(sym, i):
                toks.append(('SYM', sym, i))
                i += len(sym)
                break
        else:
            raise WQError(f"Unexpected character '{c}' at position {i}")
    toks.append(('EOF', None, n))
    return toks


class Num:
    def __init__(self, v):
        self.v = v


class Field:
    def __init__(self, name):
        self.name = name


class Call:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class Bin:
    def __init__(self, op, l, r):
        self.op = op
        self.l = l
        self.r = r


class Neg:
    def __init__(self, x):
        self.x = x


class Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, kind, value=None):
        t = self.next()
        if t[0] != kind or (value is not None and t[1] != value):
            want = value if value is not None else kind
            raise WQError(f"Expected '{want}' at position {t[2]}, got '{t[1]}'")
        return t

    def parse(self):
        node = self.or_expr()
        t = self.peek()
        if t[0] != 'EOF':
            raise WQError(f"Unexpected token '{t[1]}' at position {t[2]}")
        return node

    def or_expr(self):
        node = self.and_expr()
        while self.peek()[:2] == ('SYM', '|'):
            self.next()
            node = Bin('|', node, self.and_expr())
        return node

    def and_expr(self):
        node = self.cmp_expr()
        while self.peek()[:2] == ('SYM', '&'):
            self.next()
            node = Bin('&', node, self.cmp_expr())
        return node

    def cmp_expr(self):
        node = self.add_expr()
        while self.peek()[0] == 'SYM' and self.peek()[1] in ('<', '>', '<=', '>=', '==', '!='):
            op = self.next()[1]
            node = Bin(op, node, self.add_expr())
        return node

    def add_expr(self):
        node = self.mul_expr()
        while self.peek()[0] == 'SYM' and self.peek()[1] in ('+', '-'):
            op = self.next()[1]
            node = Bin(op, node, self.mul_expr())
        return node

    def mul_expr(self):
        node = self.unary()
        while self.peek()[0] == 'SYM' and self.peek()[1] in ('*', '/'):
            op = self.next()[1]
            node = Bin(op, node, self.unary())
        return node

    def unary(self):
        t = self.peek()
        if t[:2] == ('SYM', '-'):
            self.next()
            return Neg(self.unary())
        if t[:2] == ('SYM', '+'):
            self.next()
            return self.unary()
        return self.power()

    def power(self):
        node = self.atom()
        if self.peek()[:2] == ('SYM', '^'):
            self.next()
            return Bin('^', node, self.unary())
        return node

    def atom(self):
        t = self.next()
        if t[0] == 'NUM':
            return Num(t[1])
        if t[0] == 'ID':
            if self.peek()[:2] == ('SYM', '('):
                self.next()
                args = []
                if self.peek()[:2] != ('SYM', ')'):
                    args.append(self.or_expr())
                    while self.peek()[:2] == ('SYM', ','):
                        self.next()
                        args.append(self.or_expr())
                self.expect('SYM', ')')
                return Call(t[1], args)
            return Field(t[1])
        if t[:2] == ('SYM', '('):
            node = self.or_expr()
            self.expect('SYM', ')')
            return node
        raise WQError(f"Unexpected token '{t[1]}' at position {t[2]}")


def parse(s):
    if not isinstance(s, str) or not s.strip():
        raise WQError("Empty expression")
    return Parser(tokenize(s)).parse()


class Assign:
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr


def _split_top_level(s):
    """Split on ';' at paren depth 0 (Sec 4.3 multi-statement scripts)."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ';' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur))
    return parts


def parse_program(s):
    """Parse 'name = expr; ...; final' scripts (Sec 4.1/4.3).

    Returns (bindings, final_node) where bindings is a list of Assign and
    final_node is the alpha output expression. A trailing 'name = expr'
    counts as the output. Bare identifiers on the RHS are fields/locals.
    """
    if not isinstance(s, str) or not s.strip():
        raise WQError("Empty expression")
    stmts = [p.strip() for p in _split_top_level(s) if p.strip()]
    if not stmts:
        raise WQError("Empty expression")
    bindings = []
    final = None
    for i, st in enumerate(stmts):
        toks = tokenize(st)
        # assignment iff ID '=' ... at top level (tokens: ID, '=', rest, EOF)
        if (len(toks) >= 3 and toks[0][0] == 'ID' and toks[1][:2] == ('SYM', '=')):
            name = toks[0][1]
            sub = Parser(toks[2:])
            node = sub.or_expr()
            if sub.peek()[0] != 'EOF':
                t = sub.peek()
                raise WQError(f"Unexpected token '{t[1]}' at position {t[2]}")
            bindings.append(Assign(name, node))
            if i == len(stmts) - 1:
                final = Field(name)
        else:
            node = Parser(toks).parse()
            if i == len(stmts) - 1:
                final = node
            else:
                raise WQError(f"Only assignments allowed before the final expression (statement {i + 1})")
    return bindings, final


def eval_program(s, env):
    bindings, final = parse_program(s)
    scope = {}
    out = None
    for b in bindings:
        out = _eval_with_scope(b.expr, env, scope)
        scope[b.name] = out
    if final is not None and not (len(bindings) and isinstance(final, Field)
                                  and final.name == bindings[-1].name
                                  and len(_split_top_level(s)) == len(bindings)):
        # final bare expression distinct from trailing assignment target
        out = _eval_with_scope(final, env, scope)
    return out


def _eval_with_scope(node, env, scope):
    if not scope:
        return eval_node(node, env)
    scoped = Env(env.panel, verify_units=getattr(env, 'verify_units', True),
                 locals={**getattr(env, 'locals', {}), **scope})
    return eval_node(node, scoped)


def _expr(n, parent):
    if isinstance(n, Assign):
        return f"{n.name} = {_expr(n.expr, 0)}"
    if isinstance(n, Num):
        return str(n.v)
    if isinstance(n, Field):
        return n.name
    if isinstance(n, Call):
        return n.name + '(' + ', '.join(_expr(a, 0) for a in n.args) + ')'
    if isinstance(n, Neg):
        s = '-' + _expr(n.x, 7)
        return '(' + s + ')' if 7 < parent else s
    if isinstance(n, Bin):
        p = _PREC[n.op]
        ls = _expr(n.l, p)
        rs = _expr(n.r, p + 1)
        s = f"{ls} {n.op} {rs}"
        return '(' + s + ')' if p < parent else s
    raise WQError(f"Cannot serialize node {type(n).__name__}")


def to_expr(node):
    return _expr(node, 0)


def iter_nodes(node):
    yield node
    if isinstance(node, Assign):
        yield from iter_nodes(node.expr)
    elif isinstance(node, Call):
        for a in node.args:
            yield from iter_nodes(a)
    elif isinstance(node, Bin):
        yield from iter_nodes(node.l)
        yield from iter_nodes(node.r)
    elif isinstance(node, Neg):
        yield from iter_nodes(node.x)


def transform(node, fn):
    if isinstance(node, Assign):
        node = Assign(node.name, transform(node.expr, fn))
    elif isinstance(node, Call):
        node = Call(node.name, [transform(a, fn) for a in node.args])
    elif isinstance(node, Bin):
        node = Bin(node.op, transform(node.l, fn), transform(node.r, fn))
    elif isinstance(node, Neg):
        node = Neg(transform(node.x, fn))
    return fn(node)


def field_names(node):
    return {n.name for n in iter_nodes(node) if isinstance(n, Field)}


def program_to_expr(bindings, final):
    parts = [f"{b.name} = {_expr(b.expr, 0)}" for b in bindings]
    if bindings and isinstance(final, Field) and final.name == bindings[-1].name:
        return '; '.join(parts)
    return '; '.join(parts + [_expr(final, 0)])


def parse_any(s):
    """Parse single expressions and multi-statement programs (Sec 4.3).

    Returns ('single', node) or ('program', bindings, final).
    """
    bindings, final = parse_program(s)
    if not bindings:
        return ('single', final)
    return ('program', bindings, final)


def to_source(parsed):
    if isinstance(parsed, tuple) and parsed and parsed[0] == 'program':
        return program_to_expr(parsed[1], parsed[2])
    if isinstance(parsed, tuple) and parsed and parsed[0] == 'single':
        return to_expr(parsed[1])
    return to_expr(parsed)


def program_field_names(bindings, final):
    out = set()
    for b in bindings:
        out |= {n.name for n in iter_nodes(b.expr) if isinstance(n, Field)}
    out |= {n.name for n in iter_nodes(final) if isinstance(n, Field)}
    return out - {b.name for b in bindings}


def backfilled_fields(node):
    out = set()
    for n in iter_nodes(node):
        if isinstance(n, Call) and n.name == 'ts_backfill' and n.args and isinstance(n.args[0], Field):
            out.add(n.args[0].name)
    return out


def has_division(node):
    return any(isinstance(n, Bin) and n.op == '/' for n in iter_nodes(node))


TS_WINDOW_ARG = {'ts_mean': 1, 'ts_std_dev': 1, 'ts_decay_linear': 1, 'ts_delta': 1,
                 'ts_zscore': 1, 'ts_backfill': 1, 'ts_rank': 1, 'ts_covariance': 2,
                 'ts_sum': 1, 'ts_product': 1, 'ts_count_nans': 1, 'ts_av_diff': 1,
                 'ts_scale': 1, 'ts_arg_max': 1, 'ts_arg_min': 1, 'ts_delay': 1,
                 'ts_corr': 2, 'ts_regression': 2, 'ts_quantile': 1,
                 'kth_element': 1, 'last_diff_value': 1, 'ts_std': 1}


def bump_windows(node, factor):
    def fn(n):
        if isinstance(n, Call) and n.name in TS_WINDOW_ARG:
            i = TS_WINDOW_ARG[n.name]
            if i < len(n.args) and isinstance(n.args[i], Num) and isinstance(n.args[i].v, int):
                args = list(n.args)
                args[i] = Num(max(2, int(round(n.args[i].v * factor))))
                return Call(n.name, args)
        return n
    return transform(node, fn)


def add_backfill(node, w=20, sparse_tokens=('nws', 'snt', 'buzz', 'news', 'implied_volatility', 'est')):
    skip = backfilled_fields(node)

    def fn(n):
        if isinstance(n, Field) and n.name not in skip and any(t in n.name for t in sparse_tokens):
            return Call('ts_backfill', [Field(n.name), Num(w)])
        return n
    return transform(node, fn)


def swap_fields(node, mapping):
    def fn(n):
        if isinstance(n, Field) and n.name in mapping:
            return Field(mapping[n.name])
        return n
    return transform(node, fn)


def wrap_ts_rank(node, w=120):
    return Call('ts_rank', [node, Num(w)])


def wrap_decay(node, w=15):
    return Call('ts_decay_linear', [node, Num(w)])


def contains_group_neutralize(node):
    return any(isinstance(n, Call) and n.name == 'group_neutralize' for n in iter_nodes(node))


def wrap_group_neutralize(node, group='industry'):
    if contains_group_neutralize(node):
        return node
    return Call('group_neutralize', [node, Field(group)])


def wrap_ts_delta(node, w=5):
    return Call('ts_delta', [node, Num(w)])


def wrap_trade_when(node, cond=None, exit=-1):
    # cond defaults to a liquidity event: volume > adv20
    if cond is None:
        cond = Bin('>', Field('volume'), Field('adv20'))
    return Call('trade_when', [cond, node, Num(exit)])


def wrap_hump(node, h=0.01):
    return Call('hump', [node, Num(h)])


def wrap_truncate(node, m=0.05):
    return Call('truncate', [node, Num(m)])


def wrap_regression_neut(node, factor='returns'):
    return Call('regression_neut', [node, Field(factor)])


class GroupVal:
    def __init__(self, arr):
        self.arr = arr


class VectorVal:
    def __init__(self, parts):
        self.parts = parts


class Env:
    def __init__(self, panel, verify_units=True, locals=None):
        self.panel = panel
        # Sec 2.6 unitHandling: VERIFY=True raises structured unit errors;
        # OFF=False coerces leniently instead of raising.
        self.verify_units = verify_units
        self.locals = dict(locals) if locals else {}


def _field(name, env):
    if name in getattr(env, 'locals', {}):
        return env.locals[name]
    if name in env.panel.vector_fields:
        return VectorVal(env.panel.vector_fields[name])
    if name in env.panel.fields:
        return env.panel.fields[name]
    if name in env.panel.groups:
        return GroupVal(env.panel.groups[name])
    if name == 'nan':
        return float('nan')
    raise WQError(f"Unknown field or identifier '{name}'")


def _need_matrix(x, fname, env=None):
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return x.astype(np.float64, copy=False)
    if env is not None and not env.verify_units:
        # unitHandling=OFF: lenient coercion instead of structured error.
        if isinstance(x, (int, float)):
            T, N = env.panel.fields['returns'].shape
            return np.full((T, N), float(x))
        return np.full_like(env.panel.fields['returns'], np.nan)
    kind = ('scalar' if isinstance(x, (int, float)) else
            'Group' if isinstance(x, GroupVal) else
            'Vector' if isinstance(x, VectorVal) else type(x).__name__)
    raise WQError(f"Incompatible unit for input of '{fname}', expected Unit[Matrix], got Unit[{kind}]")


def _need_group(x, fname, env=None):
    if isinstance(x, GroupVal):
        return x
    if env is not None and not env.verify_units:
        return None
    kind = ('Matrix' if isinstance(x, np.ndarray) and x.ndim == 2 else
            'scalar' if isinstance(x, (int, float)) else
            'Vector' if isinstance(x, VectorVal) else type(x).__name__)
    raise WQError(f"Incompatible unit for input of '{fname}', expected Unit[Group], got Unit[{kind}]")


def _need_window(node, fname, idx=1):
    if idx < len(node.args) and isinstance(node.args[idx], Num) and isinstance(node.args[idx].v, int) and node.args[idx].v >= 1:
        return node.args[idx].v
    got = node.args[idx].v if idx < len(node.args) and isinstance(node.args[idx], Num) else '?'
    raise WQError(f"Window for '{fname}' must be a positive integer literal, got '{got}'")


def _arity(name, ev, lo, hi=None):
    hi = lo if hi is None else hi
    if not (lo <= len(ev) <= hi):
        raise WQError(f"Invalid number of arguments for '{name}': got {len(ev)}, expected {lo if lo == hi else f'{lo}-{hi}'}")


def _arith(op, a, b, opname):
    if isinstance(a, (GroupVal, VectorVal)) or isinstance(b, (GroupVal, VectorVal)):
        raise WQError(f"Incompatible unit for input of '{opname}', expected Unit[Matrix]")
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if op == '+':
            r = a + b
        elif op == '-':
            r = a - b
        elif op == '*':
            r = a * b
        elif op == '/':
            r = a / b
        else:
            r = np.power(a, b)
    if isinstance(r, np.ndarray):
        r = np.where(np.isinf(r), np.nan, r).astype(np.float64)
    elif isinstance(r, float) and not np.isfinite(r):
        r = float('nan')
    return r


def _compare(op, a, b, opname):
    if isinstance(a, (GroupVal, VectorVal)) or isinstance(b, (GroupVal, VectorVal)):
        raise WQError(f"Incompatible unit for input of '{opname}', expected Unit[Matrix]")
    with np.errstate(invalid='ignore'):
        if op == '<':
            r = a < b
        elif op == '>':
            r = a > b
        elif op == '<=':
            r = a <= b
        elif op == '>=':
            r = a >= b
        elif op == '==':
            r = a == b
        else:
            r = a != b
    r = r.astype(np.float64)
    if isinstance(a, np.ndarray):
        r[np.isnan(a)] = np.nan
    if isinstance(b, np.ndarray):
        r[np.isnan(b)] = np.nan
    return r


def _boolify(x):
    if isinstance(x, np.ndarray):
        return np.isfinite(x) & (x != 0)
    if isinstance(x, float):
        return np.isfinite(x) and x != 0
    return bool(x)


def _shift(x, k):
    out = np.full_like(x, np.nan)
    if k < x.shape[0]:
        out[k:] = x[:x.shape[0] - k]
    return out


def _cs_rank(x):
    out = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        m = np.isfinite(row)
        k = int(m.sum())
        if k < 2:
            continue
        v = row[m]
        order = np.argsort(v, kind='stable')
        ranks = np.empty(k)
        ranks[order] = np.arange(k)
        out[t, m] = ranks / (k - 1)
    return out


_PPF_A = np.array([-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239])
_PPF_B = np.array([-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572])
_PPF_C = np.array([-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783])
_PPF_D = np.array([0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416])


def _norm_ppf(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    out = np.empty_like(p)
    pl = 0.02425
    central = (p >= pl) & (p <= 1 - pl)
    q = p[central] - 0.5
    r = q * q
    num = ((((_PPF_A[0] * r + _PPF_A[1]) * r + _PPF_A[2]) * r + _PPF_A[3]) * r + _PPF_A[4]) * r + _PPF_A[5]
    den = ((((_PPF_B[0] * r + _PPF_B[1]) * r + _PPF_B[2]) * r + _PPF_B[3]) * r + _PPF_B[4]) * r + 1
    out[central] = q * num / den
    lower = p < pl
    q = np.sqrt(-2 * np.log(p[lower]))
    num = ((((_PPF_C[0] * q + _PPF_C[1]) * q + _PPF_C[2]) * q + _PPF_C[3]) * q + _PPF_C[4]) * q + _PPF_C[5]
    den = ((((_PPF_D[0] * q + _PPF_D[1]) * q + _PPF_D[2]) * q + _PPF_D[3]) * q + 1)
    out[lower] = num / den
    upper = p > 1 - pl
    q = np.sqrt(-2 * np.log(1 - p[upper]))
    num = ((((_PPF_C[0] * q + _PPF_C[1]) * q + _PPF_C[2]) * q + _PPF_C[3]) * q + _PPF_C[4]) * q + _PPF_C[5]
    den = ((((_PPF_D[0] * q + _PPF_D[1]) * q + _PPF_D[2]) * q + _PPF_D[3]) * q + 1)
    out[upper] = -num / den
    return out


def _group_neutralize(x, g):
    out = x.copy()
    for val in np.unique(g.arr):
        cols = g.arr == val
        sub = out[:, cols]
        valid = np.isfinite(sub)
        cnt = valid.sum(1)
        ssum = np.where(valid, sub, 0.0).sum(1)
        mean = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0)
        out[:, cols] = sub - mean[:, None]
    return out


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


def _roll_sum(x, w):
    T, N = x.shape
    xm = np.where(np.isfinite(x), x, 0.0)
    vm = np.isfinite(x).astype(np.float64)
    cx = np.vstack([np.zeros((1, N)), np.cumsum(xm, 0)])
    cv = np.vstack([np.zeros((1, N)), np.cumsum(vm, 0)])
    S = cx[w:] - cx[:-w]
    C = cv[w:] - cv[:-w]
    out = np.full_like(x, np.nan)
    out[w - 1:] = np.where(C > 0, S, np.nan)
    return out


def _roll_product(x, w):
    # NaN-ignored product; NaN if no valid obs in window. First w-1 rows NaN.
    T, N = x.shape
    out = np.full_like(x, np.nan)
    for t in range(w - 1, T):
        win = x[t - w + 1:t + 1]
        valid = np.isfinite(win)
        prod = np.where(valid, win, 1.0).prod(0)
        cnt = valid.sum(0)
        out[t] = np.where(cnt > 0, prod, np.nan)
    return out


def _roll_count_nans(x, w):
    T, N = x.shape
    vm = (~np.isfinite(x)).astype(np.float64)
    cv = np.vstack([np.zeros((1, N)), np.cumsum(vm, 0)])
    C = cv[w:] - cv[:-w]
    out = np.full_like(x, np.nan)
    out[w - 1:] = C
    return out


def _roll_min_max(x, w):
    # Rolling min/max ignoring NaN. NaN if no valid obs. First w-1 rows NaN.
    T, N = x.shape
    mn = np.full_like(x, np.nan)
    mx = np.full_like(x, np.nan)
    for t in range(w - 1, T):
        win = x[t - w + 1:t + 1]
        with np.errstate(invalid='ignore'):
            has = np.isfinite(win).any(0)
            mn[t] = np.where(has, np.nanmin(np.where(np.isfinite(win), win, np.inf), axis=0), np.nan)
            mx[t] = np.where(has, np.nanmax(np.where(np.isfinite(win), win, -np.inf), axis=0), np.nan)
    return mn, mx


def _roll_arg_extreme(x, w, which='max'):
    T, N = x.shape
    out = np.full_like(x, np.nan)
    for t in range(w - 1, T):
        win = x[t - w + 1:t + 1]
        valid = np.isfinite(win)
        cnt = valid.sum(0)
        masked = np.where(valid, win, -np.inf if which == 'max' else np.inf)
        idx = np.argmax(masked, axis=0) if which == 'max' else np.argmin(masked, axis=0)
        days = (w - 1) - idx
        out[t] = np.where(cnt > 0, days.astype(np.float64), np.nan)
    return out


def _cs_mean_std(x):
    m = np.isfinite(x)
    cnt = m.sum(1, keepdims=True)
    s = np.where(m, x, 0.0).sum(1, keepdims=True)
    mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    v = np.where(m, (x - mean) ** 2, 0.0).sum(1, keepdims=True)
    sd = np.sqrt(v / np.maximum(cnt - 1, 1))
    return mean, sd, cnt


def _group_apply(x, g_arr_1d_or_2d, fn, t=None):
    # Row-wise grouped apply. g may be 1D static (N,) or 2D dynamic (T,N).
    # fn(sub_1d, mask_1d) -> out_1d for one group slice.
    out = np.full_like(x, np.nan)
    if g_arr_1d_or_2d.ndim == 1:
        groups = g_arr_1d_or_2d
        uniq = np.unique(groups[np.isfinite(groups)] if groups.dtype.kind == 'f' else groups)
        for val in uniq:
            cols = groups == val
            if cols.sum() == 0:
                continue
            for t in range(x.shape[0]):
                row = x[t, cols]
                mask = np.isfinite(row)
                if mask.sum() == 0:
                    continue
                res = fn(row, mask)
                sl = np.zeros_like(row)
                sl[:] = np.nan
                sl[mask] = res[mask] if len(res) == len(row) else res
                out[t, cols] = sl
        return out
    # dynamic 2D: group ids vary per row
    for t in range(x.shape[0]):
        grow = g_arr_1d_or_2d[t]
        vals = np.unique(grow[np.isfinite(grow)])
        for val in vals:
            cols = grow == val
            row = x[t, cols]
            mask = np.isfinite(row)
            if mask.sum() == 0:
                continue
            res = fn(row, mask)
            sl = np.full_like(row, np.nan)
            sl[mask] = res[mask] if len(res) == len(row) else res
            out[t, cols] = sl
    return out


def _eval_call(node, env):
    name, args = node.name, node.args
    ev = [eval_node(a, env) for a in args]

    if name == 'rank':
        _arity(name, ev, 1)
        return _cs_rank(_need_matrix(ev[0], name, env))

    if name == 'quantile':
        # quantile(x, driver=1): 0=uniform (plain rank), 1=gaussian (default),
        # 2=cauchy (tan(pi*(p-0.5))). Always dimensionless Unit[] (Sec 4.4).
        _arity(name, ev, 1, 2)
        r = np.clip(_cs_rank(_need_matrix(ev[0], name, env)), 1e-9, 1 - 1e-9)
        driver = int(float(ev[1])) if len(ev) >= 2 else 1
        if driver == 0:
            return r
        if driver == 2:
            with np.errstate(over='ignore', invalid='ignore'):
                out = np.tan(np.pi * (r - 0.5))
            return np.where(np.isinf(out), np.nan, out)
        return _norm_ppf(r)

    if name == 'group_neutralize':
        _arity(name, ev, 2)
        return _group_neutralize(_need_matrix(ev[0], name, env), _need_group(ev[1], name, env))

    if name == 'if_else':
        _arity(name, ev, 3)
        c, a, b = ev
        if isinstance(c, (GroupVal, VectorVal)) or isinstance(a, (GroupVal, VectorVal)) or isinstance(b, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        cb = _boolify(c)
        with np.errstate(invalid='ignore'):
            out = np.where(cb, a, b)
        return np.asarray(out, dtype=np.float64)

    if name in ('vec_avg', 'vec_sum', 'vec_max', 'vec_min'):
        _arity(name, ev, 1)
        v = ev[0]
        if not isinstance(v, VectorVal):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Vector]")
        stack = np.stack(v.parts)
        valid = np.isfinite(stack)
        any_valid = valid.any(0)
        if name == 'vec_sum':
            s = np.where(valid, stack, 0.0).sum(0)
            return np.where(any_valid, s, np.nan)
        if name == 'vec_max':
            s = np.where(valid, stack, -np.inf).max(0)
            return np.where(any_valid, s, np.nan)
        if name == 'vec_min':
            s = np.where(valid, stack, np.inf).min(0)
            return np.where(any_valid, s, np.nan)
        cnt = valid.sum(0)
        s = np.where(valid, stack, 0.0).sum(0)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)

    if name in ('vector_neut', 'group_vector_neut'):
        # Sec 2.5: project alpha orthogonal to a risk-factor vector, per day
        # (vector_neut) or per group-day (group_vector_neut) — not group-demean.
        _arity(name, ev, 2, 3)
        x = _need_matrix(ev[0], name, env)
        f = _need_matrix(ev[1], name, env)
        if name == 'vector_neut':
            out = np.full_like(x, np.nan)
            for t in range(x.shape[0]):
                xv, fv = x[t], f[t]
                m = np.isfinite(xv) & np.isfinite(fv)
                if m.sum() < 2:
                    continue
                ff = float((fv[m] ** 2).sum())
                beta = float((xv[m] * fv[m]).sum() / ff) if ff > 1e-12 else 0.0
                row = np.full(x.shape[1], np.nan)
                row[m] = xv[m] - beta * fv[m]
                out[t] = row
            return out
        g = ev[2] if len(ev) >= 3 else None
        if g is None:
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Group]")
        garr = g.arr if isinstance(g, GroupVal) else _need_matrix(g, name, env)
        def _gvneut(row, mask, t, cols):
            xv = x[t, cols][mask]
            fv = f[t, cols][mask]
            ok = np.isfinite(xv) & np.isfinite(fv)
            res = np.full_like(row, np.nan)
            if ok.sum() < 2:
                res[mask] = xv
                return res
            ff = float((fv[ok] ** 2).sum())
            beta = float((xv[ok] * fv[ok]).sum() / ff) if ff > 1e-12 else 0.0
            full = np.full_like(row, np.nan)
            full[mask] = xv - beta * fv
            return full
        # group-wise projection reusing _group_apply plumbing
        out = np.full_like(x, np.nan)
        import numpy as _np
        if isinstance(g, GroupVal):
            for val in _np.unique(g.arr):
                cols = g.arr == val
                for t in range(x.shape[0]):
                    row = x[t, cols]
                    mask = _np.isfinite(row) & _np.isfinite(f[t, cols])
                    if mask.sum() == 0:
                        continue
                    xv = row[mask]
                    fv = f[t, cols][mask]
                    ff = float((fv ** 2).sum())
                    beta = float((xv * fv).sum() / ff) if ff > 1e-12 else 0.0
                    sl = _np.full_like(row, _np.nan)
                    sl[mask] = xv - beta * fv
                    out[t, cols] = sl
            return out
        return _group_apply(x, garr, lambda row, mask: row, None)

    if name in ('abs', 'log', 'sign', 'sqrt'):
        _arity(name, ev, 1)
        x = ev[0]
        if isinstance(x, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        with np.errstate(divide='ignore', invalid='ignore'):
            if name == 'abs':
                return np.abs(x)
            if name == 'log':
                return np.where(np.asarray(x) > 0, np.log(np.maximum(np.asarray(x, dtype=np.float64), 1e-300)), np.nan) if isinstance(x, np.ndarray) else float(np.log(x)) if x > 0 else float('nan')
            if name == 'sign':
                return np.sign(x)
            return np.sqrt(np.maximum(np.asarray(x, dtype=np.float64), 0.0)) if isinstance(x, np.ndarray) else float(np.sqrt(max(x, 0.0)))

    if name in ('max', 'min'):
        _arity(name, ev, 2)
        a, b = ev
        if isinstance(a, (GroupVal, VectorVal)) or isinstance(b, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        with np.errstate(invalid='ignore'):
            return np.maximum(a, b) if name == 'max' else np.minimum(a, b)

    if name == 'ts_mean':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        return _roll_mean(_need_matrix(ev[0], name, env), w)

    if name == 'ts_std_dev':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        return _roll_std(_need_matrix(ev[0], name, env), w)

    if name == 'ts_zscore':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        x = _need_matrix(ev[0], name, env)
        mean = _roll_mean(x, w)
        sd = _roll_std(x, w)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (x - mean) / sd
        return np.where(np.isinf(out), np.nan, out)

    if name == 'ts_decay_linear':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        x = _need_matrix(ev[0], name, env)
        xm = np.where(np.isfinite(x), x, 0.0)
        vm = np.isfinite(x).astype(np.float64)
        num = np.zeros_like(xm)
        den = np.zeros_like(vm)
        for i in range(w):
            wt = w - i
            num += wt * _shift(xm, i)
            den += wt * _shift(vm, i)
        out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        out[:w - 1] = np.nan
        return out

    if name == 'ts_delta':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        x = _need_matrix(ev[0], name, env)
        return _arith('-', x, _shift(x, w), name)

    if name == 'ts_backfill':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        x = _need_matrix(ev[0], name, env)
        out = x.copy()
        for i in range(1, w + 1):
            miss = ~np.isfinite(out)
            if not miss.any():
                break
            sh = _shift(x, i)
            take = miss & np.isfinite(sh)
            out[take] = sh[take]
        return out

    if name == 'ts_rank':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        x = _need_matrix(ev[0], name, env)
        valid = np.isfinite(x)
        less = np.zeros_like(x)
        cnt = np.zeros_like(x)
        for i in range(w):
            sh = _shift(x, i)
            shv = np.isfinite(sh)
            both = valid & shv
            less += (both & (sh < x)).astype(np.float64)
            cnt += both.astype(np.float64)
        ok = (np.arange(x.shape[0]) >= w - 1)[:, None] & (cnt > 0)
        return np.where(ok, less / np.maximum(cnt, 1e-12), np.nan)

    if name == 'ts_covariance':
        _arity(name, ev, 3)
        w = _need_window(node, name, 2)
        x = _need_matrix(ev[0], name, env)
        y = _need_matrix(ev[1], name, env)
        valid = (np.isfinite(x) & np.isfinite(y)).astype(np.float64)
        xv = np.where(valid > 0, x, 0.0)
        yv = np.where(valid > 0, y, 0.0)
        xy = xv * yv
        T, N = x.shape
        cs = lambda a: np.vstack([np.zeros((1, N)), np.cumsum(a, 0)])
        Sx = cs(xv)[w:] - cs(xv)[:-w]
        Sy = cs(yv)[w:] - cs(yv)[:-w]
        Sxy = cs(xy)[w:] - cs(xy)[:-w]
        C = cs(valid)[w:] - cs(valid)[:-w]
        out = np.full_like(x, np.nan)
        Csafe = np.maximum(C, 1e-12)
        cov = Sxy / Csafe - (Sx / Csafe) * (Sy / Csafe)
        out[w - 1:] = np.where(C > 0, cov, np.nan)
        return out

    # ---- Batch 1: arithmetic aliases (strict: NaN propagates unless filter=true) ----
    if name in ('add', 'subtract', 'multiply', 'divide', 'power'):
        _arity(name, ev, 2, 4)
        filt = bool(ev[2]) if len(ev) >= 3 and isinstance(ev[2], (int, float)) else False
        vals = list(ev[:2]) + ([ev[3]] if len(ev) == 4 else [])
        # filter=true: treat NaN as 0 before op; multi-arg fold for add/multiply/subtract
        ops = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/', 'power': '^'}
        op = ops[name]
        if filt:
            conv = [np.where(v != v, 0.0, v) if isinstance(v, np.ndarray) else (0.0 if isinstance(v, float) and v != v else v) for v in vals]
            r = conv[0]
            for b in conv[1:]:
                r = _arith(op, r, b, name)
            return r
        r = vals[0]
        for b in vals[1:]:
            r = _arith(op, r, b, name)
        return r

    if name == 'inverse':
        _arity(name, ev, 1)
        x = ev[0]
        if isinstance(x, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        return _arith('/', 1.0, x, name)

    if name == 'reverse':
        _arity(name, ev, 1)
        x = ev[0]
        if isinstance(x, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        return -x if isinstance(x, (int, float)) else np.negative(x)

    if name == 'signed_power':
        _arity(name, ev, 2)
        a, b = ev
        if isinstance(a, (GroupVal, VectorVal)) or isinstance(b, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.sign(a)
            r = s * np.power(np.abs(a), b)
        if isinstance(r, np.ndarray):
            r = np.where(np.isinf(r), np.nan, r).astype(np.float64)
        elif isinstance(r, float) and not np.isfinite(r):
            r = float('nan')
        return r

    # ---- Batch 1: logical ----
    if name in ('and', 'or'):
        _arity(name, ev, 2)
        a, b = ev
        if isinstance(a, (GroupVal, VectorVal)) or isinstance(b, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        ba, bb = _boolify(a), _boolify(b)
        r = (ba & bb) if name == 'and' else (ba | bb)
        out = r.astype(np.float64)
        # strict: NaN in either input -> NaN
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            out[np.isnan(a) | np.isnan(b)] = np.nan
        elif isinstance(a, np.ndarray):
            out[np.isnan(a)] = np.nan
        elif isinstance(b, np.ndarray):
            out[np.isnan(b)] = np.nan
        return out

    if name == 'not':
        _arity(name, ev, 1)
        x = ev[0]
        if isinstance(x, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        if isinstance(x, np.ndarray):
            out = np.full_like(x, np.nan)
            m = np.isfinite(x)
            out[m] = np.where(_boolify(x[m]), 0.0, 1.0)
            return out
        if isinstance(x, float):
            return float('nan') if not np.isfinite(x) else (0.0 if x != 0 else 1.0)
        return 0.0 if not x else 1.0

    if name == 'is_nan':
        _arity(name, ev, 1)
        x = ev[0]
        if isinstance(x, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        if isinstance(x, np.ndarray):
            return np.isnan(x).astype(np.float64)
        if isinstance(x, float):
            return 1.0 if np.isnan(x) else 0.0
        return 0.0

    # ---- Batch 1: cross-sectional ----
    if name == 'zscore':
        _arity(name, ev, 1)
        x = _need_matrix(ev[0], name, env)
        mean, sd, cnt = _cs_mean_std(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (x - mean) / np.where(sd > 1e-12, sd, np.nan)
        return np.where(np.isinf(out), np.nan, out)

    if name == 'normalize':
        # normalize(x, useStd=false, limit=0.0): demean; /std if useStd; clamp to [-limit,limit]
        _arity(name, ev, 1, 3)
        x = _need_matrix(ev[0], name, env)
        use_std = bool(ev[1]) if len(ev) >= 2 else False
        limit = float(ev[2]) if len(ev) >= 3 else 0.0
        mean, sd, cnt = _cs_mean_std(x)
        out = x - mean
        if use_std:
            out = out / np.where(sd > 1e-12, sd, np.nan)
        if limit and limit > 0:
            out = np.clip(out, -limit, limit)
        return np.where(np.isinf(out), np.nan, out)

    if name == 'winsorize':
        # winsorize(x, std=4): clamp to mean +/- std*sd
        _arity(name, ev, 1, 2)
        x = _need_matrix(ev[0], name, env)
        k = float(ev[1]) if len(ev) >= 2 else 4.0
        mean, sd, cnt = _cs_mean_std(x)
        lo = mean - k * sd
        hi = mean + k * sd
        return np.where(np.isfinite(x), np.clip(x, lo, hi), np.nan)

    if name == 'scale':
        # scale(x, scale=1, longscale=1, shortscale=1)
        _arity(name, ev, 1, 4)
        x = _need_matrix(ev[0], name, env)
        sc = float(ev[1]) if len(ev) >= 2 else 1.0
        ls = float(ev[2]) if len(ev) >= 3 else 1.0
        ss = float(ev[3]) if len(ev) >= 4 else 1.0
        out = np.full_like(x, np.nan)
        for t in range(x.shape[0]):
            row = x[t]
            m = np.isfinite(row)
            if m.sum() == 0:
                continue
            pos = row[m] > 0
            neg = row[m] < 0
            pv = row[m][pos]
            nv = np.abs(row[m][neg])
            ps, ns = pv.sum(), nv.sum()
            r = np.zeros_like(row[m])
            if ps > 1e-12:
                r[pos] = pv / ps * sc * ls
            if ns > 1e-12:
                r[neg] = -nv / ns * sc * ss
            out[t, m] = r
        return out

    # ---- Batch 1: group ----
    if name in ('group_rank', 'group_zscore', 'group_scale', 'group_mean', 'group_backfill'):
        if name == 'group_backfill':
            _arity(name, ev, 3, 4)
            x = _need_matrix(ev[0], name, env)
            w = int(ev[2]) if isinstance(ev[2], (int, float)) else 20
            g = ev[1]
            if isinstance(g, GroupVal):
                arr = g.arr
                out = x.copy()
                for val in np.unique(arr):
                    cols = arr == val
                    sub = out[:, cols]
                    valid = np.isfinite(sub)
                    cnt = valid.sum(1)
                    ssum = np.where(valid, sub, 0.0).sum(1)
                    mean = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
                    for t in range(x.shape[0]):
                        miss = ~np.isfinite(sub[t])
                        if miss.any() and np.isfinite(mean[t]):
                            # winsorized-ish fill capped at w-day lookback mean proxy
                            sub[t, miss] = mean[t]
                    out[:, cols] = sub
                return out
            gm = _need_matrix(g, name, env)
            # dynamic: fill NaN with cross-group row mean
            m = np.isfinite(x)
            rowmean = np.where(m.any(1, keepdims=True), np.where(m, x, 0.0).sum(1, keepdims=True) / m.sum(1, keepdims=True).clip(min=1), np.nan)
            return np.where(np.isfinite(x), x, rowmean)
        _arity(name, ev, 2)
        x = _need_matrix(ev[0], name, env)
        g = ev[1]
        garr = g.arr if isinstance(g, GroupVal) else _need_matrix(g, name, env)
        if name == 'group_mean':
            gm = _group_neutralize(np.where(np.isfinite(x), x, 0.0), g) if isinstance(g, GroupVal) else x
            # mean = x - neutralized(x) where valid
            cnt = np.isfinite(x).astype(np.float64)
            if isinstance(g, GroupVal):
                out = x - _group_neutralize(x, g)
                return np.where(np.isfinite(x), out, np.nan)
            return _group_apply(x, garr, lambda row, mask: np.full_like(row[mask], row[mask].mean()), None)
        if name == 'group_rank':
            def _grank(row, mask):
                v = row[mask]
                order = np.argsort(v, kind='stable')
                ranks = np.empty_like(v, dtype=np.float64)
                ranks[order] = np.arange(len(v))
                res = np.full_like(row, np.nan)
                res[mask] = ranks / max(len(v) - 1, 1)
                return res
            return _group_apply(x, garr, _grank, None)
        if name == 'group_zscore':
            def _gz(row, mask):
                v = row[mask]
                sd = v.std()
                res = np.full_like(row, np.nan)
                res[mask] = (v - v.mean()) / sd if sd > 1e-12 else 0.0
                return res
            return _group_apply(x, garr, _gz, None)
        # group_scale: scale within group to sum|.|=1
        def _gsc(row, mask):
            v = row[mask]
            s = np.abs(v).sum()
            res = np.full_like(row, np.nan)
            res[mask] = v / s if s > 1e-12 else 0.0
            return res
        return _group_apply(x, garr, _gsc, None)

    if name in ('bucket', 'densify'):
        if name == 'bucket':
            # bucket(x, nbuckets=10) or bucket(x, low, high, step): returns Group ids per row.
            # Parser has no strings, so numeric form only. Returns Matrix of ids (float)
            # so it composes with group_* which accept Matrix groups.
            _arity(name, ev, 2, 4)
            x = _need_matrix(ev[0], name, env)
            if len(ev) == 2:
                n = max(2, int(float(ev[1])))
                r = _cs_rank(x)
                out = np.floor(np.clip(r, 0, 1 - 1e-9) * n)
                return np.where(np.isfinite(x), out, np.nan)
            lo, hi = float(ev[1]), float(ev[2])
            step = float(ev[3]) if len(ev) >= 4 else (hi - lo) / 10.0
            out = np.floor((x - lo) / step)
            return np.where(np.isfinite(x), out, np.nan)
        _arity(name, ev, 1)
        x = ev[0]
        if isinstance(x, GroupVal):
            uniq = sorted(set(x.arr.tolist()))
            remap = {v: i for i, v in enumerate(uniq)}
            return GroupVal(np.array([remap[v] for v in x.arr]))
        m = _need_matrix(x, name, env)
        out = np.full_like(m, np.nan)
        for t in range(m.shape[0]):
            row = m[t]
            vals = sorted(set(row[np.isfinite(row)].tolist()))
            remap = {v: i for i, v in enumerate(vals)}
            out[t] = np.array([remap.get(v, np.nan) for v in row])
        return np.where(np.isfinite(m), out, np.nan)

    # ---- Batch 2: time-series rollers ----
    if name == 'ts_delay':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        return _shift(_need_matrix(ev[0], name, env), w)

    if name == 'ts_sum':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        return _roll_sum(_need_matrix(ev[0], name, env), w)

    if name == 'ts_product':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        return _roll_product(_need_matrix(ev[0], name, env), w)

    if name == 'ts_count_nans':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        return _roll_count_nans(_need_matrix(ev[0], name, env), w)

    if name == 'ts_av_diff':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        x = _need_matrix(ev[0], name, env)
        return _arith('-', x, _roll_mean(x, w), name)

    if name == 'ts_scale':
        _arity(name, ev, 2, 3)
        w = _need_window(node, name)
        const = float(ev[2]) if len(ev) >= 3 else 0.0
        x = _need_matrix(ev[0], name, env)
        mn, mx = _roll_min_max(x, w)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (x - mn) / (mx - mn) + const
        return np.where(np.isinf(out), np.nan, out)

    if name == 'ts_step':
        _arity(name, ev, 1)
        # ts_step(1): day counter broadcast to panel shape
        T, N = env.panel.fields['returns'].shape
        base = np.arange(T, dtype=np.float64)[:, None]
        return np.repeat(base, N, axis=1)

    if name in ('ts_arg_max', 'ts_arg_min'):
        _arity(name, ev, 2)
        w = _need_window(node, name)
        x = _need_matrix(ev[0], name, env)
        return _roll_arg_extreme(x, w, 'max' if name == 'ts_arg_max' else 'min')

    if name == 'kth_element':
        # kth_element(x, d, k): k-th most recent valid value (k=1 -> current-or-latest)
        _arity(name, ev, 3)
        w = _need_window(node, name, 1)
        x = _need_matrix(ev[0], name, env)
        k = int(float(ev[2])) if isinstance(ev[2], (int, float)) else 1
        k = max(1, k)
        T, N = x.shape
        out = np.full_like(x, np.nan)
        for t in range(T):
            lo = max(0, t - w + 1)
            win = x[lo:t + 1][::-1]  # most recent first
            cnt = np.zeros(N, dtype=int)
            acc = np.full(N, np.nan)
            for r in range(win.shape[0]):
                take = np.isfinite(win[r]) & np.isnan(acc)
                if (cnt[take] + 1 == k).any():
                    hit = take & ((cnt + 1) == k)
                    acc[hit] = win[r, hit]
                cnt[take] += 1
                if np.isfinite(acc).all():
                    break
            out[t] = acc
        out[:w - 1] = np.nan
        return out

    if name == 'last_diff_value':
        _arity(name, ev, 2)
        w = _need_window(node, name)
        x = _need_matrix(ev[0], name, env)
        T, N = x.shape
        out = np.full_like(x, np.nan)
        for t in range(T):
            cur = x[t]
            lo = max(0, t - w)
            for k in range(1, t - lo + 1):
                prev = x[t - k]
                diff = np.isfinite(cur) & np.isfinite(prev) & (prev != cur)
                hit = diff & np.isnan(out[t])
                out[t, hit] = prev[hit]
                if np.isfinite(out[t]).all() or (out[t][np.isfinite(cur)] != out[t][np.isfinite(cur)]).all():
                    pass
            # only keep where found
        return out

    if name == 'days_from_last_change':
        _arity(name, ev, 1)
        x = _need_matrix(ev[0], name, env)
        T, N = x.shape
        out = np.full_like(x, np.nan)
        last_ch = np.full(N, np.nan)
        prev = np.full(N, np.nan)
        for t in range(T):
            cur = x[t]
            both = np.isfinite(cur) & np.isfinite(prev)
            changed = both & (cur != prev)
            first = np.isfinite(cur) & ~np.isfinite(prev)
            last_ch[changed | first] = t
            out[t] = t - last_ch
            prev = cur
        return out

    if name == 'ts_corr':
        _arity(name, ev, 3)
        w = _need_window(node, name, 2)
        x = _need_matrix(ev[0], name, env)
        y = _need_matrix(ev[1], name, env)
        T, N = x.shape
        out = np.full_like(x, np.nan)
        for t in range(w - 1, T):
            xw, yw = x[t - w + 1:t + 1], y[t - w + 1:t + 1]
            valid = np.isfinite(xw) & np.isfinite(yw)
            cnt = valid.sum(0)
            xm = np.where(valid, xw, 0.0).sum(0) / np.maximum(cnt, 1)
            ym = np.where(valid, yw, 0.0).sum(0) / np.maximum(cnt, 1)
            dx = np.where(valid, xw - xm, 0.0)
            dy = np.where(valid, yw - ym, 0.0)
            sxx = (dx * dx).sum(0)
            syy = (dy * dy).sum(0)
            sxy = (dx * dy).sum(0)
            den = np.sqrt(sxx * syy)
            out[t] = np.where((cnt >= 2) & (den > 1e-12), sxy / np.maximum(den, 1e-12), np.nan)
        return out

    if name == 'ts_regression':
        # ts_regression(y, x, d, lag=0, rettype=0): 0=slope 1=intercept 2=residual
        _arity(name, ev, 3, 5)
        w = _need_window(node, name, 2)
        y = _need_matrix(ev[0], name, env)
        x = _need_matrix(ev[1], name, env)
        lag = int(float(ev[3])) if len(ev) >= 4 else 0
        rettype = int(float(ev[4])) if len(ev) >= 5 else 0
        xs = _shift(x, lag) if lag else x
        T, N = y.shape
        out = np.full_like(y, np.nan)
        for t in range(w - 1, T):
            yw, xw = y[t - w + 1:t + 1], xs[t - w + 1:t + 1]
            valid = np.isfinite(yw) & np.isfinite(xw)
            cnt = valid.sum(0)
            xm = np.where(valid, xw, 0.0).sum(0) / np.maximum(cnt, 1)
            ym = np.where(valid, yw, 0.0).sum(0) / np.maximum(cnt, 1)
            dx = np.where(valid, xw - xm, 0.0)
            dy = np.where(valid, yw - ym, 0.0)
            sxx = (dx * dx).sum(0)
            sxy = (dx * dy).sum(0)
            slope = np.where(sxx > 1e-12, sxy / np.maximum(sxx, 1e-12), np.nan)
            intercept = ym - slope * xm
            if rettype == 1:
                out[t] = np.where(cnt >= 2, intercept, np.nan)
            elif rettype == 2:
                out[t] = np.where(cnt >= 2, y[t] - (slope * xs[t] + intercept), np.nan)
            else:
                out[t] = np.where(cnt >= 2, slope, np.nan)
        return out

    if name == 'ts_quantile':
        # ts_quantile(x, d): gaussianized time-rank (driver string unsupported -> gaussian)
        _arity(name, ev, 2, 3)
        w = _need_window(node, name)
        x = _need_matrix(ev[0], name, env)
        tr = eval_node(Call('ts_rank', [node.args[0], node.args[1]]), env)
        return _norm_ppf(np.clip(tr, 1e-9, 1 - 1e-9))

    if name == 'trade_when':
        # trade_when(condition, alpha, exit): sticky position.
        # Keeps previous alpha while condition is false, takes new alpha
        # when condition is true, closes (NaN) when exit > 0.
        # exit=-1 (or any value <= 0) disables the exit leg.
        _arity(name, ev, 3)
        cond, alpha, ext = ev
        if isinstance(cond, (GroupVal, VectorVal)) or isinstance(alpha, (GroupVal, VectorVal)):
            raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
        # Determine shape
        ref = None
        for v in (cond, alpha, ext):
            if isinstance(v, np.ndarray) and v.ndim == 2:
                ref = v
                break
        if ref is None:
            return float(ext) if isinstance(ext, (int, float)) else ext
        T, N = ref.shape

        def _as_mat(v):
            if isinstance(v, np.ndarray) and v.ndim == 2:
                return v.astype(np.float64, copy=False)
            if isinstance(v, (int, float)):
                return np.full((T, N), float(v))
            if isinstance(v, GroupVal) or isinstance(v, VectorVal):
                raise WQError(f"Incompatible unit for input of '{name}', expected Unit[Matrix]")
            return np.full((T, N), float(v))

        cm = _as_mat(cond)
        am = _as_mat(alpha)
        em = _as_mat(ext)
        cb = np.isfinite(cm) & (cm != 0)
        ex = np.isfinite(em) & (em > 0)
        out = np.full((T, N), np.nan)
        pos = np.full(N, np.nan)
        for t in range(T):
            ex_t = ex[t]
            cb_t = cb[t]
            a_t = am[t]
            # exit first: close position
            pos[ex_t] = np.nan
            # enter/update where condition true
            upd = cb_t & ~ex_t
            pos[upd] = a_t[upd]
            out[t] = pos
        return out

    if name == 'hump':
        # hump(x, h): suppress rebalances smaller than h (fraction of book).
        # w_t = w_{t-1} if |x_t - w_{t-1}| <= h else x_t. booksize is 1
        # after the simulator's sum|w|=1 normalization, so threshold = h.
        _arity(name, ev, 1, 2)
        x = _need_matrix(ev[0], name, env)
        h = float(ev[1]) if len(ev) >= 2 else 0.01
        T, N = x.shape
        out = np.full_like(x, np.nan)
        pos = np.full(N, np.nan)
        for t in range(T):
            xt = x[t]
            finite = np.isfinite(xt)
            no_pos = ~np.isfinite(pos)
            # first valid observation seeds the position
            seed = finite & no_pos
            pos[seed] = xt[seed]
            # update only when the move is large enough
            carry = np.isfinite(pos) & finite & ~seed
            move = np.full(N, np.nan)
            move[carry] = np.abs(xt[carry] - pos[carry])
            big = carry & (move > h)
            pos[big] = xt[big]
            # NaN in x -> keep previous pos (acts as backfill); if no
            # previous pos, stays NaN
            out[t] = pos
        return out

    if name == 'truncate':
        # truncate(x, m): cap signal to [-m, m] elementwise.
        # Used as truncate(rank(signal), 0.05) to bound single-name
        # influence before the simulator's demean/scale step.
        _arity(name, ev, 1, 2)
        x = _need_matrix(ev[0], name, env)
        m = float(ev[1]) if len(ev) >= 2 else 0.05
        m = abs(m)
        return np.where(np.isfinite(x), np.clip(x, -m, m), np.nan)

    if name == 'regression_neut':
        # regression_neut(x, f1, f2, ...): cross-sectional residual of x
        # after regressing on factors (plus intercept) each day.
        # x = beta*f + eps -> returns eps.
        _arity(name, ev, 2, 6)
        x = _need_matrix(ev[0], name, env)
        factors = [_need_matrix(v, name, env) for v in ev[1:]]
        T, N = x.shape
        out = np.full_like(x, np.nan)
        for t in range(T):
            y = x[t]
            cols = [f[t] for f in factors]
            valid = np.isfinite(y)
            for c in cols:
                valid &= np.isfinite(c)
            k = int(valid.sum())
            nfac = len(cols)
            if k < nfac + 2:
                continue
            Y = y[valid]
            F = np.column_stack([np.ones(k)] + [c[valid] for c in cols])
            try:
                beta, *_ = np.linalg.lstsq(F, Y, rcond=None)
                resid = Y - F @ beta
            except np.linalg.LinAlgError:
                continue
            row = np.full(N, np.nan)
            row[valid] = resid
            out[t] = row
        return out

    raise WQError(f"Unknown operator '{name}'")


def _eval_call_scoped(node, env, scope):
    return _eval_with_scope(node, env, scope)


def _coerce_bin_operands(l, r, env):
    if getattr(env, 'verify_units', True):
        return l, r
    def _cv(v):
        if isinstance(v, (GroupVal, VectorVal)):
            return np.full_like(env.panel.fields['returns'], np.nan)
        return v
    return _cv(l), _cv(r)


def eval_node(node, env):
    if isinstance(node, Num):
        return node.v
    if isinstance(node, Field):
        return _field(node.name, env)
    if isinstance(node, Neg):
        x = eval_node(node.x, env)
        if isinstance(x, (GroupVal, VectorVal)):
            if not getattr(env, 'verify_units', True):
                return np.full_like(env.panel.fields['returns'], np.nan)
            raise WQError("Incompatible unit for input of '-', expected Unit[Matrix]")
        return -x
    if isinstance(node, Bin):
        l = eval_node(node.l, env)
        r = eval_node(node.r, env)
        l, r = _coerce_bin_operands(l, r, env)
        if node.op in ('+', '-', '*', '/', '^'):
            return _arith(node.op, l, r, node.op)
        if node.op in ('<', '>', '<=', '>=', '==', '!='):
            return _compare(node.op, l, r, node.op)
        if node.op == '&':
            return (_boolify(l) & _boolify(r)).astype(np.float64)
        if node.op == '|':
            return (_boolify(l) | _boolify(r)).astype(np.float64)
        raise WQError(f"Unknown operator '{node.op}'")
    if isinstance(node, Call):
        try:
            return _eval_call(node, env)
        except WQError as e:
            if not getattr(env, 'verify_units', True) and 'Incompatible unit' in str(e):
                return np.full_like(env.panel.fields['returns'], np.nan)
            raise
    raise WQError(f"Cannot evaluate node {type(node).__name__}")
