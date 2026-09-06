"""Phase 3: grammar-constrained genetic programming over fastexpr AST.

PySR-style candidate GENERATION only. The GP loop proposes expression trees;
every proposal passes through the existing gatekeeper unchanged:
  1. node-cap + dimension_check (pre-simulate prune),
  2. standalone diagnose + subtraction-first screen (agents.py discipline),
  3. simulate() auto-logs to TrialRegistry (N_trials grows honestly),
  4. promotion_eligible() with DSR >= 0.95 (selection.py) for any promotion.

GP never scores its own homework: fitness comes from simulate(), novelty
from skeleton_of(), parsimony from node_count(). Production scale is
population 300-1000 x generations 20-50 on a SMALL search panel with final
validation on the full panel; defaults below are verification-scale.
"""
import copy

import numpy as np

from fastexpr import (parse, to_expr, WQError, field_names, iter_nodes,
                      Call, Bin, Neg, Field, Num)
from agents import (diagnose_component, compare_combined_vs_individual,
                    skeleton_of, dimension_check, node_count, window_tuple,
                    _score)
from simulator import simulate, failed_criteria
from selection import TrialRegistry, promotion_eligible, DSR_PROMOTE
import selection as _selection  # for current SCOPE in evaluate()

NODE_CAP = 40  # Phase A3 parsimony: reject proposals above this pre-simulate

TERMINAL_FIELDS = [
    "returns", "close", "volume",
    "ebitda", "sales", "debt", "assets",
    "buzz",
    "implied_volatility_put_10", "implied_volatility_call_10",
    "implied_volatility_put_60", "implied_volatility_call_60",
]
SPARSE_FIELDS = {"buzz", "implied_volatility_put_10", "implied_volatility_call_10",
                 "implied_volatility_put_60", "implied_volatility_call_60"}
WINDOWS = [3, 5, 10, 15, 20, 21, 32, 60, 63, 120]


def _children(node):
    if isinstance(node, Call):
        return list(node.args)
    if isinstance(node, Bin):
        return [node.l, node.r]
    if isinstance(node, Neg):
        return [node.x]
    return []


def _with_child(node, i, child):
    node = copy.deepcopy(node)
    if isinstance(node, Call):
        node.args[i] = child
    elif isinstance(node, Bin):
        if i == 0:
            node.l = child
        else:
            node.r = child
    elif isinstance(node, Neg):
        node.x = child
    return node


def _subtrees(node, path=()):
    yield path, node
    for i, ch in enumerate(_children(node)):
        yield from _subtrees(ch, path + (i,))


def _get(node, path):
    for i in path:
        node = _children(node)[i]
    return node


def _set(node, path, child):
    if not path:
        return copy.deepcopy(child)
    kids = _children(node)
    kids[path[0]] = _set(kids[path[0]], path[1:], child)
    return _with_child(node, path[0], kids[path[0]])


def random_tree(rng, depth=0, max_depth=3):
    """Grow a random fastexpr tree (function set mirrors team templates)."""
    if depth >= max_depth or (depth > 0 and rng.random() < 0.35):
        f = Field(str(rng.choice(TERMINAL_FIELDS)))
        if f.name in SPARSE_FIELDS and rng.random() < 0.7:
            return Call("ts_backfill", [f, Num(int(rng.choice([10, 15, 20])))])
        return f
    r = rng.random()
    w = Num(int(rng.choice(WINDOWS)))
    if r < 0.22:
        return Call("rank", [random_tree(rng, depth + 1, max_depth)])
    if r < 0.38:
        return Call(rng.choice(["ts_mean", "ts_zscore", "ts_decay_linear", "ts_delta"]),
                    [random_tree(rng, depth + 1, max_depth), w])
    if r < 0.46:
        return Neg(random_tree(rng, depth + 1, max_depth))
    op = rng.choice(["+", "-", "*", "/"])
    return Bin(str(op), random_tree(rng, depth + 1, max_depth),
               random_tree(rng, depth + 1, max_depth))


def random_expr(rng, max_depth=3):
    for _ in range(20):
        node = random_tree(rng, 0, max_depth)
        expr = to_expr(node)
        if node_count(expr) <= NODE_CAP and not dimension_check(expr):
            return expr
    return "rank(ts_mean(returns, 120))"  # safe fallback (clean + small)


def crossover(rng, a_expr, b_expr):
    """Subtree swap between two parsed parents; None if either unparseable."""
    try:
        a, b = parse(a_expr), parse(b_expr)
    except WQError:
        return None
    sa = list(_subtrees(a))
    sb = list(_subtrees(b))
    if not sa or not sb:
        return None
    pa, _ = sa[int(rng.integers(len(sa)))]
    pb, nb = sb[int(rng.integers(len(sb)))]
    try:
        return to_expr(_set(a, pa, nb))
    except Exception:
        return None


def mutate(rng, expr):
    """Point/window/field/wrap mutation; None if unparseable."""
    try:
        node = parse(expr)
    except WQError:
        return None
    subs = list(_subtrees(node))
    if not subs:
        return None
    path, target = subs[int(rng.integers(len(subs)))]
    r = rng.random()
    try:
        if isinstance(target, Num) and isinstance(target.v, int) and r < 0.4:
            new = Num(int(rng.choice(WINDOWS)))
        elif isinstance(target, Field) and r < 0.7:
            new = Field(str(rng.choice(TERMINAL_FIELDS)))
        elif isinstance(target, Bin) and target.op in ("+", "-", "*", "/") and r < 0.85:
            new = Bin(str(rng.choice(["+", "-", "*", "/"])), target.l, target.r)
        else:
            inner = copy.deepcopy(target)
            new = Call(str(rng.choice(["rank", "ts_decay_linear"])),
                       [inner, Num(15)] if rng.random() < 0.5 else [inner])
            if isinstance(new, Call) and new.name == "rank" and len(new.args) > 1:
                new = Call("rank", [inner])
        return to_expr(_set(node, path, new))
    except Exception:
        return None


def evaluate(expr, panel, refs, own_refs=(), settings=None):
    """Gatekeeper evaluation of one GP proposal. Returns a result dict.

    Order matches the Phase-3 diagram: node-cap + dimension prune ->
    standalone diagnose (+ subtraction info) -> simulate (auto-logs trial) ->
    DSR/promotion verdict attached (promotion_eligible, DSR >= 0.95).
    Unparseable or pruned proposals return {'status': ...} without simulating.
    """
    if node_count(expr) > NODE_CAP:
        return {"expr": expr, "status": "pruned-node-cap", "nodes": node_count(expr)}
    dimv = dimension_check(expr)
    if dimv:
        return {"expr": expr, "status": "pruned-dimension", "violations": dimv}
    diag = diagnose_component(expr, panel, refs, own_refs, settings)
    rep = simulate(expr, panel, refs, own_refs, settings=settings)
    if rep.get("error"):
        return {"expr": expr, "status": "error", "error": rep["error"], "diag": diag}
    score = _score(rep, expr)
    elig, promo = promotion_eligible(rep, TrialRegistry(), DSR_PROMOTE,
                                     _selection.SCOPE)
    tail_gap = rep.get("risk", {}).get("es_gap_t_vs_normal_99", float("inf"))
    if tail_gap != tail_gap:  # NaN (degenerate lens) sorts as +inf: never wins ties
        tail_gap = float("inf")
    return {"expr": expr, "status": "evaluated", "sharpe": rep["metrics"]["sharpe"],
            "fitness": rep["metrics"]["fitness"], "passed": bool(rep.get("passed")),
            "failed": failed_criteria(rep), "score": score, "diag": diag,
            "eligible": elig, "promotion": promo, "tail_gap": tail_gap,
            "skeleton": skeleton_of(expr), "nodes": node_count(expr)}


def _dominated(a, b):
    """a dominated by b on (fitness up, nodes down, tail_gap down).

    Elementwise on all three axes — NOT lexicographic tuple comparison,
    which short-circuits on the first axis. Missing tail_gap reads as +inf
    on both sides, which reduces exactly to the legacy 2D rule, so old
    result dicts keep working unchanged.
    """
    ga, gb = a.get("tail_gap", float("inf")), b.get("tail_gap", float("inf"))
    better_eq = (b["fitness"] >= a["fitness"] and b["nodes"] <= a["nodes"]
                 and gb <= ga)
    strictly = (b["fitness"] > a["fitness"] or b["nodes"] < a["nodes"]
                or gb < ga)
    return bool(better_eq and strictly)


def pareto_front(results):
    """Non-dominated subset on (fitness up, nodes down, tail_gap down)."""
    ev = [r for r in results if r.get("status") == "evaluated"]
    return [a for a in ev if not any(_dominated(a, b) for b in ev if b is not a)]


def evolve(panel, refs, seed=0, population=24, generations=2, max_depth=3,
           own_refs=(), settings=None, cx_rate=0.5, mut_rate=0.3, elite=2,
           seed_exprs=()):
    """Run the GP loop. Returns (hall_of_fame, log).

    Selection: truncation (elites clone; parents sampled from top-10 by
    _score order). Children come from crossover (cx_rate), mutation
    (mut_rate), or fresh random trees. The hall of fame is the 3D Pareto
    front on (fitness up, nodes down, tail_gap down), so risk-steering acts
    at selection output: among equal-fitness/nodes candidates, the thinner
    tail wins a hall slot. Production: population 300-1000,
    generations 20-50 on a SMALL search panel; validate finalists on the
    full panel before any promotion claim.

    seed_exprs: known-good expressions (e.g. team templates) filling the
    initial population first — PySR-style seeded islands. Pure-random search
    transfer-poorly (verified: random search-panel passers print negative
    full-panel Sharpe); seeding anchors search near structures with
    full-panel evidence while mutation/crossover explore around them.

    IMPORTANT: eligibility/DSR attached to hall-of-fame entries is evaluated
    in the SEARCH context (search panel + its trial count). Search-panel
    passers routinely fail full-panel validation (verified 2026-09-06:
    search S-pass printed -0.48 on full n=1000 with DSR ~0.0). Promotion
    always requires fresh full-panel evaluation through promotion_eligible().
    """
    rng = np.random.default_rng(seed)
    pop = [s for s in seed_exprs
           if node_count(s) <= NODE_CAP and not dimension_check(s)]
    pop += [random_expr(rng, max_depth) for _ in range(max(population - len(pop), 0))]
    pop = pop[:population]
    evaluated = {}
    log = []
    for g in range(generations):
        results = []
        for expr in pop:
            if expr not in evaluated:
                evaluated[expr] = evaluate(expr, panel, refs, own_refs, settings)
            results.append(evaluated[expr])
        ev = [r for r in results if r.get("status") == "evaluated"]
        npass = sum(1 for r in ev if r["passed"])
        best = max(ev, key=lambda r: r["score"]) if ev else None
        gaps = [r.get("tail_gap", float("inf")) for r in ev]
        gaps = [g for g in gaps if g == g]
        log.append({"generation": g, "evaluated": len(ev),
                    "pruned": len(results) - len(ev), "passed": npass,
                    "best_tail_gap": min(gaps) if gaps else None,
                    "best": (best["expr"][:80], best["score"]) if best else None})
        ranked = sorted(ev, key=lambda r: r["score"], reverse=True)
        elites = [r["expr"] for r in ranked[:elite]]
        nxt = list(elites)
        while len(nxt) < population:
            r = rng.random()
            if r < cx_rate and len(ranked) >= 2:
                kids = sorted(rng.choice(len(ranked), size=min(3, len(ranked)),
                                         replace=False))
                a = max([ranked[k] for k in kids], key=lambda x: x["score"])["expr"]
                b = ranked[int(rng.integers(len(ranked)))]["expr"]
                child = crossover(rng, a, b)
            elif r < cx_rate + mut_rate and ranked:
                parent = ranked[int(rng.integers(min(len(ranked), 10)))]["expr"]
                child = mutate(rng, parent)
            else:
                child = random_expr(rng, max_depth)
            if child is None:
                continue
            if node_count(child) > NODE_CAP or dimension_check(child):
                continue
            nxt.append(child)
        pop = nxt[:population]
    results = [evaluated[e] for e in pop if e in evaluated]
    hof = pareto_front(results)
    return hof, log
