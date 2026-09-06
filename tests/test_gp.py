"""Phase 3 GP tests: grammar validity, operators, gatekeeper wiring."""
import numpy as np

from gp import (random_expr, crossover, mutate, pareto_front, evaluate, evolve,
                NODE_CAP)
from agents import node_count, dimension_check, skeleton_of
from fastexpr import parse


def test_random_trees_valid_capped_clean():
    rng = np.random.default_rng(0)
    skels = set()
    for _ in range(50):
        e = random_expr(rng)
        parse(e)  # must parse
        assert node_count(e) <= NODE_CAP
        assert dimension_check(e) == []
        skels.add(skeleton_of(e))
    assert len(skels) > 5  # structural diversity, not one shape


def test_crossover_mutate_valid():
    rng = np.random.default_rng(1)
    a = "rank(ts_mean(returns, 120))"
    b = "-ts_zscore(returns, 21)"
    for _ in range(20):
        c = crossover(rng, a, b)
        assert c is None or node_count(parse(c) and c) >= 0
        m = mutate(rng, a)
        if m is not None:
            parse(m)
    assert crossover(rng, "((((", b) is None
    assert mutate(rng, "((((") is None


def test_pareto_front_nondominated():
    rs = [{"status": "evaluated", "fitness": f, "nodes": n, "expr": str(i)}
          for i, (f, n) in enumerate([(1.0, 5), (2.0, 5), (2.0, 9), (0.5, 3)])]
    hof = pareto_front(rs)
    got = {(r["fitness"], r["nodes"]) for r in hof}
    assert (2.0, 5) in got and (0.5, 3) in got  # non-dominated kept
    assert (2.0, 9) not in got and (1.0, 5) not in got  # dominated dropped


def test_pareto_tail_gap_breaks_ties():
    rs = [{"status": "evaluated", "fitness": 1.0, "nodes": 5,
           "tail_gap": g, "expr": str(i)} for i, g in enumerate([0.02, 0.005])]
    hof = pareto_front(rs)
    assert len(hof) == 1 and hof[0]["tail_gap"] == 0.005  # thinner tail wins


def test_evaluate_attaches_tail_gap(panel, refs):
    from gp import evaluate
    r = evaluate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs)
    assert r["status"] == "evaluated"
    assert isinstance(r["tail_gap"], float) and r["tail_gap"] == r["tail_gap"]


def test_evaluate_prunes_without_simulating():
    deep = "rank(returns)"
    for _ in range(16):
        deep = f"rank(ts_mean({deep}, 5))"
    assert node_count(deep) > NODE_CAP  # 16*3+2 = 50 nodes
    r = evaluate(deep, None, [])
    assert r["status"] == "pruned-node-cap"
    assert dimension_check("close + volume") != []
    r2 = evaluate("close + volume", None, [])
    assert r2["status"] == "pruned-dimension"


def test_mini_evolution_end_to_end(panel, refs):
    import gp as _gp
    rng = np.random.default_rng(2)
    pop = [random_expr(rng) for _ in range(8)]
    from selection import TrialRegistry
    reg = TrialRegistry()
    before = reg.count(panel.dataset_id)
    hof, log = evolve(panel, refs, seed=2, population=8, generations=1)
    assert log[0]["evaluated"] >= 1
    assert reg.count(panel.dataset_id) > before  # every sim auto-logged
    for h in hof:
        assert h["status"] == "evaluated"
        assert "eligible" in h and "promotion" in h  # DSR verdict attached
