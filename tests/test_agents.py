"""Agent machinery unit tests (no panel needed): fingerprint, dimensions, parsimony."""
from agents import (skeleton_of, dimension_check, node_count, window_tuple,
                    alignment_score, _score, TEAM1_TEMPLATES, TEAM2_TEMPLATES)
import numpy as np


def test_skeleton_groups_param_variants():
    a = "ts_decay_linear(ts_mean(returns, 10), 10)"
    b = "ts_decay_linear(ts_mean(returns, 21), 30)"
    assert skeleton_of(a) == skeleton_of(b)
    assert skeleton_of(a) != skeleton_of("rank(ts_mean(returns, 120))")
    assert window_tuple(a) != window_tuple(b)


def test_dimension_violations_caught():
    assert dimension_check("close + volume") != []
    assert dimension_check("rank(close + volume)") != []
    assert dimension_check("group_neutralize(rank(returns), volume)") != []
    assert dimension_check("vec_avg(returns)") != []


def test_templates_dimension_clean():
    rng = np.random.default_rng(5)
    for _, builder, _ in TEAM1_TEMPLATES + TEAM2_TEMPLATES:
        try:
            expr = builder(rng)
        except TypeError:
            expr = builder
        assert dimension_check(expr) == [], expr


def test_parsimony_prefers_shallower_on_ties():
    rep = {"metrics": {"fitness": 1.0, "sharpe": 2.0},
           "criteria": {"a": {"pass": True}}}
    shallow = _score(rep, "rank(returns)")
    deep = _score(rep, "rank(ts_mean(ts_decay_linear(returns, 5), 10))")
    assert shallow > deep


def test_alignment_scores_hypothesis_coverage():
    s = alignment_score("news sentiment drift predicts returns",
                        ["ts_mean(vec_avg(nws12_afterhsz_01l), 5)"])
    assert s == 1.0
    assert alignment_score("unrelated story about nothing", ["rank(returns)"]) == 0.0
    assert alignment_score("anything", ["foo(bar)"]) is None
