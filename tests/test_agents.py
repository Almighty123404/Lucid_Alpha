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

def test_vec_ops_and_codebook_dims():
    from fastexpr import Env, eval_node, parse
    from data_gen import generate
    panel = generate(seed=11, n_stocks=30, start="2020-01-01", end="2020-03-31")
    env = Env(panel)
    for e in ["vec_mean(analyst_eps_estimates)", "vec_std(analyst_eps_estimates)",
              "vec_choose(option_implied_vol_surface, 2)", "vec_sum(segment_revenue)",
              "rank(free_cash_flow / assets)", "ts_mean(est_eps, 20)",
              "-rank(short_interest)", "group_neutralize(rank(eps), market)"]:
        assert dimension_check(e) == [], e
        out = eval_node(parse(e), env)
        assert out.shape == (panel.fields["returns"].shape[0], 30), e
    assert dimension_check("vec_std(returns)") != []
    assert dimension_check("vec_choose(returns, 1)") != []


def test_family_longest_match_no_collision():
    from calibration import family_of_expression
    assert family_of_expression("rank(capex / assets)") == "fundamental"
    assert family_of_expression("rank(opt_open_interest)") == "derivatives"
    assert family_of_expression("vec_mean(analyst_eps_estimates)") == "fundamental"

def test_alias_and_subindustry_support():
    from fastexpr import Env, eval_node, parse
    from data_gen import generate
    from calibration import family_of_expression
    panel = generate(seed=11, n_stocks=30, start="2020-01-01", end="2020-03-31")
    env = Env(panel)
    for e in ["rank(revenue / assets)", "rank(ni / sales)", "rank(total_debt / assets)",
              "ts_mean(ocf, 20)", "rank(fcf)", "ts_mean(pcr, 20)",
              "ts_mean(implied_volatility_10, 20)", "ts_mean(historical_volatility_20, 20)",
              "group_rank(rank(returns), subindustry)",
              "group_neutralize(rank(returns), subindustry)"]:
        assert dimension_check(e) == [], e
        out = eval_node(parse(e), env)
        assert out.shape == (panel.fields["returns"].shape[0], 30), e
    assert family_of_expression("rank(total_debt / assets)") == "fundamental"
    assert family_of_expression("rank(ni)") == "fundamental"
    assert family_of_expression("ts_mean(pcr, 20)") == "derivatives"
    assert family_of_expression("vec_mean(analyst_eps_estimates)") == "fundamental"
