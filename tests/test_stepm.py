"""Romano-Wolf / White / PBO tests on known-truth synthetic return series."""
import numpy as np

from selection import (white_reality_check, romano_wolf_stepm, pbo_diagnostic,
                       portfolio_screen)


def _edge(T=500, mu=0.002, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([[0.0], rng.normal(mu, sd, T)])


def _noise(T=500, seed=1):
    rng = np.random.default_rng(seed)
    return np.concatenate([[0.0], rng.normal(0.0, 0.01, T)])


def test_white_detects_planted_edge():
    rej, p = white_reality_check(_edge(), np.zeros(501), B=200, seed=0)
    assert rej is True and p < 0.05


def test_white_clears_pure_noise():
    rej, p = white_reality_check(_noise(), np.zeros(501), B=200, seed=1)
    assert rej is False and p >= 0.05


def test_stepm_finds_edge_not_noise():
    models = [_edge(seed=0)] + [_noise(seed=i) for i in (1, 2, 3)]
    rej, adj = romano_wolf_stepm(models, None, alpha=0.05, B=200, seed=0)
    assert 0 in rej  # planted edge survives FWER control
    assert len(adj) == 4
    assert adj[0] <= 0.05


def test_stepm_all_noise_rejects_nothing():
    models = [_noise(seed=i) for i in (11, 12, 13)]
    rej, adj = romano_wolf_stepm(models, None, alpha=0.05, B=200, seed=0)
    assert rej == []
    assert all(p > 0.05 for p in adj)


def test_pbo_low_on_persistent_edge_high_on_noise():
    edge_set = [_edge(seed=i) for i in range(6)]
    noise_set = [_noise(seed=100 + i) for i in range(6)]
    pe = pbo_diagnostic(edge_set, S=8)
    pn = pbo_diagnostic(noise_set, S=8)
    assert pe["pbo"] is not None and pn["pbo"] is not None
    assert pe["pbo"] < pn["pbo"]
    assert 0.3 <= pn["pbo"] <= 0.7  # noise: IS-best is a coin flip OOS


def test_pbo_insufficient_models_reports_not_blocks():
    assert pbo_diagnostic([_edge()])["pbo"] is None


def test_portfolio_screen_pass_fail():
    ok, info = portfolio_screen(_edge(seed=0),
                                [_noise(seed=i) for i in (1, 2, 3, 4, 5)],
                                B=200)
    assert ok is True
    assert info["candidate_adj_p"] <= 0.05
    assert "pbo" in info
    ok2, info2 = portfolio_screen(_noise(seed=7),
                                  [_noise(seed=i) for i in (8, 9, 10, 11, 12)],
                                  B=200)
    assert ok2 is False

def test_promotion_stepm_wiring(panel, refs):
    from simulator import simulate
    from selection import TrialRegistry, promotion_eligible, DSR_PROMOTE
    rep = simulate("ts_decay_linear(-ts_zscore(returns, 21), 10)", panel, refs, (), settings=None)
    peers = [r["pnl"] for r in sorted(refs, key=lambda r: r["sharpe"], reverse=True)[:3]]
    elig, info = promotion_eligible(rep, TrialRegistry(), DSR_PROMOTE, "default", peer_pnls=peers)
    assert "stepm" in info  # screen ran and attached detail
    assert info["stepm"]["n_models"] == 4
    assert isinstance(elig, bool)
