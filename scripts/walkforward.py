"""Walk-forward evaluation + selection-bias haircut + zoo benchmark (Phase 4).

Protocol: fit/select on TRAIN (2020-2021), lock expressions, score on TEST
(2022, untouched). Reports train-vs-test decay, a Harvey-Liu-Zhu-style
haircut for the number of candidates inspected, and whether the selected
alpha beats naive zoo baselines out-of-sample (walk-forward version of the
must-beat-best-leg discipline).
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_gen import Panel, generate
from simulator import simulate, build_reference_pool

CANDIDATES = [
    ("T1", "(ts_decay_linear(ts_mean(vec_avg(nws12_afterhsz_01l), 5), 10) + rank(ts_delta(ebitda, 63) / assets))"),
    ("T2", "(rank(ts_mean(returns, 120)) + -rank(ts_backfill(implied_volatility_put_60, 15) - ts_backfill(implied_volatility_call_60, 15)))"),
    ("T1fb", "(ts_delta(ebitda / sales, 32) + ts_mean(ts_backfill(vec_avg(nws12_afterhsz_01l), 20), 20))"),
    ("T2fb", "ts_decay_linear(-ts_zscore(returns, 21), 10)"),
    ("U2lev", "rank(ts_backfill(debt, 32) / ts_backfill(assets, 32))"),
]

ZOO = [  # naive baselines: does machinery beat a linear screen?
    ("zoo_mom120", "ts_mean(returns, 120)"),
    ("zoo_rev21", "-ts_zscore(returns, 21)"),
    ("zoo_quality", "rank(ts_backfill(ebitda, 63) / ts_backfill(assets, 63))"),
]

REF_EXPRS = [
    ("ref_mom60", "ts_mean(returns, 60)"),
    ("ref_rev5", "-ts_zscore(returns, 5)"),
    ("ref_quality", "rank(ts_backfill(ebitda, 63) / ts_backfill(assets, 63))"),
    ("ref_volume_spike", "rank(volume / ts_mean(volume, 20))"),
    ("ref_mom120", "ts_zscore(close, 120)"),
    ("ref_lowvol", "-rank(ts_std_dev(returns, 20))"),
    ("ref_asset_turnover", "rank(ts_backfill(sales, 63) / ts_backfill(assets, 63))"),
    ("ref_highvol", "rank(ts_std_dev(returns, 20))"),
]


def slice_panel(panel, start, end):
    start = np.datetime64(start)
    end = np.datetime64(end)
    idx = (panel.dates >= start) & (panel.dates <= end)
    fields = {k: v[idx] for k, v in panel.fields.items()}
    vecs = {k: [p[idx] for p in parts] for k, parts in panel.vector_fields.items()}
    sub = Panel(dates=panel.dates[idx], fields=fields, vector_fields=vecs,
                groups=panel.groups, subuniverse=panel.subuniverse)
    # Lineage: derived id names the parent fingerprint + slice range.
    sub.dataset_id = (getattr(panel, "dataset_id", "") or "noid") + f"[{start}:{end}]"
    return sub


def haircut_sharpe(observed, n_trials, n_obs_years=2.0):
    """Harvey-Liu-Zhu-style haircut: expected max Sharpe under null for
    n_trials ~ sqrt(2*ln(T)/N); haircut = observed - E[max], floored at 0.
    Approximate (assumes ~independent trials, normal null) — reported as a
    lens, not a gate.
    """
    if n_trials < 2:
        return round(observed, 4)
    emax = math.sqrt(2.0 * math.log(n_trials) / max(n_obs_years, 1e-9))
    return round(max(observed - emax, 0.0), 4)


def main():
    panel = generate(seed=11, n_stocks=1000)
    train = slice_panel(panel, "2020-01-01", "2021-12-31")
    test = slice_panel(panel, "2022-01-01", "2022-12-31")
    refs_tr = build_reference_pool(train, REF_EXPRS, settings=None)
    refs_te = build_reference_pool(test, REF_EXPRS, settings=None)

    rows = []
    for name, expr in CANDIDATES + ZOO:
        tr = simulate(expr, train, refs_tr, (), settings=None)
        te = simulate(expr, test, refs_te, (), settings=None)
        ts = tr["metrics"]["sharpe"] if not tr.get("error") else 0.0
        es = te["metrics"]["sharpe"] if not te.get("error") else 0.0
        rows.append({"name": name, "train_sharpe": round(ts, 3), "test_sharpe": round(es, 3),
                     "decay": round(ts - es, 3),
                     "test_pass": bool(te.get("passed")),
                     "test_fitness": round(te["metrics"]["fitness"], 3) if not te.get("error") else 0.0,
                     "error": tr.get("error") or te.get("error")})
    for r in rows:
        r["haircut_test_sharpe"] = haircut_sharpe(r["test_sharpe"], len(CANDIDATES) + len(ZOO), 1.0)
    rows.sort(key=lambda r: r["train_sharpe"], reverse=True)
    winner = next((r for r in rows if r["name"] in dict(CANDIDATES)), None)
    zoo_best = max((r for r in rows if r["name"] in dict(ZOO)), key=lambda r: r["test_sharpe"])

    print(f"{'alpha':<12}{'train_S':>9}{'test_S':>9}{'decay':>8}{'haircut':>9}{'test_pass':>11}")
    for r in rows:
        print(f"{r['name']:<12}{r['train_sharpe']:>9.2f}{r['test_sharpe']:>9.2f}"
              f"{r['decay']:>8.2f}{r['haircut_test_sharpe']:>9.2f}{str(r['test_pass']):>11}")
    print(f"\nWalk-forward verdict: selected '{winner['name']}' (best train) test_S={winner['test_sharpe']:.2f} "
          f"vs zoo-best '{zoo_best['name']}' test_S={zoo_best['test_sharpe']:.2f} -> "
          f"{'BEATS zoo' if winner['test_sharpe'] > zoo_best['test_sharpe'] else 'does NOT beat zoo'}")

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "walkforward.json"), "w") as f:
        json.dump({"train": "2020-2021", "test": "2022", "rows": rows,
                   "zoo_best": zoo_best["name"]}, f, indent=2, default=str)
    print("wrote reports/walkforward.json")


if __name__ == "__main__":
    main()
