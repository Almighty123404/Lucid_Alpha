"""First automated production discovery campaign (Phase 4+).

Protocol: GP search on a SMALL panel (different seed from validation, to
avoid seed overfit) -> Pareto hall-of-fame -> re-evaluate finalists on the
FULL panel -> harvest promotion-eligible candidates (gates passed AND
DSR >= 0.95 with honestly logged N_trials).

Usage: python scripts/gp_campaign.py [population] [generations] [seed]
Writes reports/gp_campaign.json (gitignored).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_gen import generate
from simulator import simulate, build_reference_pool
from tests.conftest import REF_EXPRS
from gp import evolve

SEED_ISLANDS = [  # team-template structures with full-panel evidence
    "ts_decay_linear(ts_mean(vec_avg(nws12_afterhsz_01l), 5), 10)",
    "rank(ts_delta(ebitda, 63) / assets)",
    "rank(ts_mean(returns, 120))",
    "-rank(ts_backfill(implied_volatility_put_60, 15) - ts_backfill(implied_volatility_call_60, 15))",
    "ts_decay_linear(-ts_zscore(returns, 21), 10)",
    "ts_delta(ebitda / sales, 32)",
]
from selection import promotion_eligible, TrialRegistry, DSR_PROMOTE


def main():
    pop = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    print(f"SEARCH panel: seed={seed} n=200 (validation: seed=11 n=1000)", flush=True)
    search = generate(seed=seed, n_stocks=200, start="2020-01-01", end="2022-12-31")
    srefs = build_reference_pool(search, REF_EXPRS, settings=None)
    hof, log = evolve(search, srefs, seed=seed, population=pop, generations=gens,
                      seed_exprs=SEED_ISLANDS)
    for g in log:
        b = g["best"]
        print(f"gen {g['generation']}: evaluated={g['evaluated']} pruned={g['pruned']} "
              f"passed={g['passed']} best={b[0][:60] if b else None} score={b[1] if b else None}",
              flush=True)
    print(f"hall of fame: {len(hof)}", flush=True)
    full = generate(seed=11, n_stocks=1000)
    frefs = build_reference_pool(full, REF_EXPRS, settings=None)
    harvested = []
    for h in hof:
        rep = simulate(h["expr"], full, frefs, (), settings=None)
        if rep.get("error"):
            continue
        m = rep["metrics"]
        elig, promo = promotion_eligible(rep, TrialRegistry(), DSR_PROMOTE)
        print(f"FULL {h['expr'][:70]}: S={m['sharpe']:.2f} F={m['fitness']:.2f} "
              f"pass={rep['passed']} dsr={promo.get('dsr')} eligible={elig}", flush=True)
        if elig:
            harvested.append({"expr": h["expr"], "sharpe": m["sharpe"],
                              "fitness": m["fitness"], "dsr": promo.get("dsr"),
                              "n_trials": promo.get("n_trials")})
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "gp_campaign.json"), "w") as f:
        json.dump({"population": pop, "generations": gens, "search_seed": seed,
                   "hof_size": len(hof), "harvested": harvested}, f, indent=2)
    print(f"HARVESTED {len(harvested)} DSR>=0.95 promotion-eligible candidates", flush=True)


if __name__ == "__main__":
    main()
