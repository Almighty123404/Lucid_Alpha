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

from data_gen import generate, Panel
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
from selection import promotion_eligible, TrialRegistry, DSR_PROMOTE, set_scope

CAMPAIGN_SCOPE = "gp-campaign"


def main():
    pop = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    # Scoped registry: campaign trials count toward campaign multiplicity only.
    set_scope(CAMPAIGN_SCOPE)
    print(f"SEARCH panel: seed={seed} n=200 (validation: seed=11 n=1000)", flush=True)
    search = generate(seed=seed, n_stocks=200, start="2020-01-01", end="2022-12-31")
    import numpy as _np
    # Selection must never see 2022: evolve on the 2020-21 train slice so the
    # holdout below measures genuine generalization, not memorization.
    tr = (search.dates < _np.datetime64("2022-01-01"))
    tpanel = Panel(dates=search.dates[tr],
                   fields={k: v[tr] for k, v in search.fields.items()},
                   vector_fields={k: [p[tr] for p in parts]
                                  for k, parts in search.vector_fields.items()},
                   groups=search.groups, subuniverse=search.subuniverse,
                   dataset_id=(search.dataset_id or "noid") + "[2020-21-train]")
    trefs = build_reference_pool(tpanel, REF_EXPRS, settings=None)
    hof, log = evolve(tpanel, trefs, seed=seed, population=pop, generations=gens,
                      seed_exprs=SEED_ISLANDS)
    for g in log:
        b = g["best"]
        print(f"gen {g['generation']}: evaluated={g['evaluated']} pruned={g['pruned']} "
              f"passed={g['passed']} best={b[0][:60] if b else None} score={b[1] if b else None}",
              flush=True)
    print(f"hall of fame: {len(hof)}", flush=True)
    # Search-holdout filter: finalists selected on 2020-21 train must also
    # pass the search panel's 2022 slice (untouched by selection) before
    # full-panel validation. Kills static tilts that memorize one regime.
    hold = (search.dates >= _np.datetime64("2022-01-01"))
    hpanel = Panel(dates=search.dates[hold],
                   fields={k: v[hold] for k, v in search.fields.items()},
                   vector_fields={k: [p[hold] for p in parts]
                                  for k, parts in search.vector_fields.items()},
                   groups=search.groups, subuniverse=search.subuniverse,
                   dataset_id=(search.dataset_id or "noid") + "[2022-holdout]")
    hrefs = build_reference_pool(hpanel, REF_EXPRS, settings=None)
    survivors = []
    for h in hof:
        rep = simulate(h["expr"], hpanel, hrefs, (), settings=None)
        if rep.get("error"):
            continue
        if rep.get("passed"):
            survivors.append(h["expr"])
            print(f"HOLDOUT PASS {h['expr'][:70]}: S={rep['metrics']['sharpe']:.2f}", flush=True)
        else:
            print(f"holdout reject {h['expr'][:70]}", flush=True)
    print(f"holdout survivors: {len(survivors)}/{len(hof)}", flush=True)
    full = generate(seed=11, n_stocks=1000)
    frefs = build_reference_pool(full, REF_EXPRS, settings=None)
    harvested = []
    for expr in survivors:
        rep = simulate(expr, full, frefs, (), settings=None)
        if rep.get("error"):
            continue
        m = rep["metrics"]
        elig, promo = promotion_eligible(rep, TrialRegistry(), DSR_PROMOTE,
                                           CAMPAIGN_SCOPE)
        print(f"FULL {expr[:70]}: S={m['sharpe']:.2f} F={m['fitness']:.2f} "
              f"pass={rep['passed']} dsr={promo.get('dsr')} eligible={elig}", flush=True)
        if elig:
            harvested.append({"expr": expr, "sharpe": m["sharpe"],
                              "fitness": m["fitness"], "dsr": promo.get("dsr"),
                              "n_trials": promo.get("n_trials")})
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "gp_campaign.json"), "w") as f:
        json.dump({"population": pop, "generations": gens, "search_seed": seed,
                   "hof_size": len(hof), "holdout_survivors": len(survivors),
                   "harvested": harvested}, f, indent=2)
    print(f"HARVESTED {len(harvested)} DSR>=0.95 promotion-eligible candidates", flush=True)


if __name__ == "__main__":
    main()
