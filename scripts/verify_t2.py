import numpy as np
from data_gen import generate
from simulator import simulate, build_reference_pool, failed_criteria
from run import REF_EXPRS
from fastexpr import parse, to_expr, Env, eval_node

EXPR = "rank(-ts_zscore(returns, 10)) - rank(ts_backfill(implied_volatility_put_10, 15) / ts_backfill(implied_volatility_call_10, 15))"

panel = generate(seed=11, n_stocks=1000)
refs = build_reference_pool(panel, REF_EXPRS)
print(f"Panel: {panel.fields['close'].shape}, dates {panel.dates[0]}..{panel.dates[-1]}")
for r in refs:
    print(f"  ref {r['name']}: sharpe {r['sharpe']:+.4f}")

def coverage(sig, dates):
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    out = {}
    N = sig.shape[1]
    for y in sorted(set(years[1:].tolist())):
        idx = np.where(years == y)[0]
        longs, shorts = [], []
        for t in idx:
            row = sig[t]
            m = np.isfinite(row)
            if m.sum() < 5:
                continue
            v = row[m]
            mu = v.mean()
            longs.append(int((v > mu).sum()))
            shorts.append(int((v < mu).sum()))
        if not longs:
            out[y] = (0,0,0,0)
        else:
            out[y] = (float(np.mean(longs)), int(np.min(longs)), float(np.mean(shorts)), int(np.min(shorts)))
    return out

def full_report(expr, label):
    rep = simulate(expr, panel, refs, [])
    print(f"\n===== {label} =====")
    print(f"EXPR: {expr}")
    if rep.get('error'):
        print(f"ERROR: {rep['error']}")
        return rep
    m = rep['metrics']
    c = rep['criteria']
    print(f"Sharpe={m['sharpe']:.4f} Fitness={m['fitness']:.4f} Turnover={m['turnover_pct']:.2f}% Returns={m['returns_pct']:.2f}% DD={m['drawdown_pct']:.2f}% Conc={m['weight_concentration_pct']:.2f}%")
    print(f"max_weight date={rep['max_weight_date']} stock={rep['max_weight_stock']}")
    print(f"sub_sharpe={c['subuniverse_sharpe']['value']} top_ref={rep.get('top_corr_ref')} (sharpe {rep.get('top_corr_ref_sharpe')}) self_corr={c['self_correlation']['value']}")
    for k,v in c.items():
        print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}: {v['value']} (need {v['requirement']})")
    print(f"  OVERALL passed={rep['passed']} FAILED={failed_criteria(rep)}")
    print("  yearly:")
    for y in rep['yearly']:
        print(f"    {y['year']}: Sharpe {y['sharpe']:+.3f} Fit {y['fitness']:.3f} TO {y['turnover_pct']:.2f}% Ret {y['returns_pct']:+.2f}%")
    # coverage
    try:
        sig = eval_node(parse(expr), Env(panel))
        sig = np.where(np.isinf(sig), np.nan, sig).astype(float)
        cov = coverage(sig, panel.dates)
        N = sig.shape[1]
        thr = max(5, N*0.01)
        print(f"  coverage (thr={thr}):")
        for y,(ml,nl,ms,ns) in cov.items():
            flag = "FLAG-THIN" if (nl<thr or ns<thr) else "ok"
            print(f"    {y}: mean_long {ml:.1f} min_long {nl} mean_short {ms:.1f} min_short {ns} [{flag}]")
    except Exception as e:
        print(f"  coverage error: {e}")
    return rep

rep0 = full_report(EXPR, "ITER 0 - forwarded alpha exact re-sim")

# Component diagnostics
for ce in ["rank(-ts_zscore(returns, 10))", "rank(ts_backfill(implied_volatility_put_10, 15) / ts_backfill(implied_volatility_call_10, 15))", "-rank(ts_backfill(implied_volatility_put_10, 15) / ts_backfill(implied_volatility_call_10, 15))"]:
    full_report(ce, f"COMPONENT: {ce[:60]}")

# Denominator audit for IV ratio leg
try:
    denom = eval_node(parse("ts_backfill(implied_volatility_call_10, 15)"), Env(panel))
    d = denom[np.isfinite(denom)]
    print(f"\nDenominator audit (ts_backfill(call_10,15)): min={d.min():.4f} p1={np.percentile(d,1):.4f} p5={np.percentile(d,5):.4f} median={np.median(d):.4f} frac<=1.0={(d<=1.0).mean():.4f}")
    num = eval_node(parse("ts_backfill(implied_volatility_put_10, 15)"), Env(panel))
    ratio = num/denom
    r = ratio[np.isfinite(ratio)]
    print(f"Ratio audit (put/call): min={r.min():.4f} max={r.max():.4f} p1={np.percentile(r,1):.4f} p99={np.percentile(r,99):.4f} median={np.median(r):.4f}")
except Exception as e:
    print(f"audit error {e}")
