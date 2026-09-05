from data_gen import generate
from simulator import build_reference_pool, simulate
from run import REF_EXPRS

panel = generate(seed=11, n_stocks=600)
refs = build_reference_pool(panel, REF_EXPRS)
for r in refs:
    print(f"{r['name']:<20} Sharpe {r['sharpe']:+.2f}")
print()
for e in [
    'ts_decay_linear(ts_mean(vec_avg(nws12_afterhsz_01l), 5), 10)',
    'rank(ts_delta(ebitda, 63) / assets)',
    '-(ts_backfill(implied_volatility_put_10, 10) / ts_backfill(implied_volatility_call_10, 10))',
    '-ts_zscore(returns, 5)',
    'ts_mean(returns, 120)',
    'rank(implied_volatility_put_60 - implied_volatility_call_60)',
]:
    rep = simulate(e, panel, refs)
    if rep.get('error'):
        print('ERR', e, rep['error'])
        continue
    m = rep['metrics']
    c = rep['criteria']
    print(f"{e[:60]:<62} Sh {m['sharpe']:+.2f} Fit {m['fitness']:.2f} TO {m['turnover_pct']:.0f}% "
          f"conc {m['weight_concentration_pct']:.1f}% sub {c['subuniverse_sharpe']['value']:+.2f} "
          f"corr {c['self_correlation']['value']:.2f} pass={rep['passed']}")
