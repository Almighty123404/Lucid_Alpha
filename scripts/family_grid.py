"""Family-targeted grid around the Brain-validated volume reversal."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_gen import generate
from simulator import simulate, build_reference_pool
from selection import TrialRegistry, promotion_eligible, DSR_PROMOTE, set_scope
from tests.conftest import REF_EXPRS

set_scope("family-reversal-grid")
panel = generate(seed=11, n_stocks=1000)
refs = build_reference_pool(panel, REF_EXPRS, settings=None)
print(f"{'expr':<72}{'S':>7}{'F':>7}{'TO%':>7}{'sub':>7}{'corr':>7}  pass elig dsr", flush=True)
for z in (20, 40, 60):
    for m in (80, 160, 240):
        for dec in (None, 5, 10):
            base = f"-ts_zscore(returns * volume / ts_mean(volume, {m}), {z})"
            e = base if dec is None else f"ts_decay_linear({base}, {dec})"
            rep = simulate(e, panel, refs, (), settings=None)
            if rep.get("error"):
                print(f"{e[:70]:<72} ERROR {rep['error']}", flush=True)
                continue
            mt = rep["metrics"]
            el, pr = promotion_eligible(rep, TrialRegistry(), DSR_PROMOTE, "family-reversal-grid")
            print(f"{e[:70]:<72}{mt['sharpe']:>7.2f}{mt['fitness']:>7.2f}"
                  f"{mt['turnover_pct']:>7.1f}{rep['criteria']['subuniverse_sharpe']['value']:>7.2f}"
                  f"{rep['criteria']['self_correlation']['value']:>7.2f}  {str(rep['passed']):>5} {str(el):>5} {pr.get('dsr')}",
                  flush=True)
