import sys

from data_gen import generate
from simulator import build_reference_pool
from agents import make_teams
from competition import Competition

try:
    from real_data import generate_real_panel
    HAS_REAL = True
except ImportError:
    HAS_REAL = False

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


def main():
    use_real = "--real" in sys.argv
    if use_real:
        sys.argv.remove("--real")
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if use_real:
        if not HAS_REAL:
            print("yfinance not available, falling back to synthetic.")
            panel = generate(seed=11, n_stocks=1000)
        else:
            print("Fetching real market data via yfinance ...")
            panel = generate_real_panel(seed=11, n_stocks=200)
            print(f"Real panel: {panel.fields['close'].shape[1]} tickers x {len(panel.dates)} days "
                  f"({str(panel.dates[0])} to {str(panel.dates[-1])})")
    else:
        panel = generate(seed=11, n_stocks=1000)
    refs = build_reference_pool(panel, REF_EXPRS)
    print(f"Reference pool: {len(refs)} alphas")
    for r in refs:
        print(f"  {r['name']}: Sharpe {r['sharpe']:+.2f}")
    teams = make_teams(seed=5)
    comp = Competition(panel, refs, teams, rounds=rounds, max_opt_iter=5)
    comp.run()


if __name__ == '__main__':
    main()
