# Fastexp Agent Lab

A WorldQuant Brain-style **alpha competition simulator**: LLM-style Ideator/Optimizer
agent teams propose `fastexpr` alphas (sentiment, fundamentals, microstructure,
volatility, options IV skew), and every submission is scored against hard
Brain-like cutoffs on synthetic — or real — market data.

## Quickstart

```bash
pip install -r requirements.txt

# 3-round competition on synthetic data (1000 stocks, 2020-2022)
python run.py 3

# Same, on real market data (downloads via yfinance, cached under cache/)
python run.py 3 --real
```

Reports land in `reports/` (`round_<k>.json`, `leaderboard.json`).

## Layout

| File | What it is |
|---|---|
| `fastexpr.py` | Expression parser + evaluator: cross-sectional (`rank`, `zscore`, `winsorize`, `scale`, `group_neutralize`, `regression_neut`), time-series (`ts_mean`, `ts_decay_linear`, `ts_delta`, `ts_zscore`, `ts_rank`, …), trading (`trade_when`, `hump`, `truncate`) |
| `data_gen.py` | Synthetic panel generator (seeded, with regime switches) |
| `real_data.py` | Real OHLCV panel via yfinance (+ synthetic IV/sentiment overlays) |
| `simulator.py` | Backtester: Sharpe, Fitness, Turnover, Drawdown, concentration, sub-universe Sharpe, self-correlation vs reference pool |
| `agents.py` | Ideator/Optimizer teams + per-component diagnostic discipline (standalone test → hypothesis → justified combine → compare vs legs; subtraction before machinery) |
| `competition.py` | Round runner, pass-only scoring (Fitness, then Sharpe), leaderboard |
| `run.py` | Entry point + fixed 8-alpha reference pool |
| `scripts/` | Dev utilities: `calibrate.py`, `verify_t2.py`, `debug_data.py`, `debug_conc.py` (run from repo root, e.g. `python scripts/calibrate.py`) |
| `reports/report.md` | Exploration log from the 5-agent search-and-pair study |

## Submission cutoffs

Sharpe > 1.25, Fitness > 1.0, Turnover 1–70%, weight concentration ≤ 10%,
sub-universe Sharpe > 0.80, self-correlation < 0.7 (or Sharpe ≥ 1.10× the
correlated reference). Only alphas passing **all** cutoffs score.

## Notes

- Synthetic data inverts the real-market debt/assets effect (see `reports/report.md`); treat fundamental signs as environment-specific.
- `adv20` is dollar-volume: liquidity-gated trading should use `(volume*close) > adv20`, not `volume > adv20`.
- Tested with Python 3.13, numpy 2.2.6, yfinance 0.2.65.
