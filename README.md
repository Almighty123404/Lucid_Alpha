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

# Custom simulation settings (see calibration/schema.json for the shape)
python run.py 3 --settings my_settings.json
```

Reports land in `reports/` (`round_<k>.json`, `leaderboard.json`).

> Pipeline notes (WorldQuant Brain-compatible simulation engine):
> - `simulate(expression, settings)` with `settings` matching the
>   `SimulationSettings` schema; every cutoff/formula constant lives in
>   `config.py` (SOURCED vs APPROX-marked) and is overridable.
> - Returns are annualized PnL as a fraction of invested = book/2; Fitness
>   uses the exact `max(Turnover, 0.125)` floor; delay is a strict weight
>   lag (delay=1 earns R[T+2] from signal T — legacy runs were delay-0
>   mechanics, so historical numbers shift slightly under the new default).
> - Recalibrate APPROX constants against real Brain results via
>   `calibration.py` + `calibration/records.jsonl`.

## Layout

| File | What it is |
|---|---|
| `config.py` | Versioned `SimulationSettings` schema (region/universe/delay/decay/truncation/neutralization/pasteurization/nanHandling/unitHandling) + all cutoffs/formula constants (SOURCED vs APPROX-marked), overridable per `simulate()` call |
| `fastexpr.py` | Expression parser + evaluator: multi-statement `;` scripts with locals, cross-sectional (`rank`, `zscore`, `quantile` with uniform/gaussian/cauchy drivers, `scale`, `bucket`), time-series (`ts_mean`, `ts_decay_linear`, `ts_delta`, `ts_zscore`, `ts_rank`, …), group (`group_neutralize`, `group_vector_neut`, …), vector (`vec_avg`, `vec_sum`, `vec_max`, `vec_min`), `vector_neut`, trading (`trade_when`, `hump`, `truncate`), unit VERIFY/OFF |
| `data_gen.py` | Synthetic panel generator (seeded, with regime switches; tenor-graded options coverage, quarterly fundamentals, skewed news coverage) |
| `real_data.py` | Real OHLCV panel via yfinance (+ synthetic IV/sentiment overlays) |
| `providers.py` | Pluggable `DataProvider` interface (`synthetic` / `real` backends, coverage audit) |
| `simulator.py` | Backtester pipeline (parse → evaluate → universe/pasteurization → neutralization → decay → truncation → delay → book normalize → PnL): Sharpe, Fitness, Turnover, Returns (invested = book/2), Drawdown, Margin, concentration + top dates, sub-universe Sharpe (delay-0 formula), rolling-2Y self-correlation on daily changes, IS/TEST/OS + yearly long/short counts |
| `calibration.py` + `calibration/` | Living calibration log: schema + JSONL records pairing real Brain results with simulator predictions |
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
