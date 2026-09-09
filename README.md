# Fastexp Agent Lab

Fastexp is a reproducible, artifact-first quantitative research laboratory for
testing alpha expressions, simulating portfolio behavior, calibrating synthetic
data against real observations, and running Brain-style research competitions.

The project combines:

- A `fastexpr` parser and vectorized evaluator for cross-sectional, time-series,
  group, vector, trading, fundamentals, news, sentiment, and options features.
- A seeded synthetic market-panel generator with stress regimes, delistings,
  point-in-time fundamentals, sparse news, and options coverage.
- Optional real-market data through `yfinance`.
- A point-in-time data engine backed by DuckDB, including bi-temporal revision
  logs and an offline-first SEC Company Facts adapter.
- A Brain-shaped backtest pipeline with frozen promotion gates.
- Fast Gate and High-Fidelity execution lenses.
- Agent teams that propose, diagnose, optimize, and compare alpha expressions.
- Deflated Sharpe, multiple-testing, probability-of-backtest-overfitting, and
  promotion-lineage controls.
- Privacy-aware run artifacts, static reports, and an interactive dashboard.

Fastexp is a research and simulation tool. It is not a trading system, a
brokerage integration, investment advice, or an official WorldQuant Brain
implementation.

[![CI](https://github.com/Almighty123404/Fastexp-Agent-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Almighty123404/Fastexp-Agent-Lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

## Table of Contents

- [Project Status](#project-status)
- [Scope and Design Principles](#scope-and-design-principles)
- [Architecture](#architecture)
- [Capabilities](#capabilities)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Alpha Expressions](#alpha-expressions)
- [Simulation and Configuration](#simulation-and-configuration)
- [Command-Line Interface](#command-line-interface)
- [Web Studio & Python SDK](#web-studio--python-sdk)
- [Competition Workflow](#competition-workflow)
- [Run Artifacts](#run-artifacts)
- [Dashboard and Reports](#dashboard-and-reports)
- [Data and Point-in-Time Safety](#data-and-point-in-time-safety)
- [Calibration](#calibration)
- [Testing and Continuous Integration](#testing-and-continuous-integration)
- [Repository Layout](#repository-layout)
- [Privacy, Copyright, and Data Governance](#privacy-copyright-and-data-governance)
- [Limitations and Deferred Work](#limitations-and-deferred-work)
- [Author and License](#author-and-license)
- [References](#references)

## Project Status

The implemented research stack is operational and covered by the test suite.
The latest verified baseline is:

- Python 3.13
- 134 tests passed
- 6 warnings, with no test failures
- Pinned NumPy, yfinance, pytest, Numba, llvmlite, and DuckDB dependencies
- GitHub Actions CI on Python 3.13
- Docker test image available through `Dockerfile`

The repository is intentionally research-oriented. Passing the local suite does
not establish live-trading readiness, data-vendor compliance, or profitability.

## Scope and Design Principles

### Gate invariance

Existing promotion gates must not be loosened to improve pass rates. Thresholds
are centralized in `config.py`, versioned with `GATE_VERSION`, and protected by
`tests/test_gates_invariant.py`. Tightening a gate is allowed; silently weakening
one is a defect.

### Artifact-first research

Every CLI simulation can produce a complete run directory containing machine-
readable JSON/CSV source data and human-readable reports. Reports are views over
those artifacts, not the primary record of a result.

### Point-in-time correctness

Fundamentals and revisions are served using publication/effective dates. A
backtest must not use a value that was not available at the simulated decision
time.

### Diagnostic separation

Raw PnL, the linear cost lens, and High-Fidelity execution PnL are reported
separately. Diagnostic cost and impact calculations never change the frozen
promotion gates.

### Privacy by default

Public artifacts contain an SHA-256 `expression_hash`, not the original alpha
expression. The original expression is written only when
`--include-private` is explicitly supplied.

### Reproducibility

Seeds, settings, dataset identifiers, engine metadata, Git SHA, and generated
timestamps are recorded in the run manifest and provenance files. Synthetic
runs are reproducible for a fixed code revision and dependency environment.

## Architecture

```text
Alpha text / JSONL
        |
        v
fastexpr parser and evaluator
        |
        v
Synthetic provider | Real provider | SEC/PIT provider
        |
        v
Simulation pipeline
  universe -> pasteurization -> neutralization -> decay
  -> truncation -> delay -> book normalization -> PnL
        |
        +--> frozen promotion gates
        +--> statistical selection and lineage controls
        +--> raw/cost/High-Fidelity diagnostic series
        |
        v
Artifact builder
  JSON + CSV + SVG report + Plotly dashboard + manifest
        |
        +--> audit
        +--> compare
        +--> calibration
        +--> competition leaderboard
```

The simulator does not import a plotting library or a browser runtime. The
rendering boundary is `reporting.py`; this keeps the engine usable in minimal
environments and prevents visualization logic from changing gate calculations.

## Capabilities

### Expression language

`fastexpr.py` implements a parser/evaluator with:

- Arithmetic, comparisons, boolean expressions, negation, and ternary syntax.
- Multi-statement scripts with local intermediate values.
- Cross-sectional operators such as `rank`, `zscore`, `quantile`, `scale`, and
  `bucket`.
- Time-series operators such as `ts_mean`, `ts_decay_linear`, `ts_delta`,
  `ts_zscore`, `ts_rank`, `ts_std_dev`, and backfill operators.
- Group operators such as `group_neutralize` and `group_vector_neut`.
- Vector operators such as `vec_avg`, `vec_sum`, `vec_max`, `vec_min`, and
  `vector_neut`.
- Trading operators such as `trade_when`, `hump`, and `truncate`.
- Unit checking with `unitHandling=VERIFY` or `unitHandling=OFF`.
- Parser depth and expression-complexity guards.

### Synthetic data

`data_gen.py` creates deterministic panels with:

- Seeded multi-asset OHLCV data.
- GJR-GARCH-t style volatility behavior.
- Dated stress and bear regimes.
- Portable panic behavior for stress testing.
- Quarterly fundamentals served through point-in-time revision grids.
- Tenor-graded options and implied-volatility fields.
- Sparse news and sentiment fields.
- Approximately 2% annual delisting behavior.
- Validation through `validate_panel`.

Synthetic data is useful for deterministic tests and controlled experiments. It
is not a substitute for a licensed historical market database.

### Real data

`real_data.py` and `providers.py` support a `yfinance`-backed provider. Real
data is downloaded on demand, cached locally under `cache/`, and is not treated
as a redistribution asset. The real-data path also provides synthetic overlays
for fields that are not supplied by the selected market-data source. These
overlays must not be mistaken for vendor-observed fundamentals, news, or
options data.

### Simulation pipeline

`simulator.py` follows the configured sequence:

1. Parse and evaluate the expression.
2. Apply universe selection and liquidity handling.
3. Apply pasteurization and missing-value handling.
4. Apply neutralization.
5. Apply decay and truncation.
6. Apply the configured signal delay.
7. Normalize the portfolio book.
8. Calculate daily PnL and turnover.
9. Calculate performance, risk, sub-universe, yearly, and correlation metrics.
10. Evaluate the frozen promotion gates.

Reported measures include Sharpe, Fitness, turnover, returns, drawdown, margin,
weight concentration, concentration dates, sub-universe Sharpe, rolling
self-correlation, yearly and IS/TEST/OS summaries, and long/short counts.

### Promotion gates

The default gate values in `config.py` are:

| Gate | Default rule |
|---|---:|
| Sharpe | Strictly greater than `1.25` |
| Fitness | Strictly greater than `1.0` |
| Turnover | Inclusive range `1%` to `70%` |
| Weight concentration | Maximum `10%` of book |
| Self-correlation | Maximum `0.70`, with an escape test requiring a `1.10x` Sharpe improvement over every correlated reference |
| Self-correlation window | Four years of daily PnL changes |
| Sub-universe Sharpe | `0.75 * sqrt(sub_size / alpha_size) * alpha_sharpe` |
| Fitness turnover floor | `max(turnover, 0.125)` |

Sub-universe membership and universe sizes are explicit approximations unless
dated constituent data is supplied. Every constant is centralized, and
provenance labels distinguish sourced values from values requiring empirical
validation.

### Execution modes

| Mode | Purpose | Affects gates? |
|---|---|---|
| `fast_gate` | Fast Brain-shaped screening and promotion calculations | Yes, this is the gate path |
| `high_fidelity` | Fills, spread, commissions, participation, and impact diagnostics | No, diagnostic only |

The High-Fidelity layer exposes spread, urgency, permanent impact, temporary
impact, participation, commission, and optional portfolio-notional settings.
These defaults are explicitly approximate until calibrated against real
implementation-shortfall data.

### Agent research loop

`agents.py` and `competition.py` implement a structured research loop:

- Ideator teams propose expressions from balanced template families.
- Optimizers diagnose failed components before changing them.
- Component legs are tested independently before combination.
- Combined expressions are compared with their individual legs.
- Duplicate expression skeletons are filtered across rounds.
- Near-miss candidates can enter a bounded rescue pool.
- Repeated dead ends are written to a lessons log.
- Passed gates are still subject to lineage and statistical promotion checks.

`gp.py` provides grammar-constrained genetic-programming proposals and a Pareto
hall of fame. GP proposes candidates; it does not bypass the simulator gates.

### Statistical selection controls

`selection.py` contains controls intended to reduce false discoveries and
multiple-testing optimism:

- Bailey and Lopez de Prado Deflated Sharpe Ratio.
- Promotion blocking when trial lineage or trial count is missing.
- Probability of Backtest Overfitting using CSCV-style analysis.
- Stationary bootstrap utilities.
- Romano-Wolf StepM family-wise error control.
- Scope-separated trial registries for competitions and GP campaigns.
- Expected-real adjustment from empirical calibration families.

These controls improve research hygiene; they do not prove that an alpha will
work out of sample.

### Risk and market-model modules

- `risk.py`: diagnostic risk statistics, never a replacement for promotion gates.
- `iv_surface.py`: pure-NumPy SSVI-style surface construction and no-arbitrage
  sufficient-condition checks.
- `fundamentals.py`: fundamental transformations and accounting features.
- `sec_pit.py`: offline-first SEC Company Facts/submissions ingestion with an
  explicit User-Agent requirement for network access.
- `pit.py`: bi-temporal revision storage and leak-free DuckDB lookups.
- `kernels.py`: Numba top-N execution kernels with a NumPy fallback.

## Installation

### Requirements

- Python 3.13.
- A virtual environment is recommended.
- Network access is needed only for optional real-data downloads or SEC ingestion.
- Synthetic tests do not require market-data network access.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt` for reproducible CI and local
testing:

| Package | Use |
|---|---|
| NumPy | Array computation and numerical simulation |
| yfinance | Optional real historical market-data provider |
| pytest | Test suite |
| Numba | Optional accelerated execution kernels |
| llvmlite | Pinned Numba backend |
| DuckDB | Point-in-time and SEC revision queries |

### Docker

The supplied Dockerfile installs the pinned dependencies and runs the test
suite by default:

```bash
```

## Quickstart

Run a three-round synthetic competition using the fixed reference pool:

```bash
python run.py 3
```

Run the same workflow with the optional real-data provider:

```bash
python run.py 3 --real
```

Real-data mode downloads through `yfinance`, uses a smaller default panel, and
may be affected by provider availability, historical revisions, throttling, and
provider terms.

Load simulation settings from JSON:

```bash
python run.py 3 --settings my_settings.json
```

Example settings file:

```json
{
  "region": "USA",
  "universe": "TOP3000",
  "delay": 1,
  "decay": 4,
  "truncation": 0.1,
  "neutralization": "INDUSTRY",
  "pasteurization": "ON",
  "nanHandling": "ON",
  "unitHandling": "VERIFY",
  "language": "FASTEXPR",
  "instrumentType": "EQUITY"
}
```

Competition outputs are written under `reports/`, including round summaries,
leaderboards, trial records, walk-forward results, and lessons. Regenerable
outputs are ignored by Git according to `.gitignore`.

## Alpha Expressions

Create a text file such as `alpha.txt`:

```text
rank(ts_zscore(close, 120))
```

Other examples:

```text
-ts_zscore(returns, 21)
rank(ts_backfill(ebitda, 63) / ts_backfill(assets, 63))
rank(volume / ts_mean(volume, 20))
```

Expressions may use fields generated by the selected provider and operators
implemented by `fastexpr.py`. Unsupported fields or invalid unit combinations
fail explicitly rather than silently creating a result.

When writing multi-step expressions, use the parser's supported statement and
local-value syntax. Keep expressions small enough for the parser and GP depth
guards. The canonical expression identity in public artifacts is its SHA-256
hash.

## Simulation and Configuration

`SimulationSettings` is the first-class configuration schema in `config.py`.
The default values are:

```text
region          = USA
universe        = TOP3000
truncation      = 0.1
neutralization  = INDUSTRY
pasteurization  = ON
nanHandling     = ON
unitHandling    = VERIFY
language        = FASTEXPR
instrumentType  = EQUITY
```

`ExecutionSettings` is separate because High-Fidelity execution is diagnostic.
Its controls include `spread_bps`, `urgency`, `lambda_perm`, `eta_temp`,
`alpha`, `max_participation`, `commission_bps`, and
`portfolio_notional`.

Important conventions:

- Annualized PnL is measured as a fraction of invested book, where invested book
  is `book / 2`.
- Fitness uses the exact `max(turnover, 0.125)` floor.
- Delay is a strict weight lag; the default delay is `1`.
- Turnover uses the gross traded-weight convention.
- `adv20` represents dollar volume. Liquidity checks should use
  `(volume * close) > adv20`.
- High-Fidelity impact and cost assumptions are not promotion gates.

## Command-Line Interface

The artifact-first CLI is available without installing a separate package:

```bash
python -m fastexp.cli COMMAND [OPTIONS]
```

### Simulate one expression

```bash
python -m fastexp.cli simulate \
  --expr-file alpha.txt \
  --seed 11 \
  --dataset synthetic \
  --mode fast_gate \
  --output reports/runs
```

Options:

- `--expr-file`: required expression text file.
- `--seed`: deterministic panel seed; default `11`.
- `--dataset`: `synthetic` or `real`.
- `--mode`: `fast_gate` or `high_fidelity`.
- `--output`: artifact root; default `reports/runs`.
- `--include-private`: include the raw expression in provenance files.
- `--verbose`: expose additional diagnostic output.

### Batch simulation

Create `alphas.jsonl` with one JSON object per line:

```json
{"name": "momentum", "expression": "rank(ts_zscore(close, 120))"}
{"name": "reversal", "expr": "-ts_zscore(returns, 21)"}
```

Run it:

```bash
python -m fastexp.cli batch \
  --input alphas.jsonl \
  --seed 11 \
  --dataset synthetic \
  --mode fast_gate
```

Blank lines are skipped. Each valid expression receives its own run artifact.

### Compare two runs

`compare` accepts run-directory paths and writes a JSON metric comparison:

```bash
python -m fastexp.cli compare \
  --run-id reports/runs/RUN_A \
  --run-id-b reports/runs/RUN_B
```

The output includes metric A, metric B, and the numeric delta where available.

### Audit a run

Audit by run-directory path or by run ID beneath the configured output root:

```bash
python -m fastexp.cli audit --run-id reports/runs/RUN_A
python -m fastexp.cli audit --run-id RUN_A --output reports/runs
```

The audit checks the manifest, required files, private-data flag, and expression
leakage in provenance.

## Web Studio & Python SDK

Fastexp ships a zero-CLI **Web Studio** and a matching **Python SDK** so
researchers can build, run, track, and compare simulations without touching the
shell — while producing the exact same immutable `reports/runs/<run_id>/`
artifacts as the CLI. Both wrap the existing engine; neither modifies the
simulation math or gate logic.

### Web Studio

Start the local server:

```bash
python -m fastexp.server                # http://127.0.0.1:8000
python -m fastexp.server --port 9000
```

The studio provides:

- **Alpha Formula Playground** — a Monaco editor with syntax highlighting,
  operator/field autocomplete, real-time linting (balanced parentheses and
  unknown-name checks), and a slide-out Operator Reference drawer.
- **Preset Templates** — momentum, mean reversion, volume shock, quality, low/high
  volatility, and asset turnover.
- **Parameter Configurator** — dataset, mode, seed, universe, neutralization,
  truncation slider, decay, and an include-private toggle with a warning badge.
- **Batch Upload** — a drag-and-drop zone accepting `.txt` (one expression per
  line) or `.jsonl` (batch lists).
- **Job Queue** — a live table of `QUEUED / RUNNING / PASS / FAIL / BLOCKED /
  ERROR / STOPPED` jobs with per-job Stop and Open Report controls.
- **Live Console** — a collapsible, color-coded terminal streaming execution
  output in real time over a WebSocket, with an Open Interactive Report action
  on completion.
- **Runs Browser** — every run listed with Dashboard and Report links, served
  directly from the artifact directory.

Backend endpoints (local-first FastAPI service, `fastexp/server.py`):

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/simulate` | Start a single-expression run |
| `POST /api/v1/batch` | Start a batch from a JSON list of expressions |
| `POST /api/v1/batch/upload` | Start a batch from an uploaded `.txt`/`.jsonl` file |
| `GET /api/v1/runs` | List artifact runs |
| `GET /api/v1/jobs` / `GET /api/v1/jobs/{id}` | List / poll execution jobs |
| `GET /api/v1/ws/logs/{job_id}` | WebSocket live stdout/stderr stream |
| `POST /api/v1/runs/stop` | Terminate a running job |
| `GET /api/v1/operators` / `GET /api/v1/templates` | Editor autocomplete/docs data |

### Python SDK

For Jupyter notebooks and scripts, the same engine is available as:

```python
import fastexp as fe

run = fe.simulate(expr="rank(ts_zscore(close, 120))",
                  dataset="synthetic", mode="fast_gate", seed=11)

run.show_dashboard()      # interactive Plotly dashboard inline (Jupyter)
run.show_report()         # static SVG report (offline-safe, always renders)
df = run.equity_curve     # pandas.DataFrame (list-of-dicts fallback without pandas)
print(run.run_id, run.status, run.metrics)
```

A `Run` exposes `run_id`, `status`, `passed`, `metrics`, `criteria`,
`manifest`, `summary`, and the tabular artifacts `equity_curve`, `drawdown`,
`exposures`, `yearly`, and `periods`. `fe.batch(expressions)` returns a list of
`Run` objects. SDK runs reuse `simulator.simulate` + `reporting.build_run_artifact`,
so they are byte-compatible with CLI and Studio runs.

> The SDK is lazy-loaded: `import fastexp` stays lightweight, and `pandas` /
> `IPython` are only imported when `Run` display or DataFrame access is used.
> The Web Studio adds `fastapi`, `uvicorn`, `websockets`, `python-multipart`, and
> `httpx` (test client) to `requirements.txt`; the core engine and test suite
> remain dependency-light.

## Competition Workflow

`run.py` is the high-level competition entry point:

1. Build a synthetic or real panel.
2. Build the fixed reference alpha pool.
3. Create seeded Ideator/Optimizer teams.
4. Run the requested number of rounds.
5. Diagnose and optimize proposals.
6. Apply all simulation gates.
7. Apply lineage, DSR, expected-real, and multiple-testing promotion controls.
8. Record passers, near misses, lessons, and leaderboard results.

The competition is pass-only: candidates that fail required gates do not enter
the promoted history merely because they have a high score on another metric.
An alpha that passes raw gates can still be blocked if its trial lineage or
statistical evidence is insufficient.

## Run Artifacts

Each call to `build_run_artifact` creates:

```text
reports/runs/<run_id>/
|-- manifest.json
|-- summary.json
|-- gates.json
|-- provenance.json
|-- metrics.csv
|-- yearly.csv
|-- periods.csv
|-- equity_curve.csv
|-- drawdown.csv
|-- exposures.csv
|-- report.html
|-- dashboard.html
`-- charts/
    |-- equity_curve.html
    |-- drawdown.html
    |-- rolling_sharpe.html
    |-- turnover.html
    `-- exposures.html
```

### JSON files

- `manifest.json`: run ID, Git SHA, dataset ID, expression hash, settings,
  seed, mode, timestamp, privacy flag, and status.
- `summary.json`: executive metrics, criteria, periods, yearly results, risk,
  High-Fidelity diagnostics, and optional extra metrics.
- `gates.json`: gate-by-gate values, thresholds, and pass/fail decisions.
- `provenance.json`: reproducibility metadata and optional private expression.

### CSV files

- `metrics.csv`: scalar performance and risk metrics.
- `yearly.csv`: yearly performance and counts.
- `periods.csv`: IS/TEST/OS or configured period results.
- `equity_curve.csv`: date-aligned raw, cost, and optional diagnostic PnL.
- `drawdown.csv`: date-aligned drawdown series.
- `exposures.csv`: gross, long, short, net, active-name, maximum-weight, and
  concentration series.

### Status values

Artifacts can be marked `PASS`, `FAIL`, `BLOCKED`, or `ERROR`. `PASS` means the
simulation gates passed; promotion controls and downstream research decisions
may still impose additional restrictions.

## Dashboard and Reports

`report.html` is a static SVG report designed to open offline without Plotly or
other browser dependencies.

`dashboard.html` is a read-only Plotly dashboard generated from the same run
data. It provides:

- Overview gate chips and decision status.
- Equity and PnL lenses.
- Raw, linear-cost, and High-Fidelity diagnostic overlays.
- Drawdown, rolling Sharpe, and turnover.
- Gross, long, short, and net exposures.
- Concentration and active-name diagnostics.
- Yearly Sharpe bars.
- Monthly PnL heatmap.
- Daily PnL histogram.
- Responsive layout for desktop and mobile widths.
- Dark institutional visual theme.
- URL-synchronized tabs, for example `dashboard.html?tab=equity`.
- Keyboard tab shortcuts `1` through `5`.
- Plotly hover, scroll zoom, and range sliders.

Plotly is loaded from the pinned CDN URL at runtime. If external browser assets
are unavailable, use `report.html` or the CSV/JSON files directly.

The project does not currently include a web server, `/runs` catalog, browser
route system, or a multi-run web comparison workspace. Run comparison is
available through the CLI and generated JSON.

## Data and Point-in-Time Safety

### PIT fundamentals

`pit.py` stores revision logs with effective and publication timestamps. Lookup
functions return the latest value that was actually available at the simulated
decision time. `validate_pit` checks ordering and leak-free behavior.

### SEC ingestion

`sec_pit.py` is deliberately separate from panel generation. It can ingest SEC
Company Facts and submissions JSON into append-only DuckDB tables. Network access
is opt-in and requires an explicit application-identifying User-Agent containing
a contact email. No API token is accepted or required by the adapter.

### Real-data provider

The provider interface in `providers.py` separates data acquisition from the
simulator. This makes it possible to replace `yfinance` with a licensed vendor,
an internal export, or a logged research dataset without changing the engine
contract.

## Calibration

`calibration.py` stores paired simulator predictions and real observations. It
supports:

- JSONL calibration records.
- Summary statistics by family.
- Expected-real metric adjustments.
- Turnover-law calibration.
- Family classification for promotion diagnostics.

The repository includes receiver paths for future Brain export data, including
coverage and earnings-event inputs. Those receivers are intentionally local and
gitignored until real records are supplied. User-provided private calibration
records must not be committed.

The calibration workflow is empirical rather than authoritative. An adjustment
table is only as credible as the number, quality, and representativeness of its
paired observations.

## Testing and Continuous Integration

Run the complete suite:

```bash
python -m pytest tests/ -q
```

Run a focused subsystem suite:

```bash
python -m pytest tests/test_reporting.py -q
python -m pytest tests/test_gates_invariant.py tests/test_simulator.py -q
```

Compile the main Python entry points:

```bash
python -m py_compile reporting.py fastexp/cli.py
```

The test suite covers:

- Expression parsing and evaluation.
- GP generation and Pareto behavior.
- Synthetic data generation and validation.
- Agent logic and diagnosis.
- Simulation metrics and gate invariants.
- Dual-mode execution.
- Fundamentals, options, PIT, and SEC adapters.
- Risk calculations and statistical selection controls.
- Artifact schemas, privacy redaction, CSV alignment, comparison, and CLI use.

GitHub Actions installs `requirements.txt` on Ubuntu with Python 3.13 and runs
the full test suite. Tests are synthetic-only and do not require network access.

## Repository Layout

| Path | Responsibility |
|---|---|
| `config.py` | Versioned settings, cutoffs, provenance tags, gate invariants, and execution settings |
| `fastexpr.py` | Parser, AST, evaluator, operators, units, and expression guards |
| `data_gen.py` | Seeded synthetic market-panel generator and validation |
| `providers.py` | Pluggable synthetic and real data-provider interface |
| `real_data.py` | Optional `yfinance` historical panel provider and local cache |
| `pit.py` | Bi-temporal revisions and DuckDB point-in-time lookups |
| `sec_pit.py` | Offline-first SEC Company Facts/submissions adapter |
| `fundamentals.py` | Fundamental feature transforms |
| `iv_surface.py` | SSVI-style implied-volatility surface utilities |
| `simulator.py` | Main backtest and gate-evaluation pipeline |
| `kernels.py` | Numba top-N kernels with NumPy fallback |
| `risk.py` | Diagnostic risk calculations |
| `gp.py` | Grammar-constrained genetic-programming proposals |
| `selection.py` | DSR, PBO, bootstrap, StepM, trial registry, and promotion controls |
| `calibration.py` | Paired simulator/real calibration records and expected-real adjustments |
| `agents.py` | Ideator/Optimizer agents and component-level diagnosis |
| `competition.py` | Multi-round competition, lessons, promotion, and leaderboard |
| `run.py` | High-level competition entry point and reference pool |
| `reporting.py` | Artifact writer, static SVG report, Plotly dashboard, compare, and audit support |
| `fastexp/cli.py` | `simulate`, `batch`, `compare`, and `audit` commands |
| `fastexp/server.py` | Local FastAPI Web Studio service (async subprocess orchestration, REST + WebSocket) |
| `fastexp/sdk.py` | Python SDK: `simulate`/`batch` returning a `Run` with DataFrame and inline-display accessors |
| `fastexp/web/` | Web Studio frontend (Monaco editor, parameter form, queue, console) |
| `tests/` | Unit, integration, governance, reporting, studio/SDK tests |
| `scripts/` | Calibration, walk-forward, GP campaign, family-grid, debugging, and context utilities |
| `reports/` | Regenerable competition outputs and research logs |
| `calibration/` | Calibration schemas and local JSONL records; private files are ignored |
| `cache/` | Local downloaded market-data snapshots; ignored by Git |
| `risk_register.json` | Project risk and evidence register |
| `Dockerfile` | Reproducible test container |
| `LICENSE` | MIT license for original Fastexp source |
| `.github/workflows/ci.yml` | Continuous integration workflow |

## Privacy, Copyright, and Data Governance

### Expression privacy

The default artifact path writes only an expression hash. Use
`--include-private` only when the output directory is access-controlled and the
expression owner has authorized storage of the original text.

### Secrets

Do not put API keys, credentials, proprietary exports, private Brain records,
or confidential expressions in tracked files. The repository ignores local
cache, private calibration, unsubmitted records, generated runs, and
`context.json`.

### Third-party data

Downloaded market data is not automatically redistributable merely because the
Python client is open source. Users are responsible for complying with the
terms of the data provider, including Yahoo Finance/yfinance usage conditions,
SEC rate limits, applicable exchange terms, and any internal data license.

### Names and trademarks

WorldQuant and Brain are referenced only to describe the research inspiration
and compatibility target. Fastexp is independent and is not endorsed by,
sponsored by, or affiliated with WorldQuant. No WorldQuant source code, private
records, credentials, logos, or proprietary documentation are included.

The source code in this repository was written for this project. Third-party
packages remain under their own licenses and are not relicensed by the Fastexp
MIT license. Plotly is loaded from its public CDN and is not copied into this
repository. CSS font-family names are references only; no font files are
bundled.

## Limitations and Deferred Work

The following are known limitations or intentionally deferred work, not hidden
features:

- The implementation is a Brain-style proxy, not an official Brain engine.
- Synthetic fields and approximated universe membership require empirical
  calibration against licensed historical data.
- Real-data mode uses `yfinance` and may not provide point-in-time fundamentals,
  historical index membership, corporate-action completeness, or institutional
  options/news coverage.
- High-Fidelity execution is diagnostic until implementation-shortfall pairs are
  available.
- Additional real pairs are needed to close remaining sim-versus-real magnitude
  gaps for thin single legs and to recalibrate pair correlations.
- G6 coverage and G7 earnings-event receivers await supplied Brain exports.
- Some extended codebooks and tenors remain deferred, including additional
  `iv_120/20`, `rnd`, `fnd6`, and flag fields.
- A web server, `/runs` catalog, multi-run browser comparison, and artifact-tree
  web inspector are not implemented; static per-run dashboards and CLI compare
  are the supported interfaces.
- There is no live order-routing, broker, execution, or capital-management
  integration.
- No backtest result should be interpreted as a forecast, recommendation, or
  guarantee of future performance.

## Author and License

### Author and maintainer

**Abhinav Tiwari** (https://github.com/Almighty123404) is the repository owner and maintainer. 

### License

Copyright (c) 2026 Almighty123404.

This project is released under the MIT License. The complete legal text is in
[`LICENSE`](LICENSE). The MIT license applies to the original Fastexp source in
this repository; it does not change the licenses or terms of third-party
packages, external data, or referenced research.

For a production distribution, preserve the `LICENSE` file, retain copyright
and attribution notices, and include the relevant third-party notices required
by the dependency licenses.

## References

The project uses the following public documentation and research references for
concepts, implementation guidance, or attribution. References do not imply
affiliation or endorsement.

### Research and methodology

1. WorldQuant Brain platform, public platform overview and terminology:
   <https://platform.worldquantbrain.com/>
2. Bailey, D. H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio:
   Correcting for Selection Bias, Backtest Overfitting, and Non-Normality":
   <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
3. Bailey, D. H., Borwein, J. M., de Prado, M. L., and Zhu, Q. J. (2017),
   "The Probability of Backtest Overfitting":
   <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>
4. Romano, J. P. and Wolf, M. (2005), "Stepwise Multiple Testing as Formalized
   Data Snooping": <https://doi.org/10.1198/016214505000000281>
5. Gatheral, J. and Jacquier, A. (2014), "Arbitrage-free SVI volatility
   surfaces": <https://doi.org/10.1080/14697688.2013.819986>
6. Almgren, R. and Chriss, N. (2001), "Optimal Execution of Portfolio
   Transactions": <https://doi.org/10.1093/rfs/14.2.267>

### Official technical documentation

1. Python documentation: <https://docs.python.org/3/>
2. NumPy documentation: <https://numpy.org/doc/stable/>
3. Numba documentation: <https://numba.readthedocs.io/>
4. DuckDB documentation and ASOF joins:
   <https://duckdb.org/docs/stable/guides/sql_features/asof_join>
5. pytest documentation: <https://docs.pytest.org/>
6. Plotly JavaScript documentation: <https://plotly.com/javascript/>
7. yfinance project and license information:
   <https://github.com/ranaroussi/yfinance>
8. SEC EDGAR APIs and developer documentation:
   <https://www.sec.gov/edgar/sec-api-documentation>
9. SEC Company Facts endpoint documentation:
   <https://www.sec.gov/edgar/sec-api-documentation>

### Related public implementation reference

The S1 relative sub-universe formula was cross-checked against the public
`dafu-zhu/alpha-lab` project during research. Fastexp does not copy source code
from that project:

<https://github.com/dafu-zhu/alpha-lab>

Dependency license text belongs to the respective dependency authors. Consult
the installed package metadata and upstream repositories before redistributing
a packaged environment.
