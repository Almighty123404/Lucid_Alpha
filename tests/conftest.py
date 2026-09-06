"""Shared fixtures: small fast synthetic panel (no network, no large panels)."""
import os

import pytest

import selection
from data_gen import generate
from simulator import build_reference_pool

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


@pytest.fixture(scope="session")
def panel():
    return generate(seed=11, n_stocks=100, start="2020-01-01", end="2020-12-31")


@pytest.fixture(scope="session")
def refs(panel):
    return build_reference_pool(panel, REF_EXPRS, settings=None)


# Suite hygiene: three shared panels per session (small/med/big) so test
# files never rebuild the DuckDB PIT grids redundantly. Added after profiling
# showed panel builds dominating suite time (target: whole suite < ~60s).
@pytest.fixture(scope="session")
def med_panel():
    return generate(seed=11, n_stocks=200, start="2020-01-01", end="2022-12-31")


@pytest.fixture(scope="session")
def med_refs(med_panel):
    return build_reference_pool(med_panel, REF_EXPRS, settings=None)


@pytest.fixture(scope="session")
def big_panel():
    return generate(seed=11, n_stocks=1000, start="2020-01-01", end="2022-12-31")


@pytest.fixture(autouse=True)
def _isolate_trials(tmp_path_factory):
    # Invariant-2 hygiene: tests must never pollute reports/trials.jsonl.
    # Each test session logs to a temp file instead.
    path = str(tmp_path_factory.mktemp("trials") / "trials.jsonl")
    old = selection.TRIALS_PATH
    selection.TRIALS_PATH = path
    yield
    selection.TRIALS_PATH = old
