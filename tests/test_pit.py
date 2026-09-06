"""Phase 4 PIT tests: revision log, ASOF semantics, leakage invariant."""
import numpy as np

from pit import build_revision_log, pit_asof, validate_pit, DUCKDB_OK

rng = np.random.default_rng(0)
DAYS = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-04-01"),
                 dtype="datetime64[D]")
DAYS = DAYS[np.is_busday(DAYS)]
T = len(DAYS)
N = 20
Q = 2
PE = np.array([DAYS[19], DAYS[39]])
VQ = np.tile(np.arange(Q)[:, None] * 100.0, (1, N)) + np.arange(N) + 1000.0


def test_revision_log_shape_and_restatements():
    rows = build_revision_log(PE, VQ, "sales", rng, restate_p=1.0)
    assert len(rows) == Q * N * 2  # every cell restated at p=1
    r2 = [r for r in rows if r["rev_seq"] == 2][0]
    r1 = [r for r in rows if r["rev_seq"] == 1 and r["sid"] == r2["sid"]
          and r["period_end"] == r2["period_end"]][0]
    assert r2["knowledge_ts"] > r1["knowledge_ts"]  # restatement lands later
    assert r2["value"] != r1["value"]


def test_asof_serves_latest_known_only():
    rows = build_revision_log(PE, VQ, "sales", rng, restate_p=0.0)
    vals, kn = pit_asof(rows, DAYS)
    assert vals.shape == (T, N)
    validate_pit(vals, kn, DAYS)
    # Before any knowledge exists: NaN (no forward-fill from the future).
    first_k = min(r["knowledge_ts"] for r in rows)
    ti = int(np.searchsorted(DAYS.astype(str), first_k)) - 1
    if ti >= 0:
        assert bool((~np.isfinite(vals[ti])).all())


def test_restatement_not_visible_early():
    rows = build_revision_log(PE, VQ, "sales", rng, restate_p=1.0)
    vals, kn = pit_asof(rows, DAYS)
    validate_pit(vals, kn, DAYS)
    # Pick a restated cell: before its rev_seq=2 knowledge date, the served
    # value must equal the rev_seq=1 value, never the restated one.
    r2 = [r for r in rows if r["rev_seq"] == 2][0]
    r1 = [r for r in rows if r["rev_seq"] == 1 and r["sid"] == r2["sid"]
          and r["period_end"] == r2["period_end"]][0]
    k2 = np.datetime64(r2["knowledge_ts"])
    ti = int(np.searchsorted(DAYS, k2)) - 1
    assert ti >= 0
    assert vals[ti, r2["sid"]] == r1["value"]
    assert vals[ti, r2["sid"]] != r2["value"]


def test_report_lag_withholds_fresh_periods():
    rows = build_revision_log(PE, VQ, "sales", rng, restate_p=0.0)
    vals, _ = pit_asof(rows, DAYS, report_lag_days=365)
    # 365d lag: Q0 (Jan) reportable only very late; early days all NaN.
    assert bool((~np.isfinite(vals[:10])).all())

def test_multi_matches_single():
    from pit import pit_asof_multi
    r1 = build_revision_log(PE, VQ, "sales", rng)
    r2 = build_revision_log(PE, VQ + 5.0, "assets", rng)
    out = pit_asof_multi({"sales": r1, "assets": r2}, DAYS)
    v1, _ = pit_asof(r1, DAYS)
    v2, _ = pit_asof(r2, DAYS)
    assert (np.nan_to_num(out["sales"][0]) == np.nan_to_num(v1)).all()
    assert (np.nan_to_num(out["assets"][0]) == np.nan_to_num(v2)).all()
    validate_pit(out["sales"][0], out["sales"][1], DAYS)
    validate_pit(out["assets"][0], out["assets"][1], DAYS)
