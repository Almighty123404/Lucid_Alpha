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


def test_revision_log_v2_gradients_and_bounds():
    from pit import build_revision_log_v2
    r = __import__('numpy').random.default_rng(3)
    N, Q = 60, 8
    pe = __import__('numpy').array(['2020-01-01','2020-04-01','2020-07-01','2020-10-01','2021-01-01','2021-04-01','2021-07-01','2021-10-01'], dtype='datetime64[D]')
    vq = __import__('numpy').full((Q, N), 100.0)
    liq = (__import__('numpy').arange(N) % 10) / 9.0
    rows = build_revision_log_v2(pe, vq, 'sales', r, liq_rank=liq)
    lag = {(x['sid'], x['period_end']): (__import__('numpy').datetime64(x['knowledge_ts']) - __import__('numpy').datetime64(x['period_end'])).astype(int) for x in rows if x['rev_seq'] == 1}
    import numpy as _np
    top = _np.mean([v for (s, _), v in lag.items() if liq[s] > 0.8])
    bot = _np.mean([v for (s, _), v in lag.items() if liq[s] < 0.2])
    assert top < bot - 3.0, (top, bot)
    cells = {(x['sid'], x['period_end']) for x in rows}
    r2 = [x for x in rows if x['rev_seq'] == 2]
    rate = len({(x['sid'], x['period_end']) for x in r2}) / len(cells)
    assert 0.01 <= rate <= 0.03, rate
    for x in r2:
        d = (np.datetime64(x['knowledge_ts']) - np.datetime64(x['period_end'])).astype(int)
        assert d <= 45 + 10 + 120, d


def test_revision_log_v2_backcompat_uniform():
    from pit import build_revision_log_v2
    import numpy as _np
    r = _np.random.default_rng(5)
    pe = _np.array(['2020-01-01','2020-04-01'], dtype='datetime64[D]')
    rows = build_revision_log_v2(pe, _np.full((2, 20), 50.0), 'x', r, size_graded_lag=False, q4_bump=False)
    lags = [(np.datetime64(x['knowledge_ts']) - np.datetime64(x['period_end'])).astype(int) for x in rows if x['rev_seq'] == 1]
    assert min(lags) >= 20 and max(lags) <= 45


def test_pit_rejects_unsorted_decisions_and_prefers_period_then_revision():
    from pit import pit_asof
    import pytest
    with pytest.raises(ValueError, match="sorted"):
        pit_asof([], np.array(['2020-02-02', '2020-02-01'], dtype='datetime64[D]'))
    rows = [
        {"sid": 0, "field": "x", "period_end": "2020-01-01", "knowledge_ts": "2020-01-10", "value": 10., "rev_seq": 1},
        {"sid": 0, "field": "x", "period_end": "2020-02-01", "knowledge_ts": "2020-02-10", "value": 20., "rev_seq": 1},
        {"sid": 0, "field": "x", "period_end": "2020-01-01", "knowledge_ts": "2020-03-01", "value": 11., "rev_seq": 2},
        {"sid": 0, "field": "x", "period_end": "2020-02-01", "knowledge_ts": "2020-03-02", "value": 21., "rev_seq": 2},
    ]
    vals, _ = pit_asof(rows, np.array(['2020-03-03'], dtype='datetime64[D]'))
    assert vals[0, 0] == 21.0



def test_pit_connection_closed_on_query_failure():
    import pit as _pit
    closed = []
    real_connect = _pit.duckdb.connect
    class _Boom(Exception):
        pass
    class _FakeCon:
        def execute(self, *a, **k):
            raise _Boom()
        def close(self):
            closed.append(1)
    _pit.duckdb.connect = lambda: _FakeCon()
    import numpy as _np
    rows = [{'sid': 0, 'field': 'x', 'period_end': '2020-01-01', 'knowledge_ts': '2020-02-01', 'value': 1.0, 'rev_seq': 1}]
    days = _np.array(['2020-03-01'], dtype='datetime64[D]')
    try:
        import pytest as _pt
        with _pt.raises(_Boom):
            _pit.pit_asof(rows, days)
        with _pt.raises(_Boom):
            _pit.pit_asof_multi({'x': rows}, days)
        assert closed == [1, 1]  # both paths release the connection
    finally:
        _pit.duckdb.connect = real_connect

