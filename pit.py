"""Phase 4 PIT data engine: bi-temporal revision logs + DuckDB lookup.

Invariant: any read at decision date T may only see revisions with
knowledge_ts <= T (and period_end <= T - report_lag). Revisions are
append-only; restatements are new rows (rev_seq=2), never overwrites.
This is the Qlib PIT pattern (publication-date indexing) implemented as a
DuckDB sid-equality JOIN + latest-qualifying match (ASOF JOIN syntax allows
only one inequality; the two-predicate match used here has identical
semantics — see pit_asof).
"""
import numpy as np

try:
    import duckdb
    DUCKDB_OK = True
except ImportError:
    DUCKDB_OK = False


def build_revision_log(period_ends, values_q, field, rng,
                       lag_lo=20, lag_hi=45, restate_p=0.02):
    """Synthetic revision log for one quarterly field.

    period_ends: (Q,) datetime64[D]; values_q: (Q, N) true values.
    knowledge_ts = period_end + U[lag_lo, lag_hi] calendar days.
    ~restate_p of cells get a restatement row (rev_seq=2, knowledge +30..90d,
    value x (1+N(0,0.02))). Returns list of row dicts.
    """
    Q, N = values_q.shape
    rows = []
    for q in range(Q):
        pe = np.datetime64(period_ends[q])
        for j in range(N):
            v = float(values_q[q, j])
            if v != v:
                continue
            kts = pe + np.timedelta64(int(rng.integers(lag_lo, lag_hi + 1)), 'D')
            rows.append({"sid": int(j), "field": field,
                         "period_end": str(pe), "knowledge_ts": str(kts),
                         "value": v, "rev_seq": 1})
            if rng.random() < restate_p:
                k2 = kts + np.timedelta64(int(rng.integers(30, 91)), 'D')
                rows.append({"sid": int(j), "field": field,
                             "period_end": str(pe), "knowledge_ts": str(k2),
                             "value": v * float(1.0 + rng.normal(0.0, 0.02)),
                             "rev_seq": 2})
    return rows


def pit_asof(rows, decision_dates, report_lag_days=0):
    """Serve leak-free panels via DuckDB (single query per field).

    Returns (values (T,N), knowledge (T,N) datetime64): per (decision, sid)
    the latest revision with knowledge_ts <= decision AND period_end <=
    decision - report_lag_days; NaN where none qualifies. The period lag is
    folded into an EFFECTIVE knowledge timestamp
    (usable = knowledge<=dts AND period_end<=dts-lag IFF
    max(knowledge, period_end+lag)<=dts), reducing the match to ONE
    inequality that DuckDB executes as a merge ASOF JOIN — exact, not an
    approximation (a two-inequality JOIN degrades to nested-loop, ~50s per
    783k-cell field — profiled 2026-09-06). With lag=0 this is knowledge_ts.
    The knowledge grid returned is the TRUE publication timestamp (for the
    leakage validator), not the effective one.
    """
    if not DUCKDB_OK:
        raise RuntimeError("duckdb is required for pit_asof (pip install duckdb)")
    decisions = np.asarray(decision_dates).astype('datetime64[D]')
    T = len(decisions)
    sids = sorted({r["sid"] for r in rows})
    N = (max(sids) + 1) if sids else 0
    lag = np.timedelta64(int(report_lag_days), 'D')
    con = duckdb.connect()
    con.execute("CREATE TABLE rev(sid INTEGER, knowledge_ts DATE, "
                "value DOUBLE, rev_seq INTEGER, pub_ts DATE)")
    # Bulk columnar insert (profiled 2026-09-06): executemany row-at-a-time
    # costs ~1.4s per 1.2k rows (~15s per full field); UNNEST column params
    # transfer the same data in milliseconds.
    _sid = np.array([r["sid"] for r in rows], dtype=np.int32)
    _kn = np.empty(len(rows), dtype="datetime64[D]")
    _pub = np.empty(len(rows), dtype="datetime64[D]")
    _val = np.empty(len(rows), dtype=np.float64)
    _seq = np.empty(len(rows), dtype=np.int32)
    for i, r in enumerate(rows):
        pe = np.datetime64(r["period_end"])
        k = np.datetime64(r["knowledge_ts"])
        _kn[i] = k if k >= pe + lag else pe + lag
        _pub[i] = k
        _val[i] = r["value"]
        _seq[i] = r["rev_seq"]
    con.execute("INSERT INTO rev SELECT * FROM "
                "(SELECT UNNEST(CAST(? AS INTEGER[])) AS sid, "
                "UNNEST(CAST(? AS DATE[])) AS knowledge_ts, "
                "UNNEST(CAST(? AS DOUBLE[])) AS value, "
                "UNNEST(CAST(? AS INTEGER[])) AS rev_seq, "
                "UNNEST(CAST(? AS DATE[])) AS pub_ts)",
                [_sid.tolist(), _kn.tolist(), _val.tolist(),
                 _seq.tolist(), _pub.tolist()])
    # Perf note (profiled 2026-09-06): executemany-inserting the TxN decision
    # grid row-by-row costs ~40s per 78k cells; passing the grid as UNNEST
    # parameters keeps the whole lookup near one second. Same-effective-day
    # restatement collisions are measure-zero (offsets span 30-90d), so no
    # secondary tiebreak is needed beyond the ASOF match.
    dstr = [str(d) for d in decisions]
    # Perf (profiled 2026-09-06): the ASOF JOIN must run over MATERIALIZED,
    # pre-sorted inputs — against CTE/UNNEST legs the planner falls back to
    # nested-loop (~30s per 783k cells); temp tables take the merge path
    # (<1s). Non-matching decision cells are absent from ASOF output and
    # stay NaN (correct: nothing knowable yet).
    con.execute("CREATE TEMP TABLE dec AS WITH dd(dts) AS (SELECT UNNEST(CAST(? AS DATE[]))), "
                "ss(sid) AS (SELECT UNNEST(CAST(? AS INTEGER[]))) SELECT * FROM dd CROSS JOIN ss",
                [dstr, sids])
    con.execute("CREATE TEMP TABLE revs AS SELECT * FROM rev ORDER BY sid, knowledge_ts")
    got = con.execute(
        "SELECT d.dts, d.sid, r.value, r.pub_ts FROM dec d "
        "ASOF JOIN revs r ON (r.sid = d.sid AND r.knowledge_ts <= d.dts) "
        "ORDER BY d.dts, d.sid").fetchall()
    con.close()
    return _assemble(got, decisions, T, N)


def pit_asof_multi(field_rows, decision_dates):
    """Serve MULTIPLE fields in one DuckDB query (one shared decision grid).

    field_rows: {field: rows}. Returns {field: (values, knowledge)}. Same
    per-field semantics as pit_asof with report_lag_days=0 (production use:
    publication lag already encoded in knowledge_ts, so effective ==
    publication and no folding is needed). ~3x faster than one query per
    field because the TxN decision grid is built and sorted once. Added after
    profiling showed 4 separate queries dominating panel builds.
    """
    if not DUCKDB_OK:
        raise RuntimeError("duckdb is required for pit_asof_multi (pip install duckdb)")
    decisions = np.asarray(decision_dates).astype('datetime64[D]')
    T = len(decisions)
    sids = sorted({r["sid"] for rows in field_rows.values() for r in rows})
    N = (max(sids) + 1) if sids else 0
    dstr = [str(d) for d in decisions]
    con = duckdb.connect()
    parts, params = [], [dstr, sids]
    for i, (field, rows) in enumerate(field_rows.items()):
        con.execute("CREATE TABLE rev%d(sid INTEGER, knowledge_ts DATE, "
                    "value DOUBLE, rev_seq INTEGER, pub_ts DATE)" % i)
        _sid = np.array([r["sid"] for r in rows], dtype=np.int32)
        _kn = np.empty(len(rows), dtype="datetime64[D]")
        _pub = np.empty(len(rows), dtype="datetime64[D]")
        _val = np.empty(len(rows), dtype=np.float64)
        _seq = np.empty(len(rows), dtype=np.int32)
        for j, r in enumerate(rows):
            pe = np.datetime64(r["period_end"])
            k = np.datetime64(r["knowledge_ts"])
            _kn[j] = k  # effective knowledge precomputed by caller contract
            _pub[j] = k
            _val[j] = r["value"]
            _seq[j] = r["rev_seq"]
        con.execute("INSERT INTO rev%d SELECT * FROM "
                    "(SELECT UNNEST(CAST(? AS INTEGER[])) AS sid, "
                    "UNNEST(CAST(? AS DATE[])) AS knowledge_ts, "
                    "UNNEST(CAST(? AS DOUBLE[])) AS value, "
                    "UNNEST(CAST(? AS INTEGER[])) AS rev_seq, "
                    "UNNEST(CAST(? AS DATE[])) AS pub_ts)" % i,
                    [_sid.tolist(), _kn.tolist(), _val.tolist(),
                     _seq.tolist(), _pub.tolist()])
        con.execute("CREATE TEMP TABLE revs%d AS SELECT * FROM rev%d ORDER BY sid, knowledge_ts" % (i, i))
    dstr = [str(d) for d in decisions]
    con.execute("CREATE TEMP TABLE dec AS WITH dd(dts) AS (SELECT UNNEST(CAST(? AS DATE[]))), "
                "ss(sid) AS (SELECT UNNEST(CAST(? AS INTEGER[]))) SELECT * FROM dd CROSS JOIN ss",
                [dstr, sorted({r["sid"] for rows in field_rows.values() for r in rows})])
    out = {}
    names = list(field_rows.keys())
    for i, field in enumerate(names):
        got = con.execute(
            "SELECT d.dts, d.sid, r.value, r.pub_ts FROM dec d "
            "ASOF JOIN revs%d r ON (r.sid = d.sid AND r.knowledge_ts <= d.dts) "
            "ORDER BY d.dts, d.sid" % i).fetchall()
        out[field] = _assemble(got, decisions, T, N)
    con.close()
    return out


def _assemble(got, decisions, T, N):
    """Vectorized (tgrid, sid) assembly shared by pit_asof."""
    darr = np.array([d for d, _, _, _ in got])
    sarr = np.array([s for _, s, _, _ in got], dtype=np.int64)
    varr = np.array([np.nan if v is None else v for _, _, v, _ in got])
    tgrid = np.searchsorted(decisions, darr.astype("datetime64[D]"))
    vals = np.full((T, N), np.nan)
    kn = np.full((T, N), np.datetime64("NaT", "D"))
    has = ~np.isnan(varr)
    vals[tgrid[has], sarr[has]] = varr[has]
    karr = np.array([str(k) if k is not None else "NaT" for _, _, _, k in got])
    kn[tgrid[has], sarr[has]] = karr[has].astype("datetime64[D]")
    return vals, kn


def validate_pit(values, knowledge, decisions):
    """Assert the leakage invariant on a served panel.

    Every served (finite) cell must satisfy knowledge_ts <= decision_ts
    (the period_end lag is enforced inside pit_asof's JOIN). Raises
    ValueError naming the first violation. Returns True.
    """
    decisions = np.asarray(decisions).astype('datetime64[D]')
    finite = np.isfinite(values)
    bad = finite & ~(knowledge <= decisions[:, None])
    if bad.any():
        t, s = np.unravel_index(int(np.argmax(bad)), bad.shape)
        raise ValueError(f"PIT violation at day {t} sid {s}: knowledge "
                         f"{knowledge[t, s]} > decision {decisions[t]}")
    return True
