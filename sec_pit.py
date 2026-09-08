"""Offline-first SEC Company Facts PIT adapter.

This module is deliberately separate from Panel generation. It ingests raw SEC
Company Facts/submissions JSON into append-only DuckDB tables, preserving
accessions, amendments, units, periods, and timestamp quality. Network access
is opt-in through ``SecHttpClient`` and requires an explicit SEC User-Agent.
"""
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone

try:
    import duckdb
except ImportError:  # pragma: no cover - dependency is required by PIT runtime
    duckdb = None

SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
CANONICAL = {
    "revenue": ("Revenues", "SalesRevenueNet"),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "cashflow_op": ("NetCashProvidedByUsedInOperatingActivities",),
    "net_income": ("NetIncomeLoss",),
}


def _sha(payload):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


class SecHttpClient:
    """Small rate-limited SEC transport; no credentials or tokens accepted."""
    def __init__(self, user_agent, min_interval=0.12, opener=None):
        if not isinstance(user_agent, str) or not user_agent.strip() or "@" not in user_agent:
            raise ValueError("SEC User-Agent must identify an application and contact email")
        self.user_agent = user_agent
        self.min_interval = float(min_interval)
        if self.min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        self.opener = opener or urllib.request.urlopen
        self._last = 0.0

    def get_json(self, url):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip"})
        with self.opener(req, timeout=30) as response:
            body = response.read()
        self._last = time.monotonic()
        return json.loads(body)

    def company_payload(self, cik):
        return self.get_json(SEC_FACTS.format(cik=int(cik)))

    def submissions_payload(self, cik):
        return self.get_json(SEC_SUBMISSIONS.format(cik=int(cik)))


def create_schema(con):
    con.execute("""CREATE TABLE IF NOT EXISTS sec_entity(
        cik BIGINT PRIMARY KEY, entity_name VARCHAR, source_sha256 VARCHAR,
        retrieved_at TIMESTAMPTZ NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS sec_filing(
        cik BIGINT, accession VARCHAR, form VARCHAR, report_date DATE,
        filed_date DATE, accepted_ts TIMESTAMPTZ, amendment BOOLEAN,
        source_sha256 VARCHAR, retrieved_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY(cik, accession))""")
    con.execute("""CREATE TABLE IF NOT EXISTS sec_fact_revision(
        fact_id VARCHAR PRIMARY KEY, cik BIGINT, accession VARCHAR,
        taxonomy VARCHAR, concept VARCHAR, unit VARCHAR, value DOUBLE,
        period_type VARCHAR, period_start DATE, period_end DATE,
        frame VARCHAR, filed_date DATE, accepted_ts TIMESTAMPTZ,
        knowledge_ts TIMESTAMPTZ, timestamp_quality VARCHAR,
        source_sha256 VARCHAR, parser_version VARCHAR)""")


def _accepted_map(submissions):
    out = {}
    recent = submissions.get("filings", {}).get("recent", {}) if submissions else {}
    fields = list(recent.get("accessionNumber", []))
    for i, acc in enumerate(fields):
        accepted = recent.get("acceptanceDateTime", [None] * len(fields))[i]
        out[acc] = accepted
    return out


def facts_rows(cik, facts_payload, submissions=None, parser_version="secfacts-v1"):
    """Return canonical long-form PIT rows; missing acceptance times stay flagged."""
    accepted = _accepted_map(submissions or {})
    source = _sha(facts_payload)
    facts = facts_payload.get("facts", {})
    rows = []
    for taxonomy, concepts in facts.items():
        for canonical, names in CANONICAL.items():
            for concept in names:
                units = concepts.get(concept, {}).get("units", {})
                for unit, values in units.items():
                    for ordinal, fact in enumerate(values):
                        filed = fact.get("filed")
                        end = fact.get("end")
                        if not filed or not end or "val" not in fact:
                            continue
                        acc = fact.get("accn", "")
                        accepted_raw = accepted.get(acc)
                        accepted_ts = None
                        if accepted_raw:
                            accepted_ts = datetime.fromisoformat(accepted_raw.replace("Z", "+00:00"))
                        knowledge = accepted_ts or datetime.fromisoformat(filed + "T00:00:00+00:00")
                        period_type = "duration" if fact.get("start") else "instant"
                        fact_id = hashlib.sha256(json.dumps(
                            [cik, acc, taxonomy, concept, unit, fact.get("start"), end,
                             fact.get("val"), ordinal], sort_keys=True, default=str).encode()).hexdigest()
                        rows.append({"fact_id": fact_id, "cik": int(cik), "accession": acc,
                                     "taxonomy": taxonomy, "concept": canonical, "unit": unit,
                                     "value": float(fact["val"]), "period_type": period_type,
                                     "period_start": fact.get("start"), "period_end": end,
                                     "frame": fact.get("frame"), "filed_date": filed,
                                     "accepted_ts": accepted_ts, "knowledge_ts": knowledge,
                                     "timestamp_quality": "accepted_ts" if accepted_ts else "filed_date_only",
                                     "source_sha256": source, "parser_version": parser_version})
    return rows


def ingest_companyfacts(con, cik, facts_payload, submissions=None, parser_version="secfacts-v1"):
    if con is None:
        raise RuntimeError("duckdb is required for SEC PIT ingestion")
    create_schema(con)
    now = datetime.now(timezone.utc)
    source = _sha(facts_payload)
    con.execute("INSERT OR REPLACE INTO sec_entity VALUES (?, ?, ?, ?)",
                [int(cik), facts_payload.get("entityName", ""), source, now])
    rows = facts_rows(cik, facts_payload, submissions, parser_version)
    for r in rows:
        con.execute("""INSERT OR REPLACE INTO sec_fact_revision VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", [
            r[k] for k in ("fact_id", "cik", "accession", "taxonomy", "concept", "unit", "value",
                           "period_type", "period_start", "period_end", "frame", "filed_date",
                           "accepted_ts", "knowledge_ts", "timestamp_quality", "source_sha256", "parser_version")])
    return len(rows)


def asof_facts(con, cik, decision_ts, strict=True):
    """Latest known canonical facts at a timestamp; strict excludes filed-only rows."""
    quality = "AND timestamp_quality = 'accepted_ts'" if strict else ""
    return con.execute(f"""SELECT concept, period_end, value, knowledge_ts, accession
        FROM sec_fact_revision WHERE cik = ? AND knowledge_ts <= ? {quality}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY concept, period_end
          ORDER BY knowledge_ts DESC, fact_id DESC) = 1
        ORDER BY concept, period_end""", [int(cik), decision_ts]).fetchall()
