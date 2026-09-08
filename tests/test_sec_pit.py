import json
from datetime import datetime, timezone

import pytest

from sec_pit import SecHttpClient, facts_rows, create_schema, ingest_companyfacts, asof_facts


def _payload():
    return {"entityName": "Fixture Co", "facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [
            {"accn": "0001", "start": "2020-01-01", "end": "2020-03-31", "filed": "2020-05-01", "val": 10},
            {"accn": "0002", "start": "2020-01-01", "end": "2020-03-31", "filed": "2020-06-01", "val": 11},
        ]}},
        "Assets": {"units": {"USD": [
            {"accn": "0001", "end": "2020-03-31", "filed": "2020-05-01", "val": 100},
        ]}},
    }}}


def test_sec_facts_strict_pit_and_amendment():
    import duckdb
    con = duckdb.connect()
    sub = {"filings": {"recent": {"accessionNumber": ["0001", "0002"],
                                      "acceptanceDateTime": ["2020-05-01T20:00:00Z", "2020-06-01T20:00:00Z"]}}}
    assert ingest_companyfacts(con, 123, _payload(), sub) == 3
    early = asof_facts(con, 123, datetime(2020, 5, 15, tzinfo=timezone.utc))
    late = asof_facts(con, 123, datetime(2020, 6, 15, tzinfo=timezone.utc))
    assert any(r[0] == 'net_income' and r[2] == 10.0 for r in early)
    assert any(r[0] == 'net_income' and r[2] == 11.0 for r in late)
    con.close()


def test_sec_client_requires_contact_and_uses_user_agent():
    with pytest.raises(ValueError):
        SecHttpClient('no-contact')
    seen = []
    class Resp:
        def read(self): return b'{"ok": true}'
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def opener(req, timeout=30):
        seen.append(req.headers['User-agent'])
        return Resp()
    c = SecHttpClient('Fastexp audit audit@example.com', opener=opener, min_interval=0)
    assert c.get_json('https://example.test')['ok'] is True
    assert seen == ['Fastexp audit audit@example.com']
