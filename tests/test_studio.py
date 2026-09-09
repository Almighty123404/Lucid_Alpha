"""Web Studio (Phase 1-5) tests: Python SDK + FastAPI server.

These tests never spawn a real 40s simulation:
- SDK end-to-end reuses the small session `panel` fixture and writes to a temp
  artifact root, so the full engine + artifact path is exercised fast.
- Server job orchestration is tested with a monkeypatched `_run_cli` fake that
  drives status transitions and log publishing without a subprocess.
"""
import asyncio
import time

import pytest

import fastexp
import fastexp.sdk as sdk
from fastexp.server import TERMINAL, JOBS, _parse_batch_text, app

# Third-party deprecation notice from starlette's test client (httpx -> httpx2);
# unrelated to Fastexp and noisy, so it is filtered for a clean warning count.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient` is deprecated")


# --------------------------------------------------------------------------
# SDK
# --------------------------------------------------------------------------
def test_lazy_exports():
    assert fastexp.simulate is sdk.simulate
    assert fastexp.batch is sdk.batch
    assert fastexp.Run is sdk.Run


def test_settings_dict():
    assert sdk._settings_dict("TOP3000", None, None, None) == {"universe": "TOP3000"}
    assert sdk._settings_dict(None, None, None, None) is None
    assert sdk._settings_dict(None, "MARKET", 0.05, 4) == \
        {"neutralization": "MARKET", "truncation": 0.05, "decay": 4}


@pytest.fixture(scope="module")
def rep(panel):
    from simulator import simulate
    return simulate("-ts_zscore(returns, 21)", panel, (), ())


@pytest.fixture(scope="module")
def run_dir(rep, tmp_path_factory):
    from reporting import build_run_artifact
    d, _ = build_run_artifact(rep, "rank(close)", seed=11, mode="fast_gate",
                              out_root=str(tmp_path_factory.mktemp("studio")))
    return d


def test_run_accessors(run_dir):
    run = sdk.Run(run_dir)
    assert run.run_id and run.status in ("PASS", "FAIL", "ERROR", "BLOCKED")
    assert isinstance(run.metrics, dict) and "sharpe" in run.metrics
    assert isinstance(run.criteria, dict)
    eq = run.equity_curve
    if sdk.HAS_PANDAS:
        assert "date" in list(eq.columns)
    else:
        assert isinstance(eq, list) and eq and "date" in eq[0]
    assert run.drawdown is not None
    assert run.exposures is not None
    assert repr(run).startswith("<Run ")


def test_sdk_simulate_end_to_end(panel, monkeypatch, tmp_path):
    monkeypatch.setattr(sdk, "_build_panel", lambda dataset, seed: panel)
    monkeypatch.setattr(sdk, "_DEFAULT_OUT", tmp_path)
    run = sdk.simulate("rank(close)", dataset="synthetic", mode="fast_gate",
                       seed=11, truncation=0.05)
    assert run.status in ("PASS", "FAIL", "ERROR", "BLOCKED")
    assert (tmp_path / run.run_id / "manifest.json").exists()
    assert isinstance(run.metrics, dict)


def test_show_methods_fallback(run_dir, monkeypatch):
    run = sdk.Run(run_dir)
    opened = []
    monkeypatch.setattr(sdk.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(run, "_ipython", lambda: None)
    run.show_report()
    run.show_dashboard()
    assert len(opened) == 2
    assert opened[0].endswith("report.html")
    assert opened[1].endswith("dashboard.html")


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


def test_server_metadata_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200 and "Web Studio" in r.text
    r = client.get("/api/v1/operators")
    assert r.status_code == 200 and r.json()["operators"] and r.json()["fields"]
    r = client.get("/api/v1/templates")
    assert r.status_code == 200 and r.json()["templates"]
    r = client.get("/api/v1/runs")
    assert r.status_code == 200 and "runs" in r.json()
    r = client.get("/api/v1/jobs")
    assert r.status_code == 200 and "jobs" in r.json()


def test_server_validation(client):
    assert client.post("/api/v1/simulate", json={"expression": "rank(close)", "dataset": "nope"}).status_code == 422
    assert client.post("/api/v1/simulate", json={"expression": "rank(close)", "mode": "bogus"}).status_code == 422
    assert client.post("/api/v1/simulate", json={"expression": "rank(close)", "truncation": 2.0}).status_code == 422
    assert client.post("/api/v1/simulate", json={"expression": ""}).status_code == 422


def test_parse_batch_text():
    assert _parse_batch_text("a\n\nb\n", "") == ["a", "b"]
    assert _parse_batch_text('{"expression": "x"}\n{"expr": "y"}\nbad\n', "f.jsonl") == ["x", "y"]
    assert _parse_batch_text("", "") == []


def test_job_lifecycle(monkeypatch, client):
    async def fake_run_cli(job, args):
        JOBS.publish(job, "SYSTEM", "fake start")
        await asyncio.sleep(0.02)
        job["status"] = "PASS"
        job["run_ids"] = ["fake_run_1"]
        JOBS.publish(job, "SYSTEM", "finished rc=0 status=PASS runs=['fake_run_1']")

    monkeypatch.setattr("fastexp.server._run_cli", fake_run_cli)
    r = client.post("/api/v1/simulate", json={"expression": "rank(close)"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    status = None
    for _ in range(100):
        st = client.get(f"/api/v1/jobs/{job_id}").json()
        if st["status"] in TERMINAL:
            status = st
            break
        time.sleep(0.02)
    assert status and status["status"] == "PASS" and status["run_ids"] == ["fake_run_1"]

    jobs = client.get("/api/v1/jobs").json()["jobs"]
    assert any(j["job_id"] == job_id for j in jobs)


def test_ws_streaming(monkeypatch, client):
    async def fake_run_cli(job, args):
        JOBS.publish(job, "INFO", "hello")
        await asyncio.sleep(0.05)
        job["status"] = "PASS"
        job["run_ids"] = ["fake_run_1"]
        JOBS.publish(job, "SYSTEM", "done")

    monkeypatch.setattr("fastexp.server._run_cli", fake_run_cli)
    r = client.post("/api/v1/simulate", json={"expression": "rank(close)"})
    job_id = r.json()["job_id"]

    with client.websocket_connect(f"/api/v1/ws/logs/{job_id}") as ws:
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m.get("done"):
                break
    assert any(m.get("level") == "INFO" and m["text"] == "hello" for m in msgs)
    assert msgs[-1]["done"] is True and msgs[-1]["run_ids"] == ["fake_run_1"]
