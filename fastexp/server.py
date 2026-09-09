"""Fastexp Web Studio — local-first FastAPI service over the artifact engine.

Non-invasive by design:
- It never imports or alters the simulation math. It shells out to the same
  CLI (`python -m fastexp.cli ...`) a user would run by hand, so every run
  produces the exact same immutable `reports/runs/<run_id>/` directory.
- Long-running simulations are spawned as asyncio subprocesses; their
  stdout/stderr is streamed to the browser over a WebSocket.

Run it:

    python -m fastexp.server            # http://127.0.0.1:8000
    python -m fastexp.server --port 9000

Endpoints (Phase 1):
    GET  /                              -> studio UI
    POST /api/v1/simulate               -> start a single-expression run
    POST /api/v1/batch                  -> start a batch (list of expressions)
    GET  /api/v1/runs                   -> list artifact runs
    GET  /api/v1/jobs/{job_id}          -> poll a job (status + run ids)
    GET  /api/v1/ws/logs/{job_id}       -> WebSocket live log stream
    POST /api/v1/runs/stop              -> terminate a running job
    GET  /api/v1/operators              -> operator registry (autocomplete/docs)
    GET  /api/v1/templates              -> preset alpha templates
    /static/* and /reports/*            -> UI assets and generated artifacts
"""
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "reports" / "runs"
WEB_DIR = Path(__file__).resolve().parent / "web" / "static"

TERMINAL = {"PASS", "FAIL", "BLOCKED", "ERROR", "STOPPED"}
DATASETS = ["synthetic", "real"]
MODES = ["fast_gate", "high_fidelity"]
UNIVERSES = ["TOP3000", "TOP1000", "TOP500", "TOP200"]
NEUTRALIZATIONS = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "COUNTRY", "EXCHANGE"]


# --------------------------------------------------------------------------
# Operator / template registry (served to the editor for autocomplete + docs)
# --------------------------------------------------------------------------
OPERATORS = [
    # cross-sectional
    {"name": "rank", "sig": "rank(x)", "category": "Cross-sectional",
     "doc": "Cross-sectional rank scaled to [0,1]."},
    {"name": "zscore", "sig": "zscore(x)", "category": "Cross-sectional",
     "doc": "Cross-sectional z-score (mean 0, std 1)."},
    {"name": "scale", "sig": "scale(x, scale=1, longscale=1, shortscale=1)", "category": "Cross-sectional",
     "doc": "Scale so sum of absolute weights equals the target."},
    {"name": "quantile", "sig": "quantile(x, driver=1)", "category": "Cross-sectional",
     "doc": "Gaussianized rank; driver 0=uniform, 1=gaussian, 2=cauchy."},
    {"name": "bucket", "sig": "bucket(x, nbuckets=10)", "category": "Cross-sectional",
     "doc": "Bucket membership ids (or group ids via bucket(x, low, high, step))."},
    {"name": "signed_power", "sig": "signed_power(x, a)", "category": "Cross-sectional",
     "doc": "sign(x) * |x|^a."},
    {"name": "group_neutralize", "sig": "group_neutralize(x, g)", "category": "Group",
     "doc": "Demean x within group g (industry/sector/subindustry/...)."},
    {"name": "group_rank", "sig": "group_rank(x, g)", "category": "Group",
     "doc": "Rank within group g."},
    {"name": "group_zscore", "sig": "group_zscore(x, g)", "category": "Group",
     "doc": "Z-score within group g."},
    {"name": "group_scale", "sig": "group_scale(x, g)", "category": "Group",
     "doc": "Scale within group g to sum|.|=1."},
    {"name": "group_mean", "sig": "group_mean(x, g, w)", "category": "Group",
     "doc": "Rolling group mean."},
    {"name": "vector_neut", "sig": "vector_neut(x, g)", "category": "Vector",
     "doc": "Neutralize a vector across groups."},
    {"name": "group_vector_neut", "sig": "group_vector_neut(x, g)", "category": "Vector",
     "doc": "Neutralize a vector within each group-day."},
    {"name": "vec_avg", "sig": "vec_avg(x, d)", "category": "Vector",
     "doc": "Average over vector dimension d."},
    {"name": "vec_sum", "sig": "vec_sum(x, d)", "category": "Vector",
     "doc": "Sum over vector dimension d."},
    {"name": "vec_max", "sig": "vec_max(x, d)", "category": "Vector",
     "doc": "Max over vector dimension d."},
    {"name": "vec_min", "sig": "vec_min(x, d)", "category": "Vector",
     "doc": "Min over vector dimension d."},
    # time-series
    {"name": "ts_mean", "sig": "ts_mean(x, w)", "category": "Time-series",
     "doc": "Trailing mean over window w."},
    {"name": "ts_std_dev", "sig": "ts_std_dev(x, w)", "category": "Time-series",
     "doc": "Trailing standard deviation over window w."},
    {"name": "ts_zscore", "sig": "ts_zscore(x, w)", "category": "Time-series",
     "doc": "Trailing z-score over window w."},
    {"name": "ts_decay_linear", "sig": "ts_decay_linear(x, w)", "category": "Time-series",
     "doc": "Linearly decaying weighted mean over window w."},
    {"name": "ts_delta", "sig": "ts_delta(x, w)", "category": "Time-series",
     "doc": "Difference vs value w days ago."},
    {"name": "ts_backfill", "sig": "ts_backfill(x, w)", "category": "Time-series",
     "doc": "Forward-fill the most recent finite value within w days."},
    {"name": "ts_rank", "sig": "ts_rank(x, w)", "category": "Time-series",
     "doc": "Rank of the latest value within the trailing window w."},
    {"name": "ts_sum", "sig": "ts_sum(x, w)", "category": "Time-series",
     "doc": "Trailing sum over window w."},
    {"name": "ts_product", "sig": "ts_product(x, w)", "category": "Time-series",
     "doc": "Trailing product over window w."},
    {"name": "ts_count_nans", "sig": "ts_count_nans(x, w)", "category": "Time-series",
     "doc": "Count of NaN values in the trailing window w."},
    {"name": "ts_av_diff", "sig": "ts_av_diff(x, w)", "category": "Time-series",
     "doc": "Average of lagged differences over window w."},
    {"name": "ts_scale", "sig": "ts_scale(x, w)", "category": "Time-series",
     "doc": "Rescale by trailing sum of |x| over window w."},
    {"name": "ts_arg_max", "sig": "ts_arg_max(x, w)", "category": "Time-series",
     "doc": "Days since the trailing-window maximum."},
    {"name": "ts_arg_min", "sig": "ts_arg_min(x, w)", "category": "Time-series",
     "doc": "Days since the trailing-window minimum."},
    {"name": "ts_delay", "sig": "ts_delay(x, w)", "category": "Time-series",
     "doc": "Value w days earlier."},
    {"name": "ts_corr", "sig": "ts_corr(x, y, w)", "category": "Time-series",
     "doc": "Trailing correlation of x and y over window w."},
    {"name": "ts_covariance", "sig": "ts_covariance(x, y, w)", "category": "Time-series",
     "doc": "Trailing covariance of x and y over window w."},
    {"name": "ts_regression", "sig": "ts_regression(x, y, w)", "category": "Time-series",
     "doc": "Trailing regression residual of x on y over window w."},
    {"name": "ts_quantile", "sig": "ts_quantile(x, w)", "category": "Time-series",
     "doc": "Gaussianized trailing rank over window w."},
    # trading / structure
    {"name": "trade_when", "sig": "trade_when(cond, x, exit=-1)", "category": "Trading",
     "doc": "Hold position x while cond is true; exit value on exit."},
    {"name": "hump", "sig": "hump(x, h=0.01)", "category": "Trading",
     "doc": "Suppress rebalances smaller than h (fraction of book)."},
    {"name": "truncate", "sig": "truncate(x, max_weight=0.05)", "category": "Trading",
     "doc": "Clip single-name weight to max_weight."},
]

FIELDS = [
    "close", "open", "high", "low", "volume", "vwap", "returns", "adv20",
    "cap", "market_cap", "ebitda", "ebit", "assets", "liabilities", "debt",
    "sales", "opex", "equity", "rnd", "cashflow_op", "margin", "leverage",
    "sector", "industry", "subindustry", "group",
]

TEMPLATES = [
    {"name": "Momentum", "expr": "rank(ts_zscore(close, 120))",
     "desc": "Ranked trailing price momentum."},
    {"name": "Mean Reversion", "expr": "-ts_zscore(returns, 5)",
     "desc": "Short 5-day winners (reversal)."},
    {"name": "Volume Shock", "expr": "rank(volume / ts_mean(volume, 20))",
     "desc": "Volume relative to its 20-day mean."},
    {"name": "Quality", "expr": "rank(ts_backfill(ebitda, 63) / ts_backfill(assets, 63))",
     "desc": "EBITDA/assets, point-in-time backfilled."},
    {"name": "Low Volatility", "expr": "-rank(ts_std_dev(returns, 20))",
     "desc": "Short high-volatility names."},
    {"name": "Asset Turnover", "expr": "rank(ts_backfill(sales, 63) / ts_backfill(assets, 63))",
     "desc": "Sales/assets efficiency."},
    {"name": "High Volatility", "expr": "rank(ts_std_dev(returns, 20))",
     "desc": "Long high-volatility names (unusual-vol)."},
    {"name": "Slow Momentum", "expr": "ts_mean(returns, 60)",
     "desc": "Trailing 60-day mean return."},
]


# --------------------------------------------------------------------------
# Job manager (async subprocess orchestration)
# --------------------------------------------------------------------------
class JobManager:
    def __init__(self):
        self.jobs = {}

    def create(self, kind, args):
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "kind": kind, "args": args, "status": "QUEUED",
            "created_at": time.time(), "proc": None, "logs": [],
            "queues": set(), "run_ids": [], "returncode": None,
            "before": self._snapshot(),
        }
        self.jobs[job_id] = job
        return job

    @staticmethod
    def _snapshot():
        return set(os.listdir(RUNS_DIR)) if RUNS_DIR.exists() else set()

    def publish(self, job, level, text):
        entry = {"ts": time.time(), "level": level, "text": text}
        job["logs"].append(entry)
        for q in list(job["queues"]):
            q.put_nowait(entry)

    def subscribe(self, job):
        q = asyncio.Queue()
        job["queues"].add(q)
        return q


JOBS = JobManager()


def _manifest_status(run_dir):
    m = run_dir / "manifest.json"
    if not m.exists():
        return None
    try:
        return json.loads(m.read_text(encoding="utf-8")).get("status")
    except Exception:
        return None


def _new_run_ids(job):
    after = JOBS._snapshot()
    new = [d for d in (after - job["before"]) if (RUNS_DIR / d / "manifest.json").exists()]
    new.sort()
    return new


async def _pump(stream, level, job):
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip("\n")
        if text:
            JOBS.publish(job, level, text)


async def _run_cli(job, args):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-u", "-m", "fastexp.cli", *args]
    JOBS.publish(job, "SYSTEM", " ".join(cmd))
    job["status"] = "RUNNING"
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(REPO_ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except Exception as exc:
        JOBS.publish(job, "ERROR", f"failed to start process: {exc}")
        job["status"] = "ERROR"
        job["returncode"] = -1
        return
    job["proc"] = proc
    await asyncio.gather(_pump(proc.stdout, "INFO", job),
                         _pump(proc.stderr, "STDERR", job))
    rc = await proc.wait()
    job["proc"] = None
    job["returncode"] = rc
    job["run_ids"] = _new_run_ids(job)
    if rc != 0:
        job["status"] = "ERROR"
    else:
        statuses = [_manifest_status(RUNS_DIR / r) for r in job["run_ids"]]
        statuses = [s for s in statuses if s]
        if any(s == "PASS" for s in statuses):
            job["status"] = "PASS"
        elif any(s == "BLOCKED" for s in statuses):
            job["status"] = "BLOCKED"
        elif statuses and all(s == "FAIL" for s in statuses):
            job["status"] = "FAIL"
        elif not statuses:
            job["status"] = "ERROR"
        else:
            job["status"] = "ERROR"
    JOBS.publish(job, "SYSTEM",
                 f"finished rc={rc} status={job['status']} runs={job['run_ids']}")


# --------------------------------------------------------------------------
# Pydantic request models
# --------------------------------------------------------------------------
class SimulateRequest(BaseModel):
    expression: str = Field(..., min_length=1)
    dataset: str = "synthetic"
    mode: str = "fast_gate"
    seed: int = 11
    universe: str | None = None
    neutralization: str | None = None
    truncation: float | None = None
    decay: int | None = None
    include_private: bool = False


class BatchRequest(BaseModel):
    expressions: list[str] = Field(..., min_length=1)
    dataset: str = "synthetic"
    mode: str = "fast_gate"
    seed: int = 11
    universe: str | None = None
    neutralization: str | None = None
    truncation: float | None = None
    decay: int | None = None
    include_private: bool = False


class StopRequest(BaseModel):
    job_id: str


def _validate_common(dataset, mode, universe, neutralization, truncation, decay):
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if universe is not None and universe not in UNIVERSES:
        raise ValueError(f"universe must be one of {UNIVERSES}")
    if neutralization is not None and neutralization not in NEUTRALIZATIONS:
        raise ValueError(f"neutralization must be one of {NEUTRALIZATIONS}")
    if truncation is not None and not (0.0 <= truncation <= 1.0):
        raise ValueError("truncation must be in [0,1]")
    if decay is not None and decay < 0:
        raise ValueError("decay must be >= 0")


def _common_args(payload):
    args = ["--seed", str(int(payload.get("seed", 11))),
            "--dataset", payload.get("dataset", "synthetic"),
            "--mode", payload.get("mode", "fast_gate")]
    if payload.get("universe"):
        args += ["--universe", payload["universe"]]
    if payload.get("neutralization"):
        args += ["--neutralization", payload["neutralization"]]
    if payload.get("truncation") is not None:
        args += ["--truncation", str(float(payload["truncation"]))]
    if payload.get("decay") is not None:
        args += ["--decay", str(int(payload["decay"]))]
    if payload.get("include_private"):
        args += ["--include-private"]
    args += ["--output", str(RUNS_DIR)]
    return args


# --------------------------------------------------------------------------
# App + routes
# --------------------------------------------------------------------------
app = FastAPI(title="Fastexp Web Studio", version="1.0.0")


@app.get("/")
async def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/v1/operators")
async def operators():
    return {"operators": OPERATORS, "fields": FIELDS}


@app.get("/api/v1/templates")
async def templates():
    return {"templates": TEMPLATES}


@app.post("/api/v1/simulate")
async def simulate(req: SimulateRequest):
    try:
        _validate_common(req.dataset, req.mode, req.universe, req.neutralization,
                         req.truncation, req.decay)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    args = ["simulate", "--expr", req.expression] + _common_args(req.model_dump())
    job = JOBS.create("simulate", args)
    asyncio.create_task(_run_cli(job, args))
    return {"job_id": job["id"], "status": job["status"]}


@app.post("/api/v1/batch")
async def batch(req: BatchRequest):
    try:
        _validate_common(req.dataset, req.mode, req.universe, req.neutralization,
                         req.truncation, req.decay)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    return _start_batch(req.expressions, req.model_dump())


def _parse_batch_text(raw, filename=""):
    """Extract expressions from an uploaded .txt (one per line) or .jsonl file."""
    exprs = []
    if filename.lower().endswith(".jsonl"):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            e = rec.get("expression") or rec.get("expr")
            if e:
                exprs.append(e)
    else:
        exprs = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return exprs


def _start_batch(expressions, payload):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RUNS_DIR / f"_batch_{uuid.uuid4().hex[:8]}.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for i, e in enumerate(expressions):
            f.write(json.dumps({"name": f"batch_{i}", "expression": e}) + "\n")
    args = ["batch", "--input", str(tmp)] + _common_args(payload)
    job = JOBS.create("batch", args)
    asyncio.create_task(_run_cli(job, args))
    return {"job_id": job["id"], "status": job["status"]}


@app.post("/api/v1/batch/upload")
async def batch_upload(file: UploadFile = File(...),
                       dataset: str = Form("synthetic"),
                       mode: str = Form("fast_gate"),
                       seed: int = Form(11),
                       universe: str | None = Form(None),
                       neutralization: str | None = Form(None),
                       truncation: float | None = Form(None),
                       decay: int | None = Form(None),
                       include_private: bool = Form(False)):
    try:
        _validate_common(dataset, mode, universe, neutralization, truncation, decay)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    raw = (await file.read()).decode("utf-8", errors="replace")
    exprs = _parse_batch_text(raw, file.filename or "")
    if not exprs:
        return JSONResponse(status_code=422, content={"error": "no expressions found in upload"})
    payload = {"dataset": dataset, "mode": mode, "seed": seed, "universe": universe,
               "neutralization": neutralization, "truncation": truncation,
               "decay": decay, "include_private": include_private}
    return _start_batch(exprs, payload)


@app.get("/api/v1/jobs/{job_id}")
async def job_status(job_id: str):
    job = JOBS.jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return {"job_id": job_id, "status": job["status"], "kind": job["kind"],
            "returncode": job["returncode"], "run_ids": job["run_ids"]}


@app.get("/api/v1/jobs")
async def list_jobs():
    out = []
    for j in sorted(JOBS.jobs.values(), key=lambda x: x["created_at"], reverse=True):
        out.append({
            "job_id": j["id"], "kind": j["kind"], "status": j["status"],
            "created_at": j["created_at"], "returncode": j["returncode"],
            "run_ids": j["run_ids"],
        })
    return {"jobs": out}


@app.get("/api/v1/runs")
async def list_runs():
    out = []
    if not RUNS_DIR.exists():
        return {"runs": out}
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        m = d / "manifest.json"
        if not m.exists():
            continue
        try:
            man = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "run_id": man.get("run_id", d.name),
            "status": man.get("status"),
            "dataset_id": man.get("dataset_id"),
            "seed": man.get("seed"),
            "mode": man.get("mode"),
            "private_data": man.get("private_data"),
            "created_at": man.get("created_at"),
            "expression_hash": (man.get("expression_hash") or "")[:12],
        })
    return {"runs": out}


@app.post("/api/v1/runs/stop")
async def stop_run(req: StopRequest):
    job = JOBS.jobs.get(req.job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    proc = job.get("proc")
    if proc is not None and proc.returncode is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        JOBS.publish(job, "SYSTEM", "terminate signal sent")
        job["status"] = "STOPPED"
    return {"job_id": req.job_id, "status": job["status"]}


@app.websocket("/api/v1/ws/logs/{job_id}")
async def ws_logs(ws: WebSocket, job_id: str):
    await ws.accept()
    job = JOBS.jobs.get(job_id)
    if not job:
        await ws.send_text(json.dumps({"level": "SYSTEM", "text": "job not found", "done": True}))
        await ws.close()
        return
    q = JOBS.subscribe(job)
    done_sent = False
    try:
        for e in list(job["logs"]):
            await ws.send_text(json.dumps({**e, "done": False}))
        while True:
            while True:
                try:
                    e = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                await ws.send_text(json.dumps({**e, "done": False}))
            terminal = job["status"] in TERMINAL
            if terminal and q.empty():
                if not done_sent:
                    await ws.send_text(json.dumps({
                        "level": "SYSTEM", "text": "", "done": True,
                        "status": job["status"], "run_ids": job["run_ids"]}))
                    done_sent = True
                    break
            try:
                e = await asyncio.wait_for(q.get(), timeout=10)
                await ws.send_text(json.dumps({**e, "done": False}))
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"level": "PING", "text": "", "done": False}))
    except WebSocketDisconnect:
        pass
    finally:
        job["queues"].discard(q)


# Mount static assets and generated run artifacts (after routes so they don't shadow them).
RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/reports", StaticFiles(directory=str(RUNS_DIR)), name="reports")


def main():
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(prog="fastexp.server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ns = ap.parse_args()
    print(f"Fastexp Web Studio -> http://{ns.host}:{ns.port}")
    uvicorn.run("fastexp.server:app", host=ns.host, port=ns.port, reload=False)


if __name__ == "__main__":
    main()
