"""Artifact-first reporting layer (reporting.py).

Decoupled from the simulation engine: the engine emits a `rep` dict (plus the
in-memory `rep['_report']` daily series); this module turns it into an
immutable run directory of JSON/CSV source-of-truth files and renders a
self-contained HTML report with inline SVG charts.

Governance guarantees enforced here:
- No browser/plotting dependency in the engine (this module is the only
  rendering path, and it uses zero third-party graphics libraries).
- Private expression text is never written unless `include_private=True`;
  public artifacts carry only `expression_hash`.
- PnL lenses are kept distinct: raw, linear cost lens, and high-fidelity
  diagnostic — none of which alters any gate (gates live in the engine).
"""
import hashlib
import json
import os
import subprocess

import numpy as np

ENGINE_VERSION = "artifact-v1"


def hash_expr(expr):
    """SHA-256 of the raw expression string (privacy-preserving identity)."""
    return hashlib.sha256(str(expr).encode("utf-8")).hexdigest()


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True, timeout=5)
        sha = out.stdout.strip()
        return sha if len(sha) == 40 else ""
    except Exception:
        return ""


def make_run_id(expr_hash, ts=None):
    import datetime as _dt
    ts = ts or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{expr_hash[:8]}"


def _status(rep):
    if rep.get("error"):
        return "ERROR"
    return "PASS" if rep.get("passed") else "FAIL"


def _manifest(rep, expr, seed, mode, include_private, run_id):
    st = rep.get("settings") or {}
    return {
        "run_id": run_id,
        "git_sha": git_sha(),
        "dataset_id": rep.get("dataset_id", "") or "",
        "expression_hash": hash_expr(expr),
        "settings": {
            "universe": st.get("universe", ""),
            "neutralization": st.get("neutralization", ""),
            "truncation": float(st.get("truncation", 0.0)),
            "pasteurization": str(st.get("pasteurization", "ON")).upper() == "ON",
        },
        "seed": int(seed),
        "mode": str(mode),
        "created_at": _iso_now(),
        "private_data": bool(include_private),
        "status": _status(rep),
    }


def _iso_now():
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def validate_manifest(m):
    """Strict contract check; raises ValueError naming the first violation."""
    req_str = ("run_id", "git_sha", "dataset_id", "expression_hash",
               "mode", "created_at", "status")
    for k in req_str:
        if not isinstance(m.get(k), str):
            raise ValueError(f"manifest.{k} must be a string")
    if len(m["expression_hash"]) != 64:
        raise ValueError("manifest.expression_hash must be SHA-256 (64 hex)")
    for k in ("seed",):
        if not isinstance(m.get(k), int):
            raise ValueError(f"manifest.{k} must be an integer")
    for k in ("private_data",):
        if not isinstance(m.get(k), bool):
            raise ValueError(f"manifest.{k} must be boolean")
    if m["status"] not in ("PASS", "FAIL", "BLOCKED", "ERROR"):
        raise ValueError("manifest.status must be PASS|FAIL|BLOCKED|ERROR")
    if not isinstance(m.get("settings"), dict):
        raise ValueError("manifest.settings must be a dict")
    return True


# --------------------------------------------------------------------------
# Series derivation (pure numpy; no engine dependency)
# --------------------------------------------------------------------------
def _cum(x):
    return np.cumsum(np.where(np.isfinite(x), x, 0.0))


def _drawdown_pct(cum):
    peak = np.maximum.accumulate(cum)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (cum - peak) / np.where(peak != 0, np.abs(peak), 1.0) * 100.0
    return np.where(np.isfinite(dd), dd, 0.0)


def _rolling_sharpe(pnl, window, ann=252.0):
    x = np.asarray(pnl, dtype=np.float64)
    T = len(x)
    out = np.full(T, np.nan)
    for t in range(window, T):
        w = x[t - window + 1:t + 1]
        sd = w.std()
        out[t] = float(w.mean() / sd * np.sqrt(ann)) if sd > 1e-12 else 0.0
    return out


def exposures_from_weights(Wd):
    Wd = np.asarray(Wd, dtype=np.float64)
    gross = np.abs(Wd).sum(1)
    longs = np.maximum(Wd, 0.0).sum(1)
    shorts = np.maximum(-Wd, 0.0).sum(1)
    net = Wd.sum(1)
    active = (np.abs(Wd) > 1e-12).sum(1)
    maxw = np.abs(Wd).max(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        conc = maxw / np.where(gross > 0, gross, 1.0)
    return gross, longs, shorts, net, active, maxw, np.where(np.isfinite(conc), conc, 0.0)


def build_series_arrays(rep):
    """Return a dict of aligned numpy arrays for the daily CSVs/charts."""
    r = rep.get("_report") or {}
    dates = [str(d) for d in r.get("dates", [])]
    Wd = np.asarray(r.get("weights", []), dtype=np.float64)
    raw = np.asarray(r.get("raw_pnl", rep.get("pnl", [])), dtype=np.float64)
    cost = np.asarray(r.get("cost_pnl", raw), dtype=np.float64)
    hf = None
    if isinstance(rep.get("high_fidelity"), dict):
        hf = np.asarray(rep["high_fidelity"].get("net_pnl"), dtype=np.float64)
        if hf.size == 0:
            hf = None
    turnover = np.asarray(r.get("turnover", []), dtype=np.float64)
    if Wd.size == 0 and raw.size:
        Wd = np.zeros((raw.size, 0))
    gross, longs, shorts, net, active, maxw, conc = exposures_from_weights(Wd) if Wd.size else (
        np.zeros(len(dates)),) * 7
    return {
        "dates": dates, "raw": raw, "cost": cost, "hf": hf, "turnover": turnover,
        "gross": gross, "longs": longs, "shorts": shorts, "net": net,
        "active": active, "maxw": maxw, "conc": conc,
    }


# --------------------------------------------------------------------------
# CSV writers
# --------------------------------------------------------------------------
def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(_csv_cell(v) for v in row) + "\n")


def _csv_cell(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def write_metrics_csv(path, metrics):
    _write_csv(path, list(metrics.keys()), [list(metrics.values())])


def write_yearly_csv(path, yearly):
    if not yearly:
        return
    keys = list(yearly[0].keys())
    _write_csv(path, keys, [[r.get(k, "") for k in keys] for r in yearly])


def write_periods_csv(path, periods):
    if not periods:
        return
    rows = []
    for name, m in periods.items():
        rows.append([name] + [m.get(k, "") for k in
                              ("sharpe", "fitness", "turnover_pct", "returns_pct", "n_days")])
    _write_csv(path, ["period", "sharpe", "fitness", "turnover_pct", "returns_pct", "n_days"], rows)


def write_equity_csv(path, a):
    raw_cum = _cum(a["raw"])
    cost_cum = _cum(a["cost"])
    dd = _drawdown_pct(raw_cum)
    _write_csv(path, ["date", "raw_pnl", "raw_equity", "cost_pnl", "cost_equity",
                      "drawdown_pct"],
               zip(a["dates"], a["raw"], raw_cum, a["cost"], cost_cum, dd))


def write_drawdown_csv(path, a):
    dd = _drawdown_pct(_cum(a["raw"]))
    _write_csv(path, ["date", "drawdown_pct"], zip(a["dates"], dd))


def write_exposures_csv(path, a):
    _write_csv(path, ["date", "gross", "long", "short", "net", "active_names",
                      "max_weight", "concentration"],
               zip(a["dates"], a["gross"], a["longs"], a["shorts"], a["net"],
                   a["active"], a["maxw"], a["conc"]))


# --------------------------------------------------------------------------
# Minimal SVG chart helpers (zero external deps, deterministic)
# --------------------------------------------------------------------------
def _svg_line(series_list, width=640, height=200, names=None, title="", x=None):
    """series_list: list of 1-D arrays to overlay; names optional labels."""
    import html as _html
    series = [np.asarray(s, dtype=np.float64) for s in series_list]
    flat = [v for s in series for v in np.where(np.isfinite(s), s, np.nan)]
    flat = [v for v in flat if not np.isnan(v)]
    if not flat:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(min(flat)), float(max(flat))
        if hi - lo < 1e-12:
            hi = lo + 1.0
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    n = max(len(s) for s in series)
    xs = np.linspace(0, width, n)
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    def _pts(s, col):
        y = np.interp(xs, np.arange(len(s)), np.where(np.isfinite(s), s, np.nan))
        y = (hi - y) / (hi - lo) * (height - 10) + 5
        parts = []
        for xx, yy in zip(xs, y):
            if np.isnan(yy):
                parts.append("")
            else:
                parts.append(f"{xx:.1f},{yy:.1f}")
        joined = " ".join(p for p in parts if p)
        return f'<polyline fill="none" stroke="{col}" stroke-width="1.5" points="{joined}"/>'

    body = "\n".join(_pts(s, cols[i % len(cols)]) for i, s in enumerate(series))
    legend = ""
    if names:
        legend = "".join(
            f'<text x="{(10 + i * 120)}" y="{height - 2}" font-size="10" fill="{cols[i % len(cols)]}">{_html.escape(str(n))}</text>'
            for i, n in enumerate(names))
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'preserveAspectRatio="none" style="width:100%;height:{height}px">'
            f'<rect width="100%" height="100%" fill="#fff"/>'
            f'<text x="6" y="12" font-size="11" fill="#333">{_html.escape(title)}</text>'
            f'{body}{legend}</svg>')


# --------------------------------------------------------------------------
# HTML report (5 screens)
# --------------------------------------------------------------------------
def _criteria_table(rep):
    rows = []
    for name, c in rep.get("criteria", {}).items():
        mark = "PASS" if c["pass"] else "FAIL"
        rows.append(f"<tr><td>{name}</td><td>{c['value']}</td>"
                    f"<td>{c['requirement']}</td><td class='{'pass' if c['pass'] else 'fail'}'>{mark}</td></tr>")
    return "".join(rows)


def _screen1(rep, extra):
    m = rep.get("metrics", {})
    decision = "BLOCKED" if not rep.get("passed") else "PASS"
    extra_rows = ""
    if extra:
        for k, v in extra.items():
            if isinstance(v, dict):
                continue
            extra_rows += f"<tr><td>{k}</td><td>{v}</td><td>—</td><td class='info'>INFO</td></tr>"
    return f"""<section><h2>1. Executive Summary</h2>
<table><tr><th>Metric</th><th>Value</th><th>Requirement</th><th>Status / Lens</th></tr>
{extra_rows}{_criteria_table(rep)}
<tr><td colspan='3' class='final'><strong>FINAL DECISION</strong></td>
<td class='{'pass' if rep.get('passed') else 'fail'}'><strong>{decision}</strong></td></tr>
</table></section>"""


def _screen2(rep, a):
    raw_cum = _cum(a["raw"]); cost_cum = _cum(a["cost"])
    overlay = _svg_line([raw_cum, cost_cum] + ([a["hf"]] if a["hf"] is not None else []),
                        names=["Raw PnL", "Cost Lens"] + (["HF Diagnostic"] if a["hf"] is not None else []),
                        title="Cumulative PnL (multi-lens)")
    dd = _svg_line([_drawdown_pct(raw_cum)], names=["Drawdown"], title="Underwater drawdown (%)")
    rs63 = _rolling_sharpe(a["raw"], 63); rs252 = _rolling_sharpe(a["raw"], 252)
    rsh = _svg_line([rs63, rs252], names=["63d", "252d"], title="Rolling Sharpe")
    to = _svg_line([a["turnover"] * 100.0], names=["Turnover"], title="Daily turnover (%)")
    return f"""<section><h2>2. Equity &amp; PnL Dynamics</h2>
{overlay}<div class='chart'>{dd}</div><div class='chart'>{rsh}</div><div class='chart'>{to}</div></section>"""


def _screen3(rep, a):
    exp = _svg_line([a["gross"], a["longs"], a["shorts"], a["net"]],
                    names=["Gross", "Long", "Short", "Net"], title="Exposure")
    conc = _svg_line([a["conc"] * 100.0], names=["Concentration"], title="Weight concentration (%)")
    active = _svg_line([a["active"]], names=["Active"], title="Active names")
    return f"""<section><h2>3. Portfolio Behaviour &amp; Microstructure</h2>
<div class='chart'>{exp}</div><div class='chart'>{conc}</div><div class='chart'>{active}</div></section>"""


def _screen4(rep):
    risk = rep.get("risk") or {}
    hist = risk.get("hist", {})
    rows = "".join(
        f"<tr><td>{k}</td><td>{v.get('var', '')}</td><td>{v.get('es', '')}</td></tr>"
        for k, v in hist.items())
    return f"""<section><h2>4. Risk &amp; Stress</h2>
<table><tr><th>Quantile</th><th>Historical VaR</th><th>Historical ES</th></tr>{rows}</table></section>"""


def _screen5(real_pairs):
    if not real_pairs:
        return "<section><h2>5. Sim-vs-Real Calibration</h2><p>No paired Brain data supplied.</p></section>"
    rows = "".join(
        f"<tr><td>{p.get('metric','')}</td><td>{p.get('sim','')}</td>"
        f"<td>{p.get('real','')}</td><td>{p.get('bias','')}</td></tr>" for p in real_pairs)
    return f"""<section><h2>5. Sim-vs-Real Calibration</h2>
<table><tr><th>Metric</th><th>Sim</th><th>Real</th><th>Bias</th></tr>{rows}</table></section>"""


def render_report(run_dir, rep, extra=None, real_pairs=None):
    extra = extra or {}
    a = build_series_arrays(rep)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Run {rep.get('dataset_id','')[:8]} — Fastexp</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
table{{border-collapse:collapse;margin:8px 0;width:100%}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:right;font-size:13px}}
th{{background:#f4f4f4;text-align:left}}
td:first-child,th:first-child{{text-align:left}}
.pass{{color:#1a7f37;font-weight:600}}.fail{{color:#c62828;font-weight:600}}.info{{color:#555}}
.final{{text-align:right}}
.chart{{margin:12px 0}}section{{margin:24px 0}}h2{{border-bottom:1px solid #ccc;padding-bottom:4px}}</style></head><body>
<h1>Fastexp Run Report — {rep.get('run_id','')}</h1>
{_screen1(rep, extra)}
{_screen2(rep, a)}
{_screen3(rep, a)}
{_screen4(rep)}
{_screen5(real_pairs)}
</body></html>"""
    with open(os.path.join(run_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.join(run_dir, "report.html")


# --------------------------------------------------------------------------
# Artifact builder (Phase 1+2 orchestration)
# --------------------------------------------------------------------------
def build_run_artifact(rep, expr, *, seed, mode, include_private=False,
                       extra=None, real_pairs=None, out_root="reports/runs",
                       panel=None):
    extra = extra or {}
    expr_hash = hash_expr(expr)
    run_id = make_run_id(expr_hash)
    run_dir = os.path.join(out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    manifest = _manifest(rep, expr, seed, mode, include_private, run_id)
    validate_manifest(manifest)

    summary = {
        "run_id": run_id,
        "metrics": rep.get("metrics", {}),
        "criteria": rep.get("criteria", {}),
        "passed": bool(rep.get("passed")),
        "status": _status(rep),
        "yearly": rep.get("yearly", []),
        "periods": rep.get("periods", {}),
        "subuniverse_cutoff": rep.get("subuniverse_cutoff"),
        "max_weight_date": rep.get("max_weight_date"),
        "max_weight_stock": rep.get("max_weight_stock"),
        "risk": rep.get("risk", {}),
        "high_fidelity": {k: v for k, v in (rep.get("high_fidelity") or {}).items()
                          if not k.startswith("_") and not isinstance(v, np.ndarray)},
        "extra": extra,
    }

    def w(name, obj):
        with open(os.path.join(run_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_json_default)

    w("manifest.json", manifest)
    w("summary.json", summary)
    w("gates.json", rep.get("criteria", {}))
    w("provenance.json", {
        "git_sha": manifest["git_sha"], "dataset_id": manifest["dataset_id"],
        "expression_hash": expr_hash, "expression": expr if include_private else None,
        "settings": rep.get("settings", {}), "seed": seed, "mode": mode,
        "created_at": manifest["created_at"], "private_data": include_private,
        "status": manifest["status"], "engine_version": ENGINE_VERSION,
        "real_pairs": real_pairs,
    })

    # CSVs
    a = build_series_arrays(rep)
    write_metrics_csv(os.path.join(run_dir, "metrics.csv"), rep.get("metrics", {}))
    write_yearly_csv(os.path.join(run_dir, "yearly.csv"), rep.get("yearly", []))
    write_periods_csv(os.path.join(run_dir, "periods.csv"), rep.get("periods", {}))
    write_equity_csv(os.path.join(run_dir, "equity_curve.csv"), a)
    write_drawdown_csv(os.path.join(run_dir, "drawdown.csv"), a)
    write_exposures_csv(os.path.join(run_dir, "exposures.csv"), a)

    # charts + report
    os.makedirs(os.path.join(run_dir, "charts"), exist_ok=True)
    for name, svg in _charts(a).items():
        with open(os.path.join(run_dir, "charts", name + ".html"), "w", encoding="utf-8") as f:
            f.write(_wrap_chart(svg, name))
    render_report(run_dir, {**rep, "run_id": run_id}, extra=extra, real_pairs=real_pairs)
    render_dashboard(run_dir, {**rep, "run_id": run_id}, extra=extra, real_pairs=real_pairs)

    return run_dir, manifest


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _wrap_chart(svg, name):
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{name}</title>"
            f"<style>body{{font-family:system-ui;margin:24px}}</style></head><body>{svg}</body></html>")


def _charts(a):
    raw_cum = _cum(a["raw"]); cost_cum = _cum(a["cost"])
    dd = _drawdown_pct(raw_cum)
    return {
        "equity_curve": _svg_line([raw_cum, cost_cum] + ([a["hf"]] if a["hf"] is not None else []),
                                  names=["Raw", "Cost"] + (["HF"] if a["hf"] is not None else []),
                                  title="Equity curve"),
        "drawdown": _svg_line([dd], names=["Drawdown"], title="Drawdown (%)"),
        "rolling_sharpe": _svg_line([_rolling_sharpe(a["raw"], 63), _rolling_sharpe(a["raw"], 252)],
                                    names=["63d", "252d"], title="Rolling Sharpe"),
        "turnover": _svg_line([a["turnover"] * 100.0], names=["Turnover"], title="Turnover (%)"),
        "exposures": _svg_line([a["gross"], a["longs"], a["shorts"], a["net"]],
                               names=["Gross", "Long", "Short", "Net"], title="Exposure"),
    }


# --------------------------------------------------------------------------
# Interactive dashboard (Phase 7) — read-only Plotly view over the same data
# --------------------------------------------------------------------------
def _js_arr(x):
    """JSON-safe list with non-finite -> None (Plotly treats nulls as gaps)."""
    return [None if (isinstance(v, float) and not np.isfinite(v)) else float(v)
            for v in np.asarray(x, dtype=np.float64)]


def _monthly_heatmap(dates, raw):
    """Sum daily PnL into a year x month matrix for the heatmap."""
    import datetime as _dt
    years, months = [], []
    for d in dates:
        try:
            y = int(d[:4]); m = int(d[5:7])
        except (ValueError, TypeError):
            y, m = 0, 0
        years.append(y); months.append(m)
    uy = sorted(set(y for y in years if y > 0))
    z = [[0.0] * 12 for _ in uy]
    for i, (y, m, p) in enumerate(zip(years, months, raw)):
        if y in uy and 1 <= m <= 12:
            z[uy.index(y)][m - 1] += float(p)
    return uy, z


_DASH_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<title>Fastexp Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{--bg-primary:#090D16;--bg-surface:#111726;--border-subtle:#1E293B;--pass:#10B981;--fail:#EF4444;--diag:#3B82F6;--warn:#F59E0B;--text:#CBD5E1;--muted:#8491A5}
*{box-sizing:border-box}body{font-family:Inter,system-ui,sans-serif;margin:0;background:var(--bg-primary);color:var(--text)}
header{background:#0B0E14;border-bottom:1px solid var(--border-subtle);color:#fff;padding:12px 20px}
header h1{margin:0;font-size:18px}header .sub{color:#9ca3af;font-size:12px}
.wrap{display:grid;grid-template-columns:210px 1fr;min-height:100vh}
nav{background:#0B0E14;border-right:1px solid var(--border-subtle);padding:12px 8px}
nav a{display:block;padding:8px 10px;color:var(--muted);text-decoration:none;border-radius:5px;font-size:12px}
nav a:hover,nav a.active{background:#182033;color:#93C5FD}
main{padding:16px;max-width:1400px;width:100%}
.card{background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:6px;padding:12px;margin-bottom:12px}
.card h2{font-size:13px;margin:0 0 10px;color:#F8FAFC;letter-spacing:.02em}
.chips{display:flex;flex-wrap:wrap;gap:10px}
.chip{padding:5px 9px;border-radius:4px;font-size:11px;font-weight:600;font-family:"JetBrains Mono","Fira Code",monospace}
.chip.pass{background:#063b2d;color:#6EE7B7}.chip.fail{background:#48151B;color:#FDA4AF}.chip.info{background:#172554;color:#93C5FD}
table{width:100%;border-collapse:collapse;font-size:11px;font-family:"JetBrains Mono","Fira Code",monospace}
th,td{padding:6px 8px;border-bottom:1px solid var(--border-subtle);text-align:right}
th:first-child,td:first-child{text-align:left}
.plot{width:100%;height:340px;background:var(--bg-surface)}
.decision{font-size:20px;padding:12px 18px;border-radius:10px;margin-top:12px;text-align:center}
.decision.pass{background:#063b2d;color:#6EE7B7}.decision.fail{background:#48151B;color:#FDA4AF}
.note{font-size:11px;color:var(--muted)}
@media(max-width:760px){.wrap{display:block}nav{display:flex;overflow:auto;gap:4px}nav a{white-space:nowrap}main{padding:10px}.plot{height:280px}}
</style></head><body>
"""

_DASH_FOOT = """</main></div>
<script>
const DATA = __DATA__;
const DARK = {paper_bgcolor:'#111726', plot_bgcolor:'#111726', font:{color:'#CBD5E1'},
  margin:{t:28,r:16,b:42,l:54}, xaxis:{gridcolor:'#1E293B', linecolor:'#334155',
  rangeslider:{visible:true, thickness:.08}}, yaxis:{gridcolor:'#1E293B', linecolor:'#334155'}};
function plot(id, data, layout){
  const el = document.getElementById(id);
  if(el && window.Plotly){
    const merged=Object.assign({}, DARK, layout||{});
    merged.xaxis=Object.assign({},DARK.xaxis,(layout||{}).xaxis||{});
    merged.yaxis=Object.assign({},DARK.yaxis,(layout||{}).yaxis||{});
    Plotly.newPlot(id, data, merged, {responsive:true, displaylogo:false, scrollZoom:true});
  }
}
function line(x, y, name, color, fill){
  return {x:x, y:y, name:name, type:'scatter', mode:'lines', line:{color:color, width:1.6},
          fill: fill||'none', hovertemplate:'%{y:.4f}<extra>'+name+'</extra>'};
}
const D = DATA;
const eq = [line(D.dates, D.raw_cum, 'Raw PnL', '#1f77b4'),
            line(D.dates, D.cost_cum, 'Cost Lens', '#d62728')];
if(D.hf_cum) eq.push(line(D.dates, D.hf_cum, 'HF Diagnostic', '#2ca02c'));
plot('eq', eq, {margin:{t:20}, showlegend:true, xaxis:{title:'Date'}, yaxis:{title:'Cumulative PnL'}});
plot('dd', [line(D.dates, D.drawdown, 'Drawdown %', '#c62828', 'tozeroy')], {margin:{t:20}, yaxis:{title:'%'}});
plot('rsh', [line(D.dates, D.rs63, '63d', '#1f77b4'), line(D.dates, D.rs252, '252d', '#d62728')], {margin:{t:20}, showlegend:true});
plot('to', [line(D.dates, D.turnover, 'Turnover %', '#1f77b4')], {margin:{t:20}, yaxis:{title:'%'}});
plot('exp', [line(D.dates, D.gross, 'Gross', '#111827'), line(D.dates, D.longs, 'Long', '#166534'),
            line(D.dates, D.shorts, 'Short', '#991b1b'), line(D.dates, D.net, 'Net', '#4f46e5')], {margin:{t:20}, showlegend:true});
plot('conc', [line(D.dates, D.conc, 'Concentration %', '#b45309')], {margin:{t:20}, yaxis:{title:'%'}});
plot('active', [line(D.dates, D.active, 'Active names', '#0f766e')], {margin:{t:20}});
plot('heat', [{z:D.hm_z, x:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
              y:D.hm_y, type:'heatmap', colorscale:'RdBu', reversescale:true, colorbar:{title:'PnL'}}],
     {margin:{t:20}, xaxis:{side:'top'}});
plot('year', [{x:D.year_x, y:D.year_y, type:'bar', marker:{color:'#4f46e5'}}], {margin:{t:20}, yaxis:{title:'Sharpe'}});
plot('hist', [{x:D.daily, type:'histogram', nbinsx:60, marker:{color:'#64748b'}}], {margin:{t:20}, xaxis:{title:'Daily PnL'}});
function activate(tab, write){
  document.querySelectorAll('nav a').forEach(x=>x.classList.toggle('active', x.dataset.t===tab));
  document.querySelectorAll('section.panel').forEach(s=>s.style.display=s.id===tab?'block':'none');
  if(write){ const u=new URL(location.href); u.searchParams.set('tab', tab); history.replaceState(null,'',u); }
}
document.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',(e)=>{e.preventDefault();activate(a.dataset.t,true);}));
const initial=new URLSearchParams(location.search).get('tab')||'overview';
activate(['overview','equity','portfolio','risk','yearly'].includes(initial)?initial:'overview',false);
document.addEventListener('keydown',(e)=>{
  if(e.key>='1'&&e.key<='5'){e.preventDefault();activate(['overview','equity','portfolio','risk','yearly'][Number(e.key)-1],true);}
});
</script>
</body></html>"""


def render_dashboard(run_dir, rep, extra=None, real_pairs=None):
    a = build_series_arrays(rep)
    extra = extra or {}
    years, hm = _monthly_heatmap(a["dates"], a["raw"])
    data = {
        "run_id": rep.get("run_id", ""),
        "dates": a["dates"],
        "raw_cum": _js_arr(_cum(a["raw"])),
        "cost_cum": _js_arr(_cum(a["cost"])),
        "hf_cum": _js_arr(_cum(a["hf"])) if a["hf"] is not None else None,
        "drawdown": _js_arr(_drawdown_pct(_cum(a["raw"]))),
        "rs63": _js_arr(_rolling_sharpe(a["raw"], 63)),
        "rs252": _js_arr(_rolling_sharpe(a["raw"], 252)),
        "turnover": _js_arr(a["turnover"] * 100.0),
        "gross": _js_arr(a["gross"]), "longs": _js_arr(a["longs"]),
        "shorts": _js_arr(a["shorts"]), "net": _js_arr(a["net"]),
        "conc": _js_arr(a["conc"] * 100.0), "active": _js_arr(a["active"]),
        "hm_z": hm, "hm_y": [str(y) for y in years],
        "year_x": [r["year"] for r in rep.get("yearly", [])],
        "year_y": [r.get("sharpe", 0.0) for r in rep.get("yearly", [])],
        "daily": _js_arr(a["raw"]),
        "criteria": rep.get("criteria", {}),
        "metrics": rep.get("metrics", {}),
        "extra": extra,
        "passed": bool(rep.get("passed")),
        "status": _status(rep),
        "mode": rep.get("mode", ""),
        "dataset_id": (rep.get("dataset_id") or "")[:8],
    }
    chips = "".join(
        f'<span class="chip {"pass" if c.get("pass") else "fail"}">{k}: {c.get("value")}</span>'
        for k, c in rep.get("criteria", {}).items())
    extra_chips = "".join(
        f'<span class="chip info">{k}: {v}</span>' for k, v in extra.items()
        if not isinstance(v, dict))
    decision = "PASS" if rep.get("passed") else "BLOCKED"
    html = (_DASH_HEAD +
        f"""<header><h1>Fastexp Dashboard</h1>
<div class="sub">run {rep.get('run_id','')} &middot; dataset {data['dataset_id']} &middot; mode {data['mode']} &middot; status {data['status']}</div></header>
<div class="wrap"><nav>
<a href="#overview" data-t="overview" class="active">Overview</a>
<a href="#equity" data-t="equity">Equity &amp; PnL</a>
<a href="#portfolio" data-t="portfolio">Portfolio</a>
<a href="#risk" data-t="risk">Risk &amp; Stress</a>
<a href="#yearly" data-t="yearly">Yearly &amp; Monthly</a>
</nav><main>
<section class="panel" id="overview">
<div class="card"><h2>Gate Decision</h2><div class="chips">{chips}{extra_chips}</div>
<div class="decision {'pass' if rep.get('passed') else 'fail'}">{decision}</div></div>
<div class="card"><h2>Metrics</h2><table>{''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in rep.get('metrics', {}).items())}</table></div>
</section>
<section class="panel" id="equity" style="display:none">
<div class="card"><h2>Cumulative PnL (multi-lens)</h2><div class="plot" id="eq"></div></div>
<div class="card"><h2>Underwater Drawdown</h2><div class="plot" id="dd"></div></div>
<div class="card"><h2>Rolling Sharpe</h2><div class="plot" id="rsh"></div></div>
</section>
<section class="panel" id="portfolio" style="display:none">
<div class="card"><h2>Turnover</h2><div class="plot" id="to"></div></div>
<div class="card"><h2>Exposure</h2><div class="plot" id="exp"></div></div>
<div class="card"><h2>Concentration &amp; Active Names</h2><div class="plot" id="conc"></div><div class="plot" id="active"></div></div>
</section>
<section class="panel" id="risk" style="display:none">
<div class="card"><h2>Daily PnL Distribution</h2><div class="plot" id="hist"></div></div>
<div class="card"><h2>Risk Lens</h2><table>{''.join('<tr><td>' + str(k) + '</td><td>' + str(v.get("var","")) + '</td><td>' + str(v.get("es","")) + '</td></tr>' for k, v in (rep.get("risk") or {}).get("hist", {}).items())}</table>
<p class="note">VaR/ES and high-fidelity lenses are diagnostic-only and never enter promotion gates.</p></div>
</section>
<section class="panel" id="yearly" style="display:none">
<div class="card"><h2>Yearly Sharpe</h2><div class="plot" id="year"></div></div>
<div class="card"><h2>Monthly PnL Heatmap</h2><div class="plot" id="heat"></div></div>
</section>""" + _DASH_FOOT.replace("__DATA__", json.dumps(data)))
    path = os.path.join(run_dir, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# --------------------------------------------------------------------------
# Compare (Phase 5)
# --------------------------------------------------------------------------
def load_summary(run_dir):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def compare_runs(run_a, run_b, out_root="reports/runs"):
    sa, sb = load_summary(run_a), load_summary(run_b)
    ma, mb = sa.get("metrics", {}), sb.get("metrics", {})
    keys = sorted(set(ma) | set(mb))
    rows = []
    for k in keys:
        va, vb = ma.get(k), mb.get(k)
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = round(float(va) - float(vb), 4)
        rows.append({"metric": k, "a": va, "b": vb, "delta": delta})
    out = {"run_a": sa.get("run_id"), "run_b": sb.get("run_id"), "rows": rows}
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, f"{sa.get('run_id')}_vs_{sb.get('run_id')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    return out, path
