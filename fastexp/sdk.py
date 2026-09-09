"""Fastexp Python SDK (Phase 5) — notebook / scripting parity with the CLI and Web Studio.

Usage:

    import fastexp as fe

    run = fe.simulate(expr="rank(ts_zscore(close, 120))", dataset="synthetic", mode="fast_gate")
    run.show_dashboard()            # interactive Plotly dashboard inline (Jupyter)
    run.show_report()               # static SVG report (offline-safe, always renders)
    df = run.equity_curve           # pandas.DataFrame (or list of dicts without pandas)

The SDK reuses the exact same engine (`simulator.simulate`) and artifact writer
(`reporting.build_run_artifact`) as the CLI, so a run produced here is
byte-compatible with one produced by `python -m fastexp.cli simulate`.
"""
from __future__ import annotations

import csv
import json
import webbrowser
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:  # pragma: no cover - pandas is optional
    pd = None
    HAS_PANDAS = False

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO_ROOT / "reports" / "runs"


def _build_panel(dataset, seed):
    if dataset == "real":
        from real_data import generate_real_panel
        return generate_real_panel(seed=seed, n_stocks=200)
    from data_gen import generate
    return generate(seed=seed, n_stocks=1000)


def _settings_dict(universe, neutralization, truncation, decay):
    d = {}
    if universe:
        d["universe"] = universe
    if neutralization:
        d["neutralization"] = neutralization
    if truncation is not None:
        d["truncation"] = float(truncation)
    if decay is not None:
        d["decay"] = int(decay)
    return d or None


class Run:
    """A completed artifact directory, exposed with notebook-friendly accessors."""

    def __init__(self, run_dir, rep=None, expr=None):
        self.run_dir = Path(run_dir)
        self._rep = rep
        self.expr = expr
        self._manifest = None
        self._summary = None

    # -- metadata -----------------------------------------------------------
    @property
    def manifest(self):
        if self._manifest is None:
            self._manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        return self._manifest

    @property
    def summary(self):
        if self._summary is None:
            self._summary = json.loads((self.run_dir / "summary.json").read_text(encoding="utf-8"))
        return self._summary

    @property
    def run_id(self):
        return self.manifest.get("run_id")

    @property
    def status(self):
        return self.manifest.get("status")

    @property
    def passed(self):
        return bool(self.summary.get("passed"))

    @property
    def metrics(self):
        return self.summary.get("metrics", {})

    @property
    def criteria(self):
        return self.summary.get("criteria", {})

    # -- tabular artifacts --------------------------------------------------
    def _csv(self, name):
        path = self.run_dir / name
        if not path.exists():
            return None
        if HAS_PANDAS:
            return pd.read_csv(path)
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @property
    def equity_curve(self):
        return self._csv("equity_curve.csv")

    @property
    def drawdown(self):
        return self._csv("drawdown.csv")

    @property
    def exposures(self):
        return self._csv("exposures.csv")

    @property
    def yearly(self):
        return self._csv("yearly.csv")

    @property
    def periods(self):
        return self._csv("periods.csv")

    # -- display ------------------------------------------------------------
    def _ipython(self):
        try:
            from IPython import get_ipython
            shell = get_ipython()
            return shell if shell is not None else None
        except Exception:
            return None

    def show_dashboard(self, height=640):
        """Render the interactive Plotly dashboard inline (Jupyter) or in a browser.

        The dashboard is embedded via an `srcdoc` iframe so its Plotly CDN script
        and inline JS execute directly in the cell. In environments whose
        content-security policy blocks iframes, use `run.show_report()` instead.
        """
        path = self.run_dir / "dashboard.html"
        if not path.exists():
            raise FileNotFoundError(f"dashboard not found at {path}")
        ipy = self._ipython()
        if ipy is not None:
            import html as _html
            from IPython.display import HTML, display
            content = path.read_text(encoding="utf-8")
            iframe = ('<iframe srcdoc="' + _html.escape(content, quote=True)
                      + '" width="100%" height="' + str(int(height))
                      + '" style="border:0" frameborder="0"></iframe>')
            display(HTML(iframe))
            return
        webbrowser.open(path.as_uri())

    def show_report(self, height=640):
        """Render the static SVG report inline (no scripts, offline-safe) or in a browser."""
        path = self.run_dir / "report.html"
        if not path.exists():
            raise FileNotFoundError(f"report not found at {path}")
        ipy = self._ipython()
        if ipy is not None:
            from IPython.display import HTML, display
            display(HTML(path.read_text(encoding="utf-8")))
            return
        webbrowser.open(path.as_uri())

    def __repr__(self):
        return f"<Run {self.run_id} status={self.status} passed={self.passed}>"


def simulate(expr, *, dataset="synthetic", mode="fast_gate", seed=11,
             universe=None, neutralization=None, truncation=None, decay=None,
             include_private=False, out_root=None):
    """Run one expression through the engine and return a :class:`Run`.

    Mirrors `python -m fastexp.cli simulate --expr <expr> ...`; artifacts land in
    `reports/runs/<run_id>/` (or `out_root` if provided).
    """
    from simulator import simulate as _engine
    from reporting import build_run_artifact

    panel = _build_panel(dataset, seed)
    rep = _engine(expr, panel, (), (), settings=_settings_dict(universe, neutralization,
                                                               truncation, decay),
                  mode=mode)
    run_dir, _ = build_run_artifact(rep, expr, seed=seed, mode=mode,
                                    include_private=include_private,
                                    out_root=str(out_root or _DEFAULT_OUT))
    return Run(run_dir, rep=rep, expr=expr)


def batch(expressions, *, dataset="synthetic", mode="fast_gate", seed=11,
          universe=None, neutralization=None, truncation=None, decay=None,
          include_private=False, out_root=None):
    """Run several expressions and return a list of :class:`Run`."""
    runs = []
    for expr in expressions:
        runs.append(simulate(expr, dataset=dataset, mode=mode, seed=seed,
                             universe=universe, neutralization=neutralization,
                             truncation=truncation, decay=decay,
                             include_private=include_private, out_root=out_root))
    return runs
