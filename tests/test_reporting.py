"""Artifact-first reporting layer tests: manifest/schema, CSVs, privacy, compare, CLI."""
import json
import os

import numpy as np
import pytest

import reporting
from reporting import (hash_expr, validate_manifest, build_run_artifact,
                      compare_runs, build_series_arrays, make_run_id)
from simulator import simulate


@pytest.fixture(scope="module")
def rep(panel):
    return simulate("-ts_zscore(returns, 21)", panel, (), ())


def test_hash_expr_deterministic():
    assert hash_expr("rank(close)") == hash_expr("rank(close)")
    assert len(hash_expr("rank(close)")) == 64
    assert hash_expr("rank(close)") != hash_expr("rank(volume)")


def test_manifest_validation():
    good = {"run_id": "r", "git_sha": "0" * 40, "dataset_id": "d",
            "expression_hash": "a" * 64, "settings": {}, "seed": 1,
            "mode": "fast_gate", "created_at": "2026-01-01T00:00:00Z",
            "private_data": False, "status": "PASS"}
    assert validate_manifest(good)
    bad = dict(good); bad["expression_hash"] = "short"
    with pytest.raises(ValueError):
        validate_manifest(bad)
    bad = dict(good); bad["status"] = "NOPE"
    with pytest.raises(ValueError):
        validate_manifest(bad)
    bad = dict(good); bad["private_data"] = "yes"
    with pytest.raises(ValueError):
        validate_manifest(bad)


def test_build_artifact_tree_and_privacy(rep, tmp_path):
    expr = "rank(close)"
    run_dir, manifest = build_run_artifact(rep, expr, seed=11, mode="fast_gate",
                                           include_private=False, out_root=str(tmp_path))
    required = ["manifest.json", "summary.json", "metrics.csv", "yearly.csv",
                "periods.csv", "equity_curve.csv", "drawdown.csv", "exposures.csv",
                "gates.json", "provenance.json", "report.html", "dashboard.html"]
    for r in required:
        assert os.path.exists(os.path.join(run_dir, r)), r
    dash = open(os.path.join(run_dir, "dashboard.html"), encoding="utf-8").read()
    assert "cdn.plot.ly" in dash            # interactive Plotly loaded
    assert "const DATA = {" in dash         # artifact data embedded
    assert "__DATA__" not in dash           # placeholder fully substituted
    assert "#090D16" in dash and "#111726" in dash
    assert "searchParams.set('tab'" in dash
    assert "e.key>='1'" in dash
    for c in ["equity_curve", "drawdown", "rolling_sharpe", "turnover", "exposures"]:
        assert os.path.exists(os.path.join(run_dir, "charts", c + ".html"))
    prov = json.load(open(os.path.join(run_dir, "provenance.json"), encoding="utf-8"))
    assert prov["expression"] is None           # redacted by default
    assert prov["expression_hash"] == hash_expr(expr)
    assert manifest["private_data"] is False
    assert manifest["status"] in ("PASS", "FAIL", "ERROR")


def test_build_artifact_include_private(rep, tmp_path):
    expr = "rank(close)"
    _, manifest = build_run_artifact(rep, expr, seed=11, mode="fast_gate",
                                     include_private=True, out_root=str(tmp_path))
    assert manifest["private_data"] is True


def test_csv_row_counts(rep, tmp_path):
    run_dir, _ = build_run_artifact(rep, "rank(close)", seed=11, mode="fast_gate",
                                    out_root=str(tmp_path))
    eq = open(os.path.join(run_dir, "equity_curve.csv"), encoding="utf-8").readlines()
    exp = open(os.path.join(run_dir, "exposures.csv"), encoding="utf-8").readlines()
    # header + T rows
    T = len(rep["pnl"])
    assert len(eq) == T + 1
    assert len(exp) == T + 1
    assert eq[0].startswith("date,raw_pnl")
    assert exp[0].startswith("date,gross,long")


def test_series_arrays_alignment(rep):
    a = build_series_arrays(rep)
    T = len(rep["pnl"])
    assert len(a["dates"]) == T
    assert len(a["raw"]) == T and len(a["cost"]) == T
    assert len(a["turnover"]) == T


def test_compare_runs(rep, tmp_path):
    d1, _ = build_run_artifact(rep, "rank(close)", seed=11, mode="fast_gate", out_root=str(tmp_path))
    d2, _ = build_run_artifact(rep, "rank(volume)", seed=11, mode="fast_gate", out_root=str(tmp_path))
    out, path = compare_runs(d1, d2, out_root=str(tmp_path))
    assert os.path.exists(path)
    assert out["run_a"] and out["run_b"]
    assert any(r["metric"] == "sharpe" for r in out["rows"])


def test_cli_simulate(tmp_path, panel):
    from fastexp.cli import cmd_simulate
    from types import SimpleNamespace
    ef = tmp_path / "alpha.txt"
    ef.write_text("rank(close)")
    a = SimpleNamespace(expr_file=str(ef), seed=11, dataset="synthetic",
                        mode="fast_gate", include_private=False, verbose=False,
                        output=str(tmp_path / "runs"))
    cmd_simulate(a)
    runs = os.listdir(str(tmp_path / "runs"))
    assert len(runs) == 1
    assert os.path.exists(os.path.join(str(tmp_path / "runs"), runs[0], "report.html"))
