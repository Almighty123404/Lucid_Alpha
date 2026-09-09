"""Fastexp artifact-first CLI (python -m fastexp.cli).

Commands:
  simulate  --expr-file alpha.txt [--seed 11] [--dataset synthetic|real] [--mode fast_gate|high_fidelity]
            [--output reports/runs] [--include-private] [--verbose]
  batch     --input alphas.jsonl [--seed 11] [--output reports/runs] [--include-private]
  compare   --run-id A --run-id B [--output reports/runs]
  audit     --run-id A

Machine-parseable plaintext summary by default; diagnostic detail behind --verbose.
"""
import argparse
import json
import os
import sys

from data_gen import generate
from simulator import simulate


def _parse_bool_pasteurization(st):
    # no-op; SimulationSettings already normalizes
    return st


def _build_panel(dataset, seed):
    if dataset == "real":
        from real_data import generate_real_panel
        return generate_real_panel(seed=seed, n_stocks=200)
    return generate(seed=seed, n_stocks=1000)


def _settings_dict(args):
    """Build an optional SimulationSettings dict from CLI flags (all optional).

    Returns None when no setting was supplied, so simulate() falls back to its
    defaults. This is a pure addition; the engine contract is unchanged.
    """
    d = {}
    if getattr(args, "universe", None):
        d["universe"] = args.universe
    if getattr(args, "neutralization", None):
        d["neutralization"] = args.neutralization
    if getattr(args, "truncation", None) is not None:
        d["truncation"] = float(args.truncation)
    if getattr(args, "decay", None) is not None:
        d["decay"] = int(args.decay)
    return d or None


def _extra(rep, expr):
    """DSR + expected-real debiased metrics for the executive screen."""
    out = {}
    try:
        import selection as _sel
        from calibration import expected_real_adjustment, family_of_expression
        dsid = rep.get("dataset_id", "")
        n = _sel.TrialRegistry().count(dsid, "default")
        if n >= 1:
            out["dsr"] = round(_sel.deflated_sharpe_ratio(list(rep.get("pnl", [])), n), 4)
            out["n_trials"] = n
        env = expected_real_adjustment(family_of_expression(expr))
        from calibration import expected_real_metrics as _erm
        adj = _erm(rep.get("metrics", {}), env)
        out["expected_real_sharpe"] = adj.get("sharpe")
        out["expected_real_fitness"] = adj.get("fitness")
        out["expected_real_turnover"] = adj.get("turnover_pct")
    except Exception:
        pass
    return out


def _summary(rep, run_dir):
    m = rep.get("metrics", {})
    print(f"RUN      : {rep.get('run_id','')}")
    print(f"DATASET  : {(rep.get('dataset_id') or '')[:8]}")
    print(f"MODE     : {rep.get('mode','')}")
    print(f"EXPR_HASH: {rep.get('expression_hash','')[:12]}")
    print("-" * 40)
    print(f"Sharpe            : {m.get('sharpe',0):8.2f} | {'PASS' if rep['criteria'].get('sharpe',{}).get('pass') else 'FAIL'}")
    print(f"Fitness           : {m.get('fitness',0):8.2f} | {'PASS' if rep['criteria'].get('fitness',{}).get('pass') else 'FAIL'}")
    print(f"Turnover          : {m.get('turnover_pct',0):7.2f}%| {'PASS' if rep['criteria'].get('turnover',{}).get('pass') else 'FAIL'}")
    sub = rep.get('criteria', {}).get('subuniverse_sharpe', {})
    print(f"Sub-universe      : {sub.get('value',0):8.2f} | {'PASS' if sub.get('pass') else 'FAIL'}")
    print("-" * 40)
    print(f"DECISION : {'PASS' if rep.get('passed') else 'BLOCKED'}")
    print(f"REPORT   : {os.path.join(run_dir, 'report.html')}")


def cmd_simulate(args):
    if getattr(args, "expr", None) is not None:
        expr = args.expr.strip()
    elif os.path.exists(getattr(args, "expr_file", "") or ""):
        expr = open(args.expr_file, encoding="utf-8").read().strip()
    else:
        expr = getattr(args, "expr_file", None)
    panel = _build_panel(args.dataset, args.seed)
    rep = simulate(expr, panel, (), (), settings=_settings_dict(args), mode=args.mode)
    from reporting import build_run_artifact, hash_expr
    run_dir, manifest = build_run_artifact(rep, expr, seed=args.seed, mode=args.mode,
                                           include_private=args.include_private,
                                           extra=_extra(rep, expr) if not args.verbose else {},
                                           out_root=args.output)
    rep.update(run_id=manifest["run_id"], expression_hash=manifest["expression_hash"])
    _summary(rep, run_dir)


def cmd_batch(args):
    from reporting import build_run_artifact
    seen = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            expr = rec.get("expression") or rec.get("expr")
            if not expr:
                continue
            panel = _build_panel(args.dataset, args.seed)
            rep = simulate(expr, panel, (), (), settings=_settings_dict(args), mode=args.mode)
            build_run_artifact(rep, expr, seed=args.seed, mode=args.mode,
                               include_private=args.include_private,
                               extra=_extra(rep, expr) if not args.verbose else {},
                               out_root=args.output)
            seen += 1
    print(f"BATCH: {seen} runs written to {args.output}")


def cmd_compare(args):
    from reporting import compare_runs
    out, path = compare_runs(args.run_id, args.run_id_b, out_root=args.output)
    for r in out["rows"]:
        print(f"{r['metric']:<18} {str(r['a']):>10} {str(r['b']):>10}  Δ={r['delta']}")
    print(f"COMPARE : {path}")


def cmd_audit(args):
    from reporting import validate_manifest, load_summary
    run_dir = args.run_id if os.path.isdir(args.run_id) else os.path.join(args.output, args.run_id)
    required = ["manifest.json", "summary.json", "metrics.csv", "yearly.csv",
                "periods.csv", "equity_curve.csv", "drawdown.csv", "exposures.csv",
                "gates.json", "provenance.json", "report.html", "dashboard.html"]
    missing = [r for r in required if not os.path.exists(os.path.join(run_dir, r))]
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest(manifest)
    prov = json.load(open(os.path.join(run_dir, "provenance.json"), encoding="utf-8"))
    print(f"AUDIT    : {manifest['run_id']}")
    print(f"STATUS   : {manifest['status']}")
    print(f"FILES    : {len(required) - len(missing)}/{len(required)} present")
    if missing:
        print(f"MISSING  : {missing}")
    print(f"PRIVATE  : {'INCLUDED' if manifest['private_data'] else 'redacted (expression_hash only)'}")
    print(f"LEAK     : {'FAIL — raw expression in public artifact' if prov.get('expression') and not manifest['private_data'] else 'OK'}")
    print(f"VERDICT  : {'PASS' if not missing and (not prov.get('expression') or manifest['private_data']) else 'FAIL'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fastexp")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--expr-file", default=None)
    g.add_argument("--expr", default=None)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--dataset", default="synthetic", choices=["synthetic", "real"])
    s.add_argument("--mode", default="fast_gate", choices=["fast_gate", "high_fidelity"])
    s.add_argument("--universe", default=None, choices=["TOP3000", "TOP1000", "TOP500", "TOP200"])
    s.add_argument("--neutralization", default=None,
                   choices=["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "COUNTRY", "EXCHANGE"])
    s.add_argument("--truncation", type=float, default=None)
    s.add_argument("--decay", type=int, default=None)
    s.add_argument("--output", default="reports/runs")
    s.add_argument("--include-private", action="store_true")
    s.add_argument("--verbose", action="store_true")

    b = sub.add_parser("batch")
    b.add_argument("--input", required=True)
    b.add_argument("--seed", type=int, default=11)
    b.add_argument("--dataset", default="synthetic", choices=["synthetic", "real"])
    b.add_argument("--mode", default="fast_gate", choices=["fast_gate", "high_fidelity"])
    b.add_argument("--universe", default=None, choices=["TOP3000", "TOP1000", "TOP500", "TOP200"])
    b.add_argument("--neutralization", default=None,
                   choices=["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "COUNTRY", "EXCHANGE"])
    b.add_argument("--truncation", type=float, default=None)
    b.add_argument("--decay", type=int, default=None)
    b.add_argument("--output", default="reports/runs")
    b.add_argument("--include-private", action="store_true")
    b.add_argument("--verbose", action="store_true")

    c = sub.add_parser("compare")
    c.add_argument("--run-id", required=True)
    c.add_argument("--run-id-b", required=True)
    c.add_argument("--output", default="reports/runs")

    a = sub.add_parser("audit")
    a.add_argument("--run-id", required=True)
    a.add_argument("--output", default="reports/runs")

    args = p.parse_args(argv)
    {"simulate": cmd_simulate, "batch": cmd_batch,
     "compare": cmd_compare, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    main()
