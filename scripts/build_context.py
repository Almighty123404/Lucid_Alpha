"""Build context.json: machine-generated project context snapshot.

Everything factual in here is read from the repo at build time (git log,
config values, risk register, calibration records, test inventory) — never
hand-copied. Rebuild after material changes: python scripts/build_context.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    from config import (GATE_VERSION, GATE_INVARIANTS, CONFIG_VERSION,
                        DEFAULT_SETTINGS, DEFAULT_EXECUTION)
    from selection import DSR_PROMOTE
    ctx = {}
    ctx["generated_note"] = ("Machine-generated snapshot. Rebuild with "
                             "python scripts/build_context.py after material changes.")
    ctx["git"] = {"sha": sh(["git", "rev-parse", "--short", "HEAD"]),
                  "dirty": bool(sh(["git", "status", "--short"])),
                  "log": sh(["git", "log", "--oneline", "-5"]).splitlines()}
    ctx["config"] = {"config_version": CONFIG_VERSION,
                     "gate_version": GATE_VERSION,
                     "gate_invariants": GATE_INVARIANTS,
                     "default_settings": DEFAULT_SETTINGS,
                     "default_execution": DEFAULT_EXECUTION,
                     "dsr_promote": DSR_PROMOTE}
    mods = ["simulator.py", "data_gen.py", "real_data.py", "fastexpr.py",
            "agents.py", "competition.py", "gp.py", "selection.py", "risk.py",
            "pit.py", "kernels.py", "providers.py", "calibration.py",
            "config.py", "run.py"]
    ctx["modules"] = {m: os.path.exists(os.path.join(ROOT, m)) for m in mods}
    rr = json.load(open(os.path.join(ROOT, "risk_register.json")))
    ctx["risk_register"] = {"n_entries": len(rr["entries"]),
                            "ids": [e["id"] for e in rr["entries"]],
                            "open": [e["id"] for e in rr["entries"]
                                     if e.get("status") == "open"]}
    import calibration
    recs = calibration.load_records()
    ctx["calibration"] = {
        "n_records": len(recs),
        "with_real": sum(1 for r in recs if r.get("real_metrics")),
        "bias_hint": ("predicted-minus-real means; Sharpe/Fitness optimistic, "
                      "turnover/drawdown underpredicted. See calibration.summarize()."),
        "bias_table": calibration.summarize(),
        "expected_real": ("calibration.expected_real_adjustment() debiases sim "
                          "metrics with the live table (min 3 pairs); "
                          "selection.promotion_eligible(..., expect_real=...) blocks "
                          "promotion unless bias-adjusted Sharpe/Fitness clear cutoffs. "
                          "Opt-in 7th promotion criterion (tightening, gate-invariant)."),
    }
    priv_path = os.path.join(ROOT, "calibration", "private_records.jsonl")
    if os.path.exists(priv_path):
        priv = [json.loads(line) for line in open(priv_path) if line.strip()]
        ev = [r for r in priv if r.get("predicted_metrics") is not None]
        ctx["calibration"]["private_pairs"] = {
            "n": len(priv), "evaluable": len(ev),
            "note": ("User-supplied Brain alphas, LOCAL ONLY (gitignored, never "
                     "committed). 2026-09 batch: sim pessimistic on 4/5 evaluable "
                     "(S deltas -0.65,-0.40,-0.10,-0.03), optimistic only on #2 "
                     "(+0.29); TO underpredicted worst at high turnover (#2: -22.6). "
                     "Opposite sign vs old +1.81 bias: over-correction or family mix. "
                     "Expressions intentionally absent from this file."),
        }
    tdir = os.path.join(ROOT, "tests")
    ctx["tests"] = sorted(f for f in os.listdir(tdir)
                          if f.startswith("test_") and f.endswith(".py"))
    ctx["pipeline"] = (
        "run.py/competition -> providers/data_gen+real_data(+pit) -> "
        "fastexpr eval -> simulator stages (universe/pasteurize, neutralize, "
        "decay, weights/truncate, delay, PnL, metrics, S1 sub-universe, 4Y "
        "self-corr, 6 gates, trial log, risk lens, optional high-fidelity) -> "
        "agents optimize (skeleton/dimension/lesson/parsimony/alignment) -> "
        "competition promotion block (gates AND lineage AND DSR>=0.95 AND "
        "optional StepM AND optional expected-real debias) -> gp evolve (3D Pareto: fitness/nodes/tail_gap) -> "
        "calibration records + risk register + runs/lessons/trials logs"
    )
    ctx["invariants"] = [
        "GATE_VERSION pins all thresholds; loosening fails CI (tests/test_gates_invariant.py).",
        "No dataset_id or no N_trials entry -> DSR=0, blocked from promotion.",
        "Gates never loosened to raise pass rates; tightening allowed.",
        "Cost lens, risk lens, high-fidelity mode, stability/complexity: diagnostic-only, never gates.",
        "PBO info-only below ~10 models; StepM needs explicit peer_pnls.",
        "GP search-panel eligibility is not promotion; full-panel re-evaluation required.",
    ]
    ctx["open_tracks"] = [
        "2019 calm-failure conditioner (crowding-flavored, not vol-flavored).",
        "alpha_size=min(tier,N) on undersized panels (needs 2nd real data point).",
        "2022-vol 2x overshoot + P(panic)~0.2 dilution (stress-calibration tuning).",
        "Leverage magnitude + turnover floor (sign closed, magnitude uncalibrated).",
        "CPCV fold design for selection-stage multiplicity (Agent 4 spec 2a).",
        "Licensed constituents/filing timestamps (retires 2 biggest APPROXes).",
    ]
    out = os.path.join(ROOT, "context.json")
    with open(out, "w") as f:
        json.dump(ctx, f, indent=2, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
