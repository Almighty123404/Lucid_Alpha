"""Build context.json: machine-generated project context snapshot.

Everything factual in here is read from the repo at build time (git log,
config values, risk register, calibration records, test inventory, live
panel field lists) — never hand-copied. Rebuild after material changes:
python scripts/build_context.py
"""
import ast
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


def test_inventory():
    """Per-file test function names via AST (cheap: no imports, no panel builds)."""
    tdir = os.path.join(ROOT, "tests")
    inv = {}
    for f in sorted(os.listdir(tdir)):
        if not (f.startswith("test_") and f.endswith(".py")):
            continue
        try:
            tree = ast.parse(open(os.path.join(tdir, f)).read())
            inv[f] = sorted(n.name for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))
        except SyntaxError:
            inv[f] = ["<unparseable>"]
    return inv


def codebook_snapshot():
    """Live field inventory + coverage from a tiny panel (cheap build)."""
    try:
        import numpy as _np
        from data_gen import generate
        p = generate(seed=11, n_stocks=20, start="2020-01-01", end="2020-03-31")
        fields = {}
        for k, v in p.fields.items():
            fields[k] = round(float(_np.isfinite(v).mean()), 3)
        return {"n_matrix": len(p.fields), "matrix": fields,
                "n_vector": len(p.vector_fields),
                "vector": {k: len(v) for k, v in p.vector_fields.items()},
                "groups": {k: len(set(map(int, _np.asarray(v).ravel())))
                           for k, v in p.groups.items()},
                "panel_note": "seed=11, n=20, 2020Q1; coverage indicative only"}
    except Exception as e:
        return {"error": f"panel build failed: {e}"}


def main():
    from config import (GATE_VERSION, GATE_INVARIANTS, CONFIG_VERSION,
                        DEFAULT_SETTINGS, DEFAULT_EXECUTION)
    from selection import DSR_PROMOTE
    ctx = {}
    ctx["generated_note"] = ("Machine-generated snapshot. Rebuild with "
                             "python scripts/build_context.py after material changes.")
    ctx["git"] = {"sha": sh(["git", "rev-parse", "--short", "HEAD"]),
                  "dirty": bool(sh(["git", "status", "--short"])),
                  "log": sh(["git", "log", "--oneline", "-15"]).splitlines(),
                  "diffstat": sh(["git", "diff", "--stat"])}
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
                            "open": [e for e in rr["entries"]
                                     if e.get("status") == "open"],
                            "approx": [e["id"] for e in rr["entries"]
                                       if e.get("status") == "approx"],
                            "validated": [e["id"] for e in rr["entries"]
                                          if e.get("status") == "validated"]}
    import calibration
    recs = calibration.load_records()
    paired = [r for r in recs if r.get("real_metrics") and r.get("predicted_metrics")]
    rows = [{"expr": (r.get("expression") or "")[:90],
             "real_S": (r.get("real_metrics") or {}).get("sharpe"),
             "real_F": (r.get("real_metrics") or {}).get("fitness"),
             "pred_S": (r.get("predicted_metrics") or {}).get("sharpe"),
             "pred_F": (r.get("predicted_metrics") or {}).get("fitness"),
             "family": calibration.family_of_expression(r.get("expression", ""))}
            for r in paired]
    fams = {}
    for fam in ("microstructure", "fundamental", "derivatives", "sentiment",
                "mixed", "unknown"):
        try:
            fams[fam] = calibration.expected_real_adjustment(fam)
        except Exception as e:
            fams[fam] = {"error": str(e)}
    ctx["calibration"] = {
        "n_records": len(recs),
        "with_real": len(paired),
        "pairs": rows,
        "bias_table_global": calibration.summarize(),
        "bias_by_family": fams,
        "expected_real": ("calibration.expected_real_adjustment(family) debiases sim "
                          "metrics with the live table (min 3 pairs, else global "
                          "fallback); selection.promotion_eligible(..., expect_real=...) blocks "
                          "promotion unless bias-adjusted Sharpe/Fitness clear cutoffs. "
                          "Turnover uses fitted slope when >=4 pts span >=30pp. "
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
    ctx["tests"] = test_inventory()
    ctx["codebook"] = codebook_snapshot()
    ctx["pipeline"] = (
        "run.py/competition -> providers/data_gen+real_data(+pit) -> "
        "fastexpr eval -> simulator stages (universe/pasteurize, neutralize, "
        "decay, weights/truncate, delay, PnL, metrics, S1 sub-universe, 4Y "
        "self-corr, 6 gates, trial log, risk lens, optional high-fidelity) -> "
        "agents optimize (skeleton/dimension/lesson/parsimony/alignment) -> "
        "competition promotion block (gates AND lineage AND DSR>=0.95 AND "
        "optional StepM AND optional expected-real debias, per-final family) -> "
        "gp evolve (3D Pareto: fitness/nodes/tail_gap) -> "
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
        "Cashflow exact-variant gap (#3 sim +0.64 vs real +1.84; size-tailwind collision diagnosed, liqf untouched).",
        "Sentiment level-fade vs drift-only edge (#5 sign mismatch; reversal leg refused on N=1).",
    ]
    # Session changelog: hand-maintained, one entry per work session. This is
    # the narrative layer machine-readable state cannot reconstruct (why
    # decisions were made, what failed). Append, never rewrite history.
    ctx["session_log"] = [
        {"session": "calibration-bias",
         "did": "Measured sim-vs-Brain bias on 8 pairs (Sharpe +1.81 optimistic); "
                "built expected_real_adjustment + promotion 7th criterion; T2fb demo blocks as Brain did.",
         "outcome": "landed, RR-27"},
        {"session": "codebook-extension",
         "did": "Mapped Brain reference to 39 matrix + 4 vector fields + market group; "
                "identities over PIT parents; vec_mean/vec_std/vec_choose ops; "
                "longest-match family voting (fixed cap/capex, open/opt_open collisions).",
         "outcome": "landed, RR-30; 11 reference gaps closed (10 aliases + subindustry)"},
        {"session": "rng-stream-incident",
         "did": "A subindustry rng.integers() call placed mid-stream silently zeroed "
                "the 2020 panic rate; caught by gate test, fixed with deterministic "
                "mapping, verified bit-identical restoration.",
         "outcome": "landed; rule: no RNG draws before downstream consumers except in declared order"},
        {"session": "family-bias+slope",
         "did": "Per-family bias envelopes (microstructure n=5 live, rest fallback); "
                "turnover slope correction gated on >=4pts/>=30pp span; fixed logged "
                "paren typo in one calibration record with annotation.",
         "outcome": "landed, RR-28/29"},
        {"session": "raw-vs-log+risk-lens",
         "did": "Measured raw-vs-log at all 7 return use-sites: daily PnL exact in "
                "simple returns (log would be wrong); added geometric CAGR reporting "
                "(T2fb 14.70% linear vs 15.53% CAGR); 1+r>0 validator guard. Built "
                "risk.py (hist VaR/ES, normal+t parametrics, conditional series, "
                "exceedance backtest); first live read flagged MISCALIBRATED tails. "
                "Per-alpha nu fitting (4.2-5.3, heavier than Remark-1 6); t-ES gap "
                "predicts stress-DD ordering across 6 alphas.",
         "outcome": "landed, RR-26; 74/74 tests at the time"},
        {"session": "long-tenor-IV+cashflow+alt-sentiment",
         "did": "Modeled 720d IV (sparser subset, term premium, wide skew), CFO-margin "
                "AR(1)+PIT with cfo edge leg (quarterly block moved pre-edge, margin "
                "stream preserved), independent scl12 process sharing drift edge. "
                "Private #6 sim PASS reproduces real PASS (2.0/2.15 vs 1.30/1.07); "
                "#3 null->+0.64 with diagnosed cap-scaler/size-tailwind collision; "
                "#5 sign mismatch refused a reversal leg on N=1.",
         "outcome": "landed, RR-31; 89/89 tests"},
    ]
    # Problems ledger: fixed (with how) vs open (with what's missing).
    # Open RR entries are likewise derivable above; this is the short list.
    ctx["problems"] = {
        "fixed": [
            {"what": "Sentiment sparsity false confidence (47% vs ~4% real)",
             "how": "Recalibrated p_cov gradient + blackouts; T1 combo now fails synthetically as on real."},
            {"what": "Turnover half-factor (2x undercount)",
             "how": "Gross sum|dW| convention; E1 syn 33.14 vs real 33.33."},
            {"what": "Truncation rescale violating cap (50% books)",
             "how": "Plain clip, cap-strict, under-invests; verified at 5/10/20%."},
            {"what": "Stale weight carry on thin days",
             "how": "Flat-zero book; matches all surveyed engines."},
            {"what": "Static lookahead universe mask",
             "how": "Point-in-time trailing-252d sort; 13-18/500 daily misclassification removed."},
            {"what": "Dead water-filling loop (comment claimed behavior code didn't have)",
             "how": "Deleted loop, relabeled plain clip; behavior-identical verified."},
            {"what": "Bootstrap recentering bug (would reject everything)",
             "how": "Recenter by original means; caught by all-noise known-truth test."},
            {"what": "Lexicographic Pareto comparison dropping valid candidates",
             "how": "Elementwise 3D dominance; caught by unit test."},
            {"what": "Dimension gate blind inside *//comparisons/wrappers",
             "how": "Mandatory child recursion; rank(close*(sales-volume)) now flags."},
            {"what": "RNG stream shift zeroing panic gate",
             "how": "Deterministic subindustry mapping; bit-identical restoration verified."},
            {"what": "PIT lookup 184s builds (nested-loop JOIN + row-wise inserts/assembly)",
             "how": "Effective-knowledge single-inequality ASOF + UNNEST batching + vectorized assembly; ~15s builds."},
            {"what": "Turnover churn misdiagnosed as base-process defect",
             "how": "2x2 sweep proved decay-setting controls TO; churn package neutralized, comparisons must be decay-matched."},
        ],
        "open": [e["id"] + ": " + e["claim"] for e in rr["entries"]
                 if e.get("status") == "open"],
    }
    # Standing orders: durable user directives. Do-not-violate list first.
    ctx["standing_orders"] = {
        "do_not": [
            "Do NOT commit unless explicitly asked (say the word).",
            "Do NOT push tokens or secrets into commands, files, or git config; "
            "never ask for tokens — direct user to gh auth login / credential manager.",
            "Do NOT loosen any gate to raise pass rates (CI enforces via GATE_VERSION).",
            "Do NOT upload user-supplied private alphas to GitHub (gitignored private_records.jsonl).",
            "Do NOT add RNG draws before downstream consumers without verifying stream stability.",
            "Do NOT tune simulator constants on thin evidence (min 3 pairs for bias, 4pts/30pp for slopes).",
            "Do NOT let cost/risk/high-fidelity lenses enter gates (diagnostic-only).",
        ],
        "do": [
            "Verify every fix by execution (tests + measured numbers), never by reasoning alone.",
            "Log every test including nulls; falsifiers stated up front with what would disprove.",
            "Keep calibration pairs flowing: record real Brain results with full metrics incl. sub-sharpe + corr.",
            "Update risk_register.json + rebuild context.json on material changes.",
            "Prefer fixing root causes over adding machinery (subtraction before addition).",
        ],
    }
    # Continuation prompt: paste context.json + this prompt into a fresh
    # session to resume as if uninterrupted.
    ctx["continue_prompt"] = (
        "You are continuing the Fastexp project (C:\\Users\\at727\\Downloads\\Fastexp). "
        "The attached context.json is the full project state — trust it over prior "
        "assumptions, but verify cheap claims with tools before acting. Rules: "
        "(1) read the open_tracks and session_log first; (2) never commit/push unless "
        "explicitly asked, and never handle tokens; (3) verify every change by running "
        "code (pytest suite must stay green) and report measured numbers, not reasoning; "
        "(4) log null results and state falsifiers up front; (5) keep private alphas in "
        "gitignored files only; (6) update risk_register.json and rebuild context.json "
        "(python scripts/build_context.py) on material changes. "
        "Start by running git status + pytest to confirm the tree matches context.json, "
        "then state which open track you will work and your falsifiable plan."
    )
    out = os.path.join(ROOT, "context.json")
    with open(out, "w") as f:
        json.dump(ctx, f, indent=2, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
