"""Living calibration log helpers (Build Spec Sec 9).

Paste real Brain submission results alongside simulator predictions so
APPROX-marked constants (delay mechanics, sub-universe formula, coverage
assumptions, returns denominator) tighten over time.
"""
import json
import os

CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibration")
RECORDS = os.path.join(CALIB_DIR, "records.jsonl")


def record(expression, settings, real_metrics, predicted_metrics, notes=""):
    if hasattr(settings, "to_dict"):
        settings = settings.to_dict()
    entry = {"expression": expression, "settings": settings,
             "real_metrics": real_metrics, "predicted_metrics": predicted_metrics,
             "notes": notes}
    os.makedirs(CALIB_DIR, exist_ok=True)
    with open(RECORDS, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def load_records():
    out = []
    if not os.path.exists(RECORDS):
        return out
    with open(RECORDS) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize():
    """Mean predicted-vs-real deltas per metric over records with both sides."""
    import numpy as np
    recs = [r for r in load_records()
            if r.get("real_metrics") and r.get("predicted_metrics")]
    if not recs:
        return {"n": 0, "hint": "no records with real_metrics yet"}
    keys = set()
    for r in recs:
        keys |= set(r["real_metrics"]) & set(r["predicted_metrics"])
    return {"n": len(recs),
            **{k: round(float(np.mean([r["predicted_metrics"][k] - r["real_metrics"][k]
                                        for r in recs if k in r["real_metrics"]
                                        and k in r["predicted_metrics"]])), 4)
               for k in sorted(keys)}}


def expected_real_adjustment(family=None, min_records=3):
    """Bias table for the expected-real lens: mean (predicted - real) deltas.

    With family=None uses all paired records (legacy global table). With a
    family name, filters to that family's pairs and falls back to the global
    table when the family has fewer than min_records pairs — same thin-data
    discipline, one level down. Returns an envelope:
    {"bias": {...}, "family": resolved_name, "n": n_used, "fallback": bool,
     "to_slope": {...}|None}. Empty bias {} when even the global pool is thin.
    """
    s = summarize()
    if s.get("n", 0) < min_records:
        return {"bias": {}, "family": family or "global", "n": 0,
                "fallback": True, "to_slope": None}
    if family is None:
        pool = [r for r in load_records()
                if r.get("real_metrics") and r.get("predicted_metrics")]
        fam_name, fallback = "global", False
    else:
        pool = [r for r in load_records()
                if r.get("real_metrics") and r.get("predicted_metrics")
                and family_of_expression(r.get("expression", "")) == family]
        if len(pool) >= min_records:
            fam_name, fallback = family, False
        else:
            pool = [r for r in load_records()
                    if r.get("real_metrics") and r.get("predicted_metrics")]
            fam_name, fallback = "global", True
    import numpy as _np
    keys = set()
    for r in pool:
        keys |= set(r["real_metrics"]) & set(r["predicted_metrics"])
    bias = {k: round(float(_np.mean([r["predicted_metrics"][k] - r["real_metrics"][k]
                                     for r in pool if k in r["real_metrics"]
                                     and k in r["predicted_metrics"]])), 4)
            for k in sorted(keys)}
    return {"bias": bias, "family": fam_name, "n": len(pool),
            "fallback": fallback, "to_slope": turnover_slope(pool)}


def expected_real_metrics(sim_metrics, adjustment):
    """Apply bias to simulated metrics. Accepts a flat bias dict (legacy) or
    a full envelope from expected_real_adjustment (uses ["bias"] plus the
    turnover slope when present).

    Sharpe/Fitness/sub-sharpe shift by the optimism bias; turnover and
    drawdown shift by the underprediction bias (bias is predicted-minus-real,
    so turnover_real ~= turnover_sim - bias). With a fitted slope, turnover
    uses TO_real ~= TO_sim - (a*TO_sim + b) instead of the additive fallback.
    Keys missing from bias pass through unchanged. Pure function, no I/O.
    """
    bias = adjustment.get("bias", adjustment) if isinstance(adjustment, dict) else {}
    slope = adjustment.get("to_slope") if isinstance(adjustment, dict) else None
    out = dict(sim_metrics)
    for k in ("sharpe", "fitness", "subuniverse_sharpe"):
        if k in out and k in bias:
            out[k] = round(out[k] - bias[k], 4)
    for k in ("drawdown_pct",):
        if k in out and k in bias:
            out[k] = round(out[k] - bias[k], 4)
    if "turnover_pct" in out:
        if slope:
            out["turnover_pct"] = round(
                out["turnover_pct"] - (slope["slope"] * out["turnover_pct"]
                                       + slope["intercept"]), 4)
        elif "turnover_pct" in bias:
            out["turnover_pct"] = round(out["turnover_pct"] - bias["turnover_pct"], 4)
    return out


def load_unsubmitted_pairs():
    """Pool-format turnover pairs from the gitignored unsubmitted batch.

    Reads the latest calibration/unsubmitted_sim_*.jsonl (index-level sim
    metrics + real TO, no expressions leave the file). Returns [] when no
    batch file exists (callers fall back to submitted-only).
    """
    import glob as _glob
    import json as _json
    import os as _os
    files = sorted(_glob.glob(os.path.join(CALIB_DIR, "unsubmitted_sim_*.jsonl")),
                   key=_os.path.getmtime)
    if not files:
        return []
    out = []
    with open(files[-1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _json.loads(line)
            if (r.get("status") == "OK"
                    and isinstance(r.get("sim_TO"), (int, float))
                    and isinstance(r.get("real_TO"), (int, float))):
                out.append({"predicted_metrics": {"turnover_pct": r["sim_TO"]},
                            "real_metrics": {"turnover_pct": r["real_TO"]}})
    return out


def turnover_slope(pool=None, min_points=4, min_span=30.0, scope="pooled"):
    """Linear turnover-bias model: delta (= pred - real) ~= a*TO_sim + b.

    The additive correction is the wrong shape (miss grows with turnover:
    -0.3 at 60% vs -22.6 at 69%). Returns {"slope","intercept","n","span"}
    or None when fewer than min_points pairs or sim-TO span below min_span
    pp — one point doesn't make a slope. Closed-form OLS, no fitting library.

    scope="submitted" uses the 9 tracked pairs only (legacy RR-29 fit:
    slope +0.45 — an n=8 artifact). scope="pooled" (default) adds the
    gitignored 40-pair unsubmitted batch: slope collapses to ~+0.10,
    i.e. the bias is essentially additive (~-12pp) with no meaningful
    turnover dependence. Falls back to submitted-only when no batch file.
    """
    import numpy as _np
    if pool is None:
        pool = [r for r in load_records()
                if r.get("real_metrics") and r.get("predicted_metrics")]
        if scope == "pooled":
            pool = pool + load_unsubmitted_pairs()
    pts = [(r["predicted_metrics"]["turnover_pct"], r["real_metrics"]["turnover_pct"])
           for r in pool
           if isinstance(r["predicted_metrics"].get("turnover_pct"), (int, float))
           and isinstance(r["real_metrics"].get("turnover_pct"), (int, float))]
    if len(pts) < min_points:
        return None
    xs = _np.array([p[0] for p in pts])
    if float(xs.max() - xs.min()) < min_span:
        return None
    deltas = _np.array([p[0] - p[1] for p in pts])
    a, b = (float(v) for v in _np.polyfit(xs, deltas, 1))
    return {"slope": round(a, 4), "intercept": round(b, 4),
            "n": len(pts), "span": round(float(xs.max() - xs.min()), 2)}


_FAMILY_TOKENS = {
    "sentiment": ("nws", "snt", "buzz", "news"),
    "fundamental": ("ebitda", "sales", "debt", "assets", "margin", "lev", "est",
                    "cogs", "gross_profit", "income", "eps", "equity",
                    "cash", "retained", "goodwill", "working_capital",
                    "capex", "dividends", "tax", "revenue", "fcf",
                    "ocf", "analyst", "recommendation", "surprise", "segment",
                    "ni", "total_debt"),
    "microstructure": ("returns", "close", "open", "high", "low", "volume", "adv20", "cap",
                       "vwap", "shares_out", "short_interest", "days_to_cover",
                       "borrow_fee", "insider", "intraday"),
    "derivatives": ("implied_volatility", "iv_", "hv_", "historical", "put_call", "pcr",
                    "opt_open", "skew", "option"),
}


def family_of_expression(expr):
    """Majority-vote data family from field tokens (ties -> mixed).

    Longest-match voting per field (see agents._leg_family): prevents
    substring collisions ('cap' ⊂ 'capex', 'open' ⊂ 'opt_open_interest').
    """
    from fastexpr import parse, iter_nodes, Field, WQError
    try:
        names = {n.name for n in iter_nodes(parse(expr)) if isinstance(n, Field)}
    except WQError:
        return "unknown"
    scores = {}
    for nm in names:
        best, bestlen = [], -1
        for fam, toks in _FAMILY_TOKENS.items():
            ml = max((len(t) for t in toks if t in nm), default=-1)
            if ml > bestlen:
                best, bestlen = [fam], ml
            elif ml == bestlen and ml >= 0:
                best.append(fam)
        for fam in best:
            scores[fam] = scores.get(fam, 0) + 1
    if not scores:
        return "unknown"
    top = max(scores.values())
    winners = [f for f, s in scores.items() if s == top]
    return winners[0] if len(winners) == 1 else "mixed"
