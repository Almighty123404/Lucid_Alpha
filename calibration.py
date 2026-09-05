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
