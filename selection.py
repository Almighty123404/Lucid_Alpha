"""Core Operational Invariant 2 — STRICT LINEAGE LOGGING (Pareto-upgrade program).

Rule: any alpha evaluated WITHOUT an associated dataset_id AND a logged
N_trials entry is automatically assigned DSR = 0 and blocked from promotion.

Pieces:
- TrialRegistry: append-only JSONL log of every simulate() trial
  {skeleton, dataset_id, settings, sharpe, fitness, passed}. N_trials for a
  dataset = its logged trial count (all variants count — understated N
  inflates DSR, the #1 abuse mode per Agent 4).
- deflated_sharpe_ratio(): Bailey & Lopez de Prado (2014) closed form on
  daily PnL changes + trial count. Approximate (independent-normal trials);
  lens-grade, matching the walk-forward haircut philosophy.
- promotion_eligible(): the promotion block. Returns (eligible, reasons).
"""
import json
import math
import os

TRIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'reports', 'trials.jsonl')
AUTO_LOG = True  # simulate() logs every trial; tests set False or redirect
DSR_PROMOTE = 0.95  # Agent 4 threshold: DSR>=0.95 promotes, 0.5 = coin flip


def set_trials_path(path):
    global TRIALS_PATH
    TRIALS_PATH = path


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p):
    """Inverse standard normal CDF by bisection (no scipy dependency)."""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def deflated_sharpe_ratio(pnl_daily, n_trials, annualization=252.0):
    """Bailey-LP Deflated Sharpe Ratio in [0, 1].

    pnl_daily: day-t profits (NOT cumulative). n_trials: logged trial count
    on the same dataset. Uses sample skew/kurtosis and T in observations.
    Returns 0.0 for degenerate inputs (constant series, T < 30, N < 2 records
    no evidence — note N<2 alone does not zero it; missing LINEAGE does).
    """
    x = [float(v) for v in pnl_daily[1:] if v == v]
    T = len(x)
    if T < 30 or n_trials < 1:
        return 0.0
    mu = sum(x) / T
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / max(T - 1, 1))
    if sd < 1e-12:
        return 0.0
    sr = mu / sd * math.sqrt(annualization)
    skew = (sum((v - mu) ** 3 for v in x) / T) / (sd ** 3)
    kurt = (sum((v - mu) ** 4 for v in x) / T) / (sd ** 4)
    N = max(n_trials, 2)
    gamma = 0.5772156649
    q1 = _phi_inv(1.0 - 1.0 / N)
    qe = _phi_inv(1.0 - 1.0 / (N * math.e))
    sr0 = math.sqrt(1.0) * ((1.0 - gamma) * q1 + gamma * qe)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    dsr = _phi((sr - sr0) * math.sqrt(max(T - 1, 1)) / denom)
    return round(min(max(dsr, 0.0), 1.0), 4)


class TrialRegistry:
    """Append-only trial log backing N_trials and the promotion block."""

    def __init__(self, path=None):
        self.path = path or TRIALS_PATH

    def log(self, skeleton, dataset_id, settings, sharpe, fitness, passed):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({"skeleton": skeleton, "dataset_id": dataset_id,
                                    "settings": settings, "sharpe": sharpe,
                                    "fitness": fitness, "passed": bool(passed)},
                                   default=str) + "\n")
        except OSError:
            pass

    def _iter(self):
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except ValueError:
                            continue
        except OSError:
            return

    def count(self, dataset_id=None):
        n = 0
        for e in self._iter():
            if dataset_id is None or e.get("dataset_id") == dataset_id:
                n += 1
        return n

    def has_dataset(self, dataset_id):
        if not dataset_id:
            return False
        return self.count(dataset_id) > 0


def promotion_eligible(rep, registry, dstar=DSR_PROMOTE):
    """Promotion block. Returns (eligible: bool, info: dict).

    eligible requires ALL of: non-empty dataset_id on the report, >=1 logged
    trial for that dataset (the N_trials entry), gates passed, DSR >= dstar.
    Missing lineage -> DSR reported as 0.0 and eligible False, always.
    """
    dsid = rep.get("dataset_id", "")
    if not dsid:
        return False, {"dsr": 0.0, "reasons": ["missing dataset_id: DSR=0, blocked"]}
    n = registry.count(dsid)
    if n < 1:
        return False, {"dsr": 0.0, "reasons": [f"no logged N_trials entry for dataset {dsid[:8]}: DSR=0, blocked"]}
    pnl = rep.get("pnl", [])
    dsr = deflated_sharpe_ratio(list(pnl), n)
    reasons = []
    if rep.get("error"):
        reasons.append(f"simulator error: {rep['error']}")
    if not rep.get("passed"):
        from simulator import failed_criteria as _fc
        reasons.append(f"gates failed: {_fc(rep)}")
    if dsr < dstar:
        reasons.append(f"DSR {dsr} < {dstar} (N={n})")
    info = {"dsr": dsr, "n_trials": n, "dataset_id": dsid[:12], "reasons": reasons}
    return (len(reasons) == 0), info
