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
from itertools import combinations

import numpy as np

TRIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'reports', 'trials.jsonl')
AUTO_LOG = True  # simulate() logs every trial; tests set False or redirect
SCOPE = "default"  # research-question namespace for N_trials multiplicity.
# Multiplicity is only meaningful within one search: campaign trials must not
# tax competition candidates or vice versa. Entry points set this once
# (competition -> "competition", gp_campaign -> "gp-campaign", walkforward ->
# "walkforward"); count()/promotion_eligible() filter on it. None = all scopes
# (legacy/audit reads only, never promotion).
DSR_PROMOTE = 0.95  # Agent 4 threshold: DSR>=0.95 promotes, 0.5 = coin flip


def set_trials_path(path):
    global TRIALS_PATH
    TRIALS_PATH = path


def set_scope(scope):
    global SCOPE
    SCOPE = scope or "default"


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

    def log(self, skeleton, dataset_id, settings, sharpe, fitness, passed,
            scope=None):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({"skeleton": skeleton, "dataset_id": dataset_id,
                                    "settings": settings, "sharpe": sharpe,
                                    "fitness": fitness, "passed": bool(passed),
                                    "scope": scope or SCOPE},
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

    def count(self, dataset_id=None, scope="default"):
        """Trial count, optionally filtered. scope=None counts ALL scopes
        (audit/legacy reads); promotion always passes an explicit scope."""
        n = 0
        for e in self._iter():
            if dataset_id is not None and e.get("dataset_id") != dataset_id:
                continue
            if scope is not None and e.get("scope", "default") != scope:
                continue
            n += 1
        return n

    def has_dataset(self, dataset_id, scope="default"):
        if not dataset_id:
            return False
        return self.count(dataset_id, scope) > 0


def promotion_eligible(rep, registry, dstar=DSR_PROMOTE, scope="default",
                       peer_pnls=None, alpha_fw=0.05, expect_real=None):
    """Promotion block. Returns (eligible: bool, info: dict).

    eligible requires ALL of: non-empty dataset_id on the report, >=1 logged
    trial for that dataset IN THE GIVEN SCOPE (the N_trials entry), gates
    passed, DSR >= dstar (computed with scoped N). Missing lineage -> DSR
    reported as 0.0 and eligible False, always. scope=None counts all scopes
    and is for audit reads only, never promotion.

    peer_pnls (optional): list of peer daily-PnL series for the Romano-Wolf
    portfolio screen. When provided, the candidate must ALSO survive StepM
    FWER control at alpha_fw against the peers (pass = candidate in the
    rejection set); PBO is attached as info only (uninformative below ~10
    models, never blocks alone). Peers should be the strongest alternatives
    (e.g. top-3 refs by Sharpe) so the adjustment prices real competition.

    expect_real (optional): bias table from calibration.expected_real_adjustment.
    When provided, the candidate must ALSO pass Sharpe/Fitness cutoffs on
    bias-adjusted (expected-real) metrics. This is a TIGHTENING (a 7th
    promotion criterion, allowed under gate invariance): with measured
    optimism near +1.8 Sharpe it blocks simulator-passes Brain would fail —
    e.g. real T2fb failed fitness while sim passed it. None (default)
    preserves legacy behavior exactly; callers opt in explicitly.
    """
    dsid = rep.get("dataset_id", "")
    if not dsid:
        return False, {"dsr": 0.0, "reasons": ["missing dataset_id: DSR=0, blocked"]}
    n = registry.count(dsid, scope)
    if n < 1:
        return False, {"dsr": 0.0, "reasons": [f"no logged N_trials entry for dataset {dsid[:8]} (scope={scope}): DSR=0, blocked"]}
    pnl = rep.get("pnl", [])
    dsr = deflated_sharpe_ratio(list(pnl), n)
    reasons = []
    if rep.get("error"):
        reasons.append(f"simulator error: {rep['error']}")
    if not rep.get("passed"):
        from simulator import failed_criteria as _fc
        reasons.append(f"gates failed: {_fc(rep)}")
    if dsr < dstar:
        reasons.append(f"DSR {dsr} < {dstar} (N={n}, scope={scope})")
    info = {"dsr": dsr, "n_trials": n, "scope": scope, "dataset_id": dsid[:12], "reasons": reasons}
    if expect_real:
        from calibration import expected_real_metrics as _erm
        from config import get_cutoff as _gc
        adj = _erm(rep.get("metrics", {}), expect_real)
        info["expected_real"] = {k: adj.get(k) for k in
                                 ("sharpe", "fitness", "turnover_pct", "drawdown_pct")}
        info["expected_real_family"] = (expect_real.get("family", "global")
                                        if isinstance(expect_real, dict) else "global")
        bias_used = (expect_real.get("bias", expect_real)
                     if isinstance(expect_real, dict) else expect_real)
        if not bias_used:
            info["expected_real"]["note"] = "no bias data: check vacuous, legacy verdict stands"
        elif not (adj.get("sharpe", -99) > _gc("sharpe_min")
                  and adj.get("fitness", -99) > _gc("fitness_min")):
            reasons.append(f"expected-real check failed: adj Sharpe {adj.get('sharpe')} / "
                           f"Fitness {adj.get('fitness')} below cutoffs")
    if peer_pnls:
        ok, detail = portfolio_screen(list(pnl), [list(p) for p in peer_pnls],
                                      alpha_fw=alpha_fw)
        info["stepm"] = detail
        if not ok:
            reasons.append(f"StepM FWER screen failed at {alpha_fw} "
                           f"(candidate adj-p {detail['candidate_adj_p']}, "
                           f"{detail['n_models']} models)")
    return (len(reasons) == 0), info


# ---------------------------------------------------------------------------
# Portfolio-level false-discovery control (RR-14 next_check). DSR is
# per-alpha; as mining scales to hundreds of trials, promotion additionally
# requires surviving a Romano-Wolf StepM screen against peer return streams.
# PBO is reported as a diagnostic (informative only at >=10 models).
# ---------------------------------------------------------------------------

def _stationary_bootstrap_index(T, B, q, rng):
    """Politis-Romano stationary bootstrap: BxT index array, mean block 1/q."""
    idx = np.zeros((B, T), dtype=np.int64)
    for b in range(B):
        t = 0
        start = int(rng.integers(T))
        while t < T:
            L = 1 + int(rng.geometric(q))
            for k in range(L):
                if t >= T:
                    break
                idx[b, t] = (start + k) % T
                t += 1
            start = int(rng.integers(T))
    return idx


def _tstat(d):
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return 0.0
    sd = d.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(d.mean() / (sd / math.sqrt(n)))


def white_reality_check(candidate, benchmark, B=500, q=0.1, seed=0):
    """White (2000) Reality Check: does the candidate beat the benchmark,
    accounting for having looked? Single-hypothesis bootstrap max-t test.
    Returns (reject: bool, p_value)."""
    cand = np.asarray(candidate[1:], dtype=np.float64)
    bench = np.asarray(benchmark[1:], dtype=np.float64)
    n = min(len(cand), len(bench))
    if n < 30:
        return False, 1.0
    d = cand[-n:] - bench[-n:]
    t_obs = _tstat(d)
    if t_obs <= 0:
        return False, 1.0
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_index(n, B, q, rng)
    # Recenter by the ORIGINAL mean (H0: mu = 0 in bootstrap world). Subtracting
    # the bootstrap sample's own mean instead would force every bootstrap
    # t-stat to exactly 0 and reject everything — caught by unit test.
    db = d[idx] - d.mean()
    t_boot = np.array([_tstat(db[b]) for b in range(B)])
    p = float((t_boot >= t_obs).mean())
    return bool(p < 0.05), round(p, 4)


def romano_wolf_stepm(model_pnls, benchmark_pnl=None, alpha=0.05, B=500,
                      q=0.1, seed=0):
    """Romano-Wolf (2005) StepM: which models beat the benchmark with FWER
    control at `alpha`? model_pnls: list of daily series; benchmark defaults
    to zeros (i.e., any positive edge). Returns (rejected_idx, adj_p list)
    with stepdown-adjusted p-values (monotone, sharper than Holm)."""
    Ms = [np.asarray(m[1:], dtype=np.float64) for m in model_pnls]
    n = min([len(m) for m in Ms])
    if benchmark_pnl is None:
        bench = np.zeros(n)
    else:
        bench = np.asarray(benchmark_pnl[1:], dtype=np.float64)[-n:]
    Ms = [m[-n:] for m in Ms]
    K = len(Ms)
    if n < 30 or K < 1:
        return [], [1.0] * K
    D = np.array([m - bench for m in Ms])
    t_obs = np.array([_tstat(D[k]) for k in range(K)])
    order = list(np.argsort(-t_obs))
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_index(n, B, q, rng)
    adj = [1.0] * K
    remaining = list(order)
    step_p = 0.0
    rej = []
    while remaining:
        # bootstrap max-t over remaining models, recentered by ORIGINAL means
        # (least-favorable null; same caveat as White above).
        boot_max = np.empty(B)
        R = np.array(remaining)
        orig_mean = D[R].mean(axis=1, keepdims=True)
        for b in range(B):
            samp = D[R][:, idx[b]] - orig_mean
            vals = np.array([_tstat(samp[j]) for j in range(len(R))])
            boot_max[b] = vals.max() if len(vals) else -np.inf
        k0 = remaining[0]
        p_k = float((boot_max >= t_obs[k0]).mean())
        step_p = max(step_p, p_k)  # stepdown monotonicity
        adj[k0] = round(min(step_p, 1.0), 4)
        if step_p < alpha:
            rej.append(k0)
            remaining = remaining[1:]
        else:
            for k in remaining[1:]:
                adj[k] = round(min(max(step_p, float((boot_max >= t_obs[k]).mean())), 1.0), 4)
            break
    return rej, adj


def pbo_diagnostic(model_pnls, S=8):
    """Probability of Backtest Overfitting (Bailey et al.) via CSCV.

    Partition time into S blocks; all C(S, S/2) IS/OOS splits; PBO = fraction
    where the IS-optimal model ranks below the OOS median. Returns
    {"pbo": x, "n_splits": m} or {"pbo": None} when too few models/splits
    to be informative (<10 models or <10 splits) — diagnostic only, never
    a promotion block on its own.
    """
    Ms = [np.asarray(m[1:], dtype=np.float64) for m in model_pnls]
    K = len(Ms)
    if K < 2:
        return {"pbo": None, "reason": "need >= 2 models"}
    n = min(len(m) for m in Ms)
    Ms = [m[-n:] for m in Ms]
    blocks = np.array_split(np.arange(n), S)
    blocks = [b for b in blocks if len(b) >= 10]
    if len(blocks) < 4:
        return {"pbo": None, "reason": "splits too short"}
    Sb = len(blocks)
    splits = list(combinations(range(Sb), Sb // 2))
    if len(splits) < 10:
        return {"pbo": None, "reason": "fewer than 10 splits"}
    below = 0
    for iso in splits:
        oos = [b for b in range(Sb) if b not in iso]
        is_idx = np.concatenate([blocks[b] for b in iso])
        oo_idx = np.concatenate([blocks[b] for b in oos])
        is_sr = [float(Ms[k][is_idx].mean() / max(Ms[k][is_idx].std(ddof=1), 1e-12)) for k in range(K)]
        oo_sr = [float(Ms[k][oo_idx].mean() / max(Ms[k][oo_idx].std(ddof=1), 1e-12)) for k in range(K)]
        best = int(np.argmax(is_sr))
        med = float(np.median(oo_sr))
        if oo_sr[best] < med:
            below += 1
    return {"pbo": round(below / len(splits), 4), "n_splits": len(splits)}


def portfolio_screen(candidate_pnl, peer_pnls, alpha_fw=0.05, B=500,
                     benchmark_pnl=None):
    """Promotion-time portfolio gate: candidate + peers vs benchmark with
    Romano-Wolf FWER control. Returns (pass: bool, detail: dict).

    pass requires the candidate (index 0) to be in the StepM rejection set
    at `alpha_fw`. Peers should be the strongest alternatives (e.g. top-5
    refs by Sharpe) so the adjustment prices real competition, not strawmen.
    PBO is attached as info (informative only below ~10 models).
    """
    models = [candidate_pnl] + list(peer_pnls)
    rej, adj = romano_wolf_stepm(models, benchmark_pnl, alpha_fw, B)
    pbo = pbo_diagnostic(models)
    ok = 0 in rej
    return ok, {"candidate_adj_p": adj[0] if adj else 1.0,
                "n_models": len(models), "alpha_fw": alpha_fw,
                "rejected_idx": rej, "pbo": pbo}
