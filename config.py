"""Versioned, overridable simulation configuration (Build Spec Sec 2 + Sec 9).

Every constant from the spec lives here — never hardcoded in the simulation
core. Users recalibrate APPROX-marked values against real Brain results and
record pairs in calibration/*.json.
"""
import json

CONFIG_VERSION = "1.0.0"

# Core Operational Invariant 1 — GATE INVARIANCE (Pareto-upgrade program).
# No upgrade phase may loosen an existing gate to raise passing rates.
# Enforcement: tests/test_gates_invariant.py pins every threshold below;
# any change requires bumping GATE_VERSION with a written justification,
# and CI fails otherwise. Tightening is allowed; loosening is a defect.
GATE_VERSION = 1
GATE_INVARIANTS = {
    "sharpe_min": 1.25,          # strict >
    "fitness_min": 1.0,          # strict >
    "turnover_min": 0.01,        # inclusive, fraction of book/day
    "turnover_max": 0.70,        # inclusive
    "weight_conc_max": 0.10,     # max|W|/book, inclusive
    "self_corr_max": 0.70,       # |corr| at/above triggers escape check
    "corr_sharpe_improve": 1.10,  # escape: beat EVERY correlated ref by this
    "self_corr_window_years": 4,  # rolling window on daily PnL changes
    # Sub-universe uses the S1 relative formula (not a fixed number):
    #   0.75*sqrt(sub_size/alpha_size)*alpha_sharpe  (strict >)
    # Turnover uses the gross convention: mean(sum|dW|) (NOT half).
}

# Provenance tags preserved from the build spec.
SOURCED = "SOURCED"
APPROX = "APPROX — VALIDATE EMPIRICALLY"

DEFAULT_SETTINGS = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 4,
    "truncation": 0.1,
    "neutralization": "INDUSTRY",
    "pasteurization": "ON",
    "nanHandling": "ON",
    "unitHandling": "VERIFY",
    "language": "FASTEXPR",
    "instrumentType": "EQUITY",
}

# Suggested neutralization defaults per dataset category (Sec 2.5, SOURCED guidance).
NEUTRALIZATION_BY_DATASET = {
    "fundamental": "INDUSTRY",
    "analyst": "INDUSTRY",
    "estimates": "INDUSTRY",
    "model": "EXPERIMENT",
    "news": "SUBINDUSTRY",
    "sentiment": "SUBINDUSTRY",
    "price_volume": "MARKET",
}

CUTOFFS = {
    # [SOURCED] standard single-alpha submission gates; vary by competition/track.
    "sharpe_min": {"value": 1.25, "provenance": SOURCED},
    "fitness_min": {"value": 1.0, "provenance": SOURCED},
    "turnover_min": {"value": 0.01, "provenance": SOURCED},
    "turnover_max": {"value": 0.70, "provenance": SOURCED},
    "weight_conc_max": {"value": 0.10, "provenance": SOURCED},
    "self_corr_max": {"value": 0.70, "provenance": SOURCED},
    "corr_sharpe_improve": {"value": 1.10, "provenance": SOURCED},
    # Sub-universe gate: S1 official-doc formula (audit 2026-09-06, item 1).
    # sub_cutoff = 0.75*sqrt(sub_size/alpha_universe_size)*alpha_sharpe
    # (worked example: 0.75*sqrt(1000/3000)*2.73 = 1.18). Scales with candidate
    # Sharpe, so stricter on inflated synths. Supersedes the delay-0-only
    # absolute variant and the unsourced 0.80 fallback (kept as dict entries
    # for back-compat; explicit override still wins if provided).
    "subuniverse_sharpe_min": {"value": 0.80, "provenance": APPROX + " — SUPERSEDED by S1 relative formula; kept as explicit-override escape hatch only"},
    "subuniverse_formula": {
        "value": "0.75*sqrt(sub/alpha)*alpha_sharpe",
        "provenance": APPROX + " — S1 official Brain doc 'Clear these tests before submitting an Alpha' + worked example; corroborated by dafu-zhu/alpha-lab",
    },
    "fitness_turnover_floor": {"value": 0.125, "provenance": SOURCED},
    # Phase 3 cost lens (reporting only, never a gate): linear slippage bps.
    "cost_bps": {"value": 10.0, "provenance": "local reporting assumption (not a Brain gate)"},
    "annualization": {"value": 252, "provenance": SOURCED},
    "largest_universe_size": {"value": 3000, "provenance": APPROX},
    # Audit 2026-09-06 (item 8): 4Y per S1 official-doc scrape + DeepWiki +
    # skill README; old 2Y traced to 2023 seminar notes (superseded). 2Y still
    # reported as transition info (see simulate() self_corr_2y fields).
    "self_corr_window_years": {"value": 4, "provenance": APPROX + " — S1 4Y window; was 2Y (Khdeng/jglazar 2023 notes, superseded)"},
    "book_size": {"value": 1.0, "provenance": SOURCED},
}

# Universe tiers as liquidity-ranked slices (Sec 2.1).
# [APPROX] membership rule (audit 2026-09-06, item 13): point-in-time trailing
# 252d mean adv20 (or dollar volume), re-sorted daily — no lookahead. Still an
# approximation of real dated index constituents. Override with real index
# membership if available.
UNIVERSES = {
    "TOP3000": {"size": 3000, "provenance": APPROX},
    "TOP1000": {"size": 1000, "provenance": APPROX},
    "TOP500": {"size": 500, "provenance": APPROX},
    "TOP200": {"size": 200, "provenance": APPROX},
}

SUPPORTED_REGIONS = ["USA", "ASI", "EUR", "CHN", "KOR", "TWN", "HKG", "JPN",
                     "AMS", "GLB"]


# Phase 2 dual-mode execution (Agent 1 blueprint). Fast Gate keeps 100% of
# the Brain-proxy semantics; High-Fidelity adds fills + Almgren-Chriss-style
# impact + variable spread as a DIAGNOSTIC-ONLY lens (never gates) until
# calibrated against real implementation shortfall. Impact/latency defaults
# are APPROX by construction.
DEFAULT_EXECUTION = {
    "mode": "fast_gate",       # fast_gate | high_fidelity
    "spread_bps": 5.0,         # one-way half-spread baseline, urgency-scaled
    "urgency": 1.0,            # 0..1 VWAP-to-aggressive dial on spread cost
    "lambda_perm": 1.0,        # permanent impact scale (sqrt-law coefficient)
    "eta_temp": 1.0,           # temporary impact scale (linear-in-rate)
    "alpha": 0.5,              # permanent-law exponent (sqrt law; Gatheral-safe)
    "max_participation": 0.10,  # FIFO back-of-queue proxy: fill at most this
                               # fraction of daily dollar liquidity per name
    "commission_bps": 1.0,     # fixed per-side commission
}


class ExecutionSettings:
    """Execution-layer schema for High-Fidelity mode (Phase 2)."""

    FIELDS = ("mode", "spread_bps", "urgency", "lambda_perm", "eta_temp",
              "alpha", "max_participation", "commission_bps")

    def __init__(self, **kw):
        cfg = dict(DEFAULT_EXECUTION)
        cfg.update(kw)
        for k in self.FIELDS:
            setattr(self, k, cfg[k])
        self.validate()

    def validate(self):
        if str(self.mode) not in ("fast_gate", "high_fidelity"):
            raise ValueError(f"mode must be fast_gate|high_fidelity, got {self.mode!r}")
        for k in ("spread_bps", "lambda_perm", "eta_temp", "commission_bps"):
            if not float(getattr(self, k)) >= 0:
                raise ValueError(f"{k} must be >= 0")
        if not 0.0 <= float(self.urgency) <= 1.0:
            raise ValueError("urgency must be in [0,1]")
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError("alpha must be in (0,1]")
        if not 0.0 < float(self.max_participation) <= 1.0:
            raise ValueError("max_participation must be in (0,1]")

    def to_dict(self):
        return {k: getattr(self, k) for k in self.FIELDS}


class SimulationSettings:
    """First-class config schema: every simulate() call takes one (Sec 2)."""

    FIELDS = ("region", "universe", "delay", "decay", "truncation",
              "neutralization", "pasteurization", "nanHandling",
              "unitHandling", "language", "instrumentType")

    def __init__(self, **kw):
        cfg = dict(DEFAULT_SETTINGS)
        cfg.update(kw)
        for k in self.FIELDS:
            setattr(self, k, cfg[k])
        self.validate()

    def validate(self):
        if self.delay not in (0, 1):
            raise ValueError(f"delay must be 0 or 1, got {self.delay!r}")
        if not isinstance(self.decay, int) or self.decay < 0:
            raise ValueError(f"decay must be a non-negative int, got {self.decay!r}")
        if not (0 <= self.truncation <= 1):
            raise ValueError(f"truncation must be in [0,1], got {self.truncation!r}")
        neut = str(self.neutralization).upper()
        if neut not in ("NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY",
                        "SUB-INDUSTRY", "COUNTRY", "EXCHANGE"):
            raise ValueError(f"unknown neutralization {self.neutralization!r}")
        if str(self.pasteurization).upper() not in ("ON", "OFF"):
            raise ValueError("pasteurization must be ON/OFF")
        if str(self.nanHandling).upper() not in ("ON", "OFF"):
            raise ValueError("nanHandling must be ON/OFF")
        if str(self.unitHandling).upper() not in ("VERIFY", "OFF"):
            raise ValueError("unitHandling must be VERIFY/OFF")

    def to_dict(self):
        return {k: getattr(self, k) for k in self.FIELDS}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.FIELDS})

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def get_cutoff(name, overrides=None):
    ov = (overrides or {}).get(name, None)
    if ov is not None:
        return float(ov)
    return float(CUTOFFS[name]["value"])


def subuniverse_cutoff(sub_size, alpha_size=None, alpha_sharpe=None,
                       largest_size=None, delay=1, overrides=None):
    """S1 relative formula (audit 2026-09-06, item 1).

    sub_cutoff = 0.75 * sqrt(sub_size / alpha_universe_size) * alpha_sharpe.
    Explicit `subuniverse_sharpe_min` override still wins if provided
    (back-compat escape hatch). Legacy kwargs (largest_size/delay) accepted
    but ignored apart from the override path.
    """
    ov = (overrides or {}).get("subuniverse_sharpe_min", None)
    if ov is not None:
        return float(ov)
    if alpha_size and alpha_sharpe is not None:
        import math
        return 0.75 * math.sqrt(max(sub_size, 0) / max(alpha_size, 1)) * float(alpha_sharpe)
    return float(CUTOFFS["subuniverse_sharpe_min"]["value"])
