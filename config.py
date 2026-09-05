"""Versioned, overridable simulation configuration (Build Spec Sec 2 + Sec 9).

Every constant from the spec lives here — never hardcoded in the simulation
core. Users recalibrate APPROX-marked values against real Brain results and
record pairs in calibration/*.json.
"""
import json

CONFIG_VERSION = "1.0.0"

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
    # [APPROX] sub-universe Sharpe: fixed fallback + documented delay-0 formula.
    # formula: sqrt(252) * max(0.065, (sub_size / largest_size) * 0.25)
    "subuniverse_sharpe_min": {"value": 0.80, "provenance": APPROX},
    "subuniverse_formula": {
        "value": "sqrt252*max(0.065,(sub/largest)*0.25)",
        "provenance": APPROX,
        "delay0_only": True,
    },
    "fitness_turnover_floor": {"value": 0.125, "provenance": SOURCED},
    "annualization": {"value": 252, "provenance": SOURCED},
    "largest_universe_size": {"value": 3000, "provenance": APPROX},
    "self_corr_window_years": {"value": 2, "provenance": SOURCED},
    "book_size": {"value": 1.0, "provenance": SOURCED},
}

# Universe tiers as liquidity-ranked slices (Sec 2.1).
# [APPROX] membership/rebalance rule: top N by trailing avg dollar volume or
# market cap, reconstituted periodically. Implemented as a pluggable ranking
# function; override with real index membership if available.
UNIVERSES = {
    "TOP3000": {"size": 3000, "provenance": APPROX},
    "TOP1000": {"size": 1000, "provenance": APPROX},
    "TOP500": {"size": 500, "provenance": APPROX},
    "TOP200": {"size": 200, "provenance": APPROX},
}

SUPPORTED_REGIONS = ["USA", "ASI", "EUR", "CHN", "KOR", "TWN", "HKG", "JPN",
                     "AMS", "GLB"]


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


def subuniverse_cutoff(sub_size, largest_size=None, delay=1, overrides=None):
    """Documented delay-0 formula; configurable fallback otherwise (Sec 7)."""
    ov = (overrides or {}).get("subuniverse_sharpe_min", None)
    if ov is not None:
        return float(ov)
    if delay == 0:
        import math
        largest = largest_size or CUTOFFS["largest_universe_size"]["value"]
        return math.sqrt(252) * max(0.065, (sub_size / largest) * 0.25)
    return float(CUTOFFS["subuniverse_sharpe_min"]["value"])
