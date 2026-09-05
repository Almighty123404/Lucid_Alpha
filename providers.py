"""Pluggable data providers (Build Spec Sec 9).

The synthetic generator is one swappable backend behind this interface —
plug in real historical data (exports, licensed data, logged Brain results)
without touching the simulation core.
"""
import numpy as np


class DataProvider:
    """Common interface every backend implements."""
    name = "base"

    def get_panel(self, **kw):
        raise NotImplementedError

    def describe_coverage(self, panel):
        """Per-field finite-coverage fractions (Sec 3.3 realism audit)."""
        out = {}
        for k, v in panel.fields.items():
            out[k] = round(float(np.isfinite(v).mean()), 4)
        for k, parts in panel.vector_fields.items():
            out[f"vector:{k}"] = round(float(np.isfinite(parts[0]).mean()), 4)
        return out


class SyntheticProvider(DataProvider):
    name = "synthetic"

    def __init__(self, seed=7, n_stocks=1000, start='2020-01-01', end='2022-12-31'):
        self.seed, self.n_stocks, self.start, self.end = seed, n_stocks, start, end

    def get_panel(self, **kw):
        from data_gen import generate
        args = dict(seed=self.seed, n_stocks=self.n_stocks,
                    start=self.start, end=self.end)
        args.update(kw)
        return generate(**args)


class RealProvider(DataProvider):
    """yfinance-backed panel (cached under cache/)."""
    name = "real"

    def __init__(self, seed=11, n_stocks=200, start="2020-01-01", end="2022-12-31"):
        self.seed, self.n_stocks, self.start, self.end = seed, n_stocks, start, end

    def get_panel(self, **kw):
        from real_data import generate_real_panel
        args = dict(seed=self.seed, n_stocks=self.n_stocks,
                    start=self.start, end=self.end)
        args.update(kw)
        return generate_real_panel(**args)


PROVIDERS = {"synthetic": SyntheticProvider, "real": RealProvider}


def get_provider(name, **kw):
    try:
        return PROVIDERS[name](**kw)
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}")
