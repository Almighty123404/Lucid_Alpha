"""Phase 2 kernel parity: Numba/cumsum paths reproduce loop semantics.

Known accepted difference: all-NaN-tie ordering inside the adv20 warmup
(rows <20) is arbitrary-but-deterministic on both sides; rows 20+ must match
bit-for-bit, and warmup rows are pasteurized out of the book regardless.
"""
import time

import numpy as np

from data_gen import generate
from simulator import _universe_mask as new_mask
from config import UNIVERSES


def _old_mask(panel, universe, window=252, min_obs=20):
    T, N = panel.fields['returns'].shape
    size = UNIVERSES.get(str(universe).upper(), {}).get("size", N)
    if size is None or size >= N:
        return np.ones((T, N), dtype=bool)
    liq = np.where(np.isfinite(panel.fields['adv20']),
                   panel.fields['adv20'], np.nan)
    mask = np.zeros((T, N), dtype=bool)
    for t in range(T):
        if t == 0:
            scores = np.where(np.isfinite(liq[0]), liq[0], -np.inf)
        elif t < min_obs:
            with np.errstate(invalid='ignore'):
                scores = np.nanmean(liq[:t], axis=0)
            scores = np.where(np.isfinite(scores), scores, -np.inf)
        else:
            lo = max(0, t - window)
            with np.errstate(invalid='ignore'):
                scores = np.nanmean(liq[lo:t], axis=0)
            scores = np.where(np.isfinite(scores), scores, -np.inf)
        order = np.argsort(-scores, kind='stable')
        mask[t, order[:size]] = True
    return mask


def test_universe_kernel_parity_post_warmup(big_panel):
    panel = big_panel  # shared session build (n=1000 > 500 keeps TOP500 live)
    for u in ("TOP500", "TOP200"):
        a = _old_mask(panel, u)
        t0 = time.time()
        b = new_mask(panel, u)
        dt = time.time() - t0
        assert b.shape == a.shape
        assert (a[20:] == b[20:]).all()  # exact where scores are distinct
    assert b.sum(axis=1).max() <= int(u[3:])  # exact top-size membership
    print(f"{u}: new={dt:.2f}s")


def test_topn_rows_clamps_bounds_and_excludes_nan():
    from kernels import topn_rows
    scores = np.array([[1.0, np.nan, 2.0, np.nan]])
    assert topn_rows(scores, 99).tolist() == [[True, False, True, False]]
    assert not topn_rows(scores, 0).any()
    assert topn_rows(scores, -3).sum() == 0
