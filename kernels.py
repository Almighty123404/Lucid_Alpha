"""Phase 2 execution kernels (Numba). Conventions:

- Every kernel has an exact observable contract documented below; parity is
  enforced by tests/test_kernels.py against the pre-kernel implementations.
- njit uses cache=True (works in module files, not in `python -c` strings).
  If numba import fails, simulator falls back to the numpy paths — kernels
  are a speed optimization, never a semantic dependency... (see KERNELS_OK).
"""
import numpy as np

try:
    import numba

    @numba.njit(cache=True)
    def _topn_rows(scores, size, out):
        """Per-row top-`size` membership. scores: (T,N) float, NaN = -inf.
        out: (T,N) bool written in place. Ties at the boundary are broken by
        column order (old threshold code included all ties; float ties at the
        exact cutoff are measure-zero — see parity test tolerance note)."""
        T, N = scores.shape
        for t in numba.prange(T):
            order = np.argsort(scores[t])
            for k in range(size):
                out[t, order[N - 1 - k]] = True

    KERNELS_OK = True
except ImportError:
    KERNELS_OK = False
    _topn_rows = None


def topn_rows(scores, size):
    """scores (T,N) with NaN treated as ineligible -> (T,N) bool top-size mask."""
    out = np.zeros(scores.shape, dtype=np.bool_)
    if KERNELS_OK:
        filled = np.where(np.isfinite(scores), scores, -np.inf)
        _topn_rows(filled, int(size), out)
        return out
    # numpy fallback (same contract): exact top-size by order.
    order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf),
                       axis=1, kind="stable")
    T, N = scores.shape
    out[np.arange(T)[:, None], order[:, N - int(size):]] = True
    return out
