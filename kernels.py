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
        size = min(max(size, 0), N)
        if size == 0:
            return
        for t in numba.prange(T):
            order = np.argsort(scores[t])
            written = 0
            for pos in range(N - 1, -1, -1):
                idx = order[pos]
                if np.isfinite(scores[t, idx]):
                    out[t, idx] = True
                    written += 1
                    if written == size:
                        break

    KERNELS_OK = True
except ImportError:
    KERNELS_OK = False
    _topn_rows = None


def topn_rows(scores, size):
    """scores (T,N) with NaN treated as ineligible -> (T,N) bool top-size mask."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D matrix")
    T, N = scores.shape
    if not isinstance(size, (int, np.integer)):
        raise ValueError("size must be an integer")
    size = min(max(int(size), 0), N)
    out = np.zeros(scores.shape, dtype=np.bool_)
    if KERNELS_OK:
        filled = np.where(np.isfinite(scores), scores, -np.inf)
        _topn_rows(filled, size, out)
        return out
    # numpy fallback (same contract): exact top-size by order.
    order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf),
                       axis=1, kind="stable")
    for t in range(T):
        finite = np.isfinite(scores[t])
        eligible = order[t][finite[order[t]]]
        out[t, eligible[-size:] if size else []] = True
    return out
