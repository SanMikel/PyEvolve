import numpy as np
from numba import njit, prange


@njit(cache=True)
def linear_interp(xi, x, y, uniform, dx):
    """
    Single-point linear interpolation, uniform or non-uniform grid.
    Callable from Python (pays boundary-crossing cost) or, more
    importantly, from inside other @njit functions (near-zero cost).
    """
    n = x.shape[0]

    # Boundary handling (clamp, matches original FastInterp behaviour)
    if xi <= x[0]:
        return y[0]
    if xi >= x[n - 1]:
        return y[n - 1]

    if uniform:
        i = int((xi - x[0]) / dx)
        # guard against float rounding pushing i to n-1
        if i >= n - 1:
            i = n - 2
    else:
        i = np.searchsorted(x, xi) - 1
        if i < 0:
            i = 0
        elif i >= n - 1:
            i = n - 2

    t = (xi - x[i]) / (x[i + 1] - x[i])
    return y[i] + t * (y[i + 1] - y[i])


@njit(parallel=True, cache=True)
def linear_interp_batch(xi_arr, x, y, uniform, dx):
    """
    Vectorized/batched version for standalone use outside the hot loop
    (e.g. diagnostics, plotting, precomputation over large arrays).
    Not needed inside generate_emission_history — use linear_interp there.
    """
    out = np.empty_like(xi_arr)
    for k in prange(xi_arr.shape[0]):
        out[k] = linear_interp(xi_arr[k], x, y, uniform, dx)
    return out
