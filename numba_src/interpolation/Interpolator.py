#import numpy as np
#from .linear import linear_interpolation_uniform, linear_interpolation_nonuniform
#import time


import numpy as np
from .linear import linear_interp, linear_interp_batch


class FastInterp:
    """
    Thin Python-side convenience wrapper. Does grid-type detection ONCE
    at construction (cheap, happens rarely). Do NOT call this object's
    __call__ from inside a hot loop or from inside another @njit function
    -- it's plain Python. For hot loops, pull out .x, .y, .uniform, .dx
    and call linear_interp(...) directly from your own @njit code.
    """
    def __init__(self, x, y, kind='linear'):
        if kind != 'linear':
            raise NotImplementedError("Only 'linear' kind is supported")

        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.kind = kind

        dxs = np.diff(self.x)
        self.uniform = bool(np.allclose(dxs, dxs[0]))
        self.dx = float(dxs[0]) if self.uniform else 0.0

        # Warm up JIT compilation once, at setup time, so later calls
        # (from Python or from other njit functions) don't pay compile cost
        _ = linear_interp(self.x[0] + 1e-9, self.x, self.y, self.uniform, self.dx)

    def __call__(self, xi):
        # Convenience path for occasional/Python-side use only.
        return linear_interp(xi, self.x, self.y, self.uniform, self.dx)

    def call_batch(self, xi_arr):
        # Convenience path for batched/Python-side use only.
        return linear_interp_batch(
            np.ascontiguousarray(xi_arr, dtype=np.float64),
            self.x, self.y, self.uniform, self.dx
        )

    def as_args(self):
        """
        Unpack into the plain arguments your other @njit functions need,
        e.g.: sudakov.evaluate(parton, mu2, *interp.as_args())
        """
        return self.x, self.y, self.uniform, self.dx