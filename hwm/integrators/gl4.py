"""Two-stage Gauss-Legendre (order 4) implicit Runge-Kutta, vectorised over a batch, float64.

GL methods are symplectic, so with u = 0 and no damping the ground-truth energy error stays bounded
(no secular drift) over arbitrarily long horizons. Stage equations are solved by fixed-point iteration.
"""

from collections.abc import Callable

import numpy as np

_S3 = np.sqrt(3.0)
A11, A12 = 0.25, 0.25 - _S3 / 6.0
A21, A22 = 0.25 + _S3 / 6.0, 0.25


def gl4_step(
    f: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    h: float,
    tol: float = 1e-13,
    max_iter: int = 50,
) -> np.ndarray:
    """One GL4 step of size h for y' = f(y); y has shape (B, d)."""
    k1 = f(y)
    k2 = k1
    for _ in range(max_iter):
        k1_new = f(y + h * (A11 * k1 + A12 * k2))
        k2_new = f(y + h * (A21 * k1 + A22 * k2))
        delta = max(np.abs(k1_new - k1).max(), np.abs(k2_new - k2).max())
        scale = 1.0 + max(np.abs(k1_new).max(), np.abs(k2_new).max())
        k1, k2 = k1_new, k2_new
        if h * delta <= tol * scale:
            break
    return y + 0.5 * h * (k1 + k2)


def integrate(
    f: Callable[[np.ndarray], np.ndarray], y: np.ndarray, dt: float, substeps: int, **kw
) -> np.ndarray:
    h = dt / substeps
    for _ in range(substeps):
        y = gl4_step(f, y, h, **kw)
    return y
