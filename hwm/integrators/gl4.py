"""Two-stage Gauss-Legendre (order 4) implicit Runge-Kutta, vectorised over a batch, float64.

GL methods are symplectic, so with u = 0 and no damping the ground-truth energy error stays bounded
(no secular drift) over arbitrarily long horizons. Stage equations are solved by fixed-point iteration,
warm-started across substeps by extrapolating the previous step's stage slopes.
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
    guess: tuple[np.ndarray, np.ndarray] | None = None,
    stats: dict | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """One GL4 step of size h for y' = f(y); y has shape (B, d).

    Returns (y_next, (k1, k2)); pass the stage slopes back via ``integrate`` for the next warm start.
    The converged result does not depend on the guess (up to ``tol``), only the iteration count does.
    """
    B = y.shape[0]
    if guess is None:
        k1 = f(y)
        k2 = k1
    else:
        k1, k2 = guess
    for it in range(max_iter):
        # both stages in one batched call: halves Python overhead for small batches
        ys = np.concatenate([y + h * (A11 * k1 + A12 * k2), y + h * (A21 * k1 + A22 * k2)], 0)
        ks = f(ys)
        k1_new, k2_new = ks[:B], ks[B:]
        delta = max(np.abs(k1_new - k1).max(), np.abs(k2_new - k2).max())
        scale = 1.0 + max(np.abs(k1_new).max(), np.abs(k2_new).max())
        k1, k2 = k1_new, k2_new
        if delta <= tol * scale:
            break
    if stats is not None:
        stats["iters"] = stats.get("iters", 0) + it + 1
        stats["steps"] = stats.get("steps", 0) + 1
    return y + 0.5 * h * (k1 + k2), (k1, k2)


def _extrapolate(k1: np.ndarray, k2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear extrapolation of the stage slopes (at c1, c2 = 1/2 -+ sqrt(3)/6) to 1 + c1, 1 + c2."""
    d = _S3 * (k2 - k1)  # (k2 - k1) / (c2 - c1)
    return k1 + d, k2 + d


def integrate(
    f: Callable[[np.ndarray], np.ndarray], y: np.ndarray, dt: float, substeps: int, **kw
) -> np.ndarray:
    h = dt / substeps
    guess = None
    for _ in range(substeps):
        y, ks = gl4_step(f, y, h, guess=guess, **kw)
        guess = _extrapolate(*ks)
    return y
