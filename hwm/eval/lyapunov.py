"""Maximal Lyapunov exponent of a simulator (design §7, Req 9.2; PREREGISTRATION.md §5 H4).

Benettin's two-trajectory method on the ground-truth GL4 simulator with u = 0: a reference state x
and a companion x + d0 v (v a random unit vector in canonical (q, p) space) are stepped together;
after every step the separation d is measured, log(d / d0) accumulated, and the companion pulled back
to distance d0 along the current separation. lambda = (sum of logs) / (steps * dt), averaged over the
initial states; the Lyapunov time is t_lambda = 1 / lambda (in seconds).
"""

from __future__ import annotations

import numpy as np

from hwm.eval.thresholds import H4_LYAPUNOV_D0, H4_LYAPUNOV_INITIAL_STATES


def lyapunov_exponent(
    env,
    qp0: np.ndarray,
    steps: int = 4000,
    d0: float = H4_LYAPUNOV_D0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-initial-state maximal Lyapunov exponent estimates (1/s) from canonical states qp0 (N, 2n)."""
    rng = rng or np.random.default_rng(0)
    x = np.array(qp0, dtype=np.float64)
    v = rng.standard_normal(x.shape)
    y = x + d0 * v / np.linalg.norm(v, axis=1, keepdims=True)
    total = np.zeros(len(x))
    for _ in range(steps):
        both = env.step(np.concatenate([x, y]))
        x, y = both[: len(x)], both[len(x) :]
        diff = y - x
        d = np.linalg.norm(diff, axis=1)
        total += np.log(d / d0)
        y = x + diff * (d0 / d)[:, None]
    return total / (steps * env.dt)


def lyapunov_by_band(
    env,
    bands: dict[str, tuple[float, float]],
    n_states: int = H4_LYAPUNOV_INITIAL_STATES,
    steps: int = 4000,
    seed: int = 0,
) -> dict[str, dict]:
    """{band: {"lambda", "lambda_per_state", "t_lambda"}} with initial states sampled from each band."""
    out = {}
    for i, (name, band) in enumerate(bands.items()):
        rng = np.random.default_rng([seed, i])
        qp0 = env.sample_band(rng, tuple(band), n_states)
        lam = lyapunov_exponent(env, qp0, steps=steps, rng=rng)
        mean = float(lam.mean())
        out[name] = {
            "band": list(band),
            "lambda": mean,
            "lambda_per_state": [float(x) for x in lam],
            "t_lambda": 1.0 / mean if mean > 0 else float("inf"),
            "steps": steps,
        }
    return out
