"""Reusable physics assertions, applied to every env in tests/test_envs_*.py."""

import time

import numpy as np


def check_dH_finite_difference(env, qp, eps=1e-6, rtol=1e-5):
    n = env.n
    q, p = qp[:, :n].copy(), qp[:, n:].copy()
    dHdq, dHdp = env.dH(q, p)
    for k in range(n):
        e = np.zeros(n)
        e[k] = eps
        fd_q = (env.H(q + e, p) - env.H(q - e, p)) / (2 * eps)
        fd_p = (env.H(q, p + e) - env.H(q, p - e)) / (2 * eps)
        scale = 1.0 + np.abs(fd_q).max() + np.abs(fd_p).max()
        np.testing.assert_allclose(dHdq[:, k], fd_q, atol=rtol * scale)
        np.testing.assert_allclose(dHdp[:, k], fd_p, atol=rtol * scale)


def relative_drift(env, qp0, steps):
    """max_t |H(x_t) - H(x_0)| / E_ref for u = 0, per trajectory; also returns wall time."""
    t0 = time.perf_counter()
    E0 = env.energy(qp0)
    qp, worst = qp0, np.zeros(len(qp0))
    for _ in range(steps):
        qp = env.step(qp)
        worst = np.maximum(worst, np.abs(env.energy(qp) - E0))
    return worst / env.E_ref, time.perf_counter() - t0
