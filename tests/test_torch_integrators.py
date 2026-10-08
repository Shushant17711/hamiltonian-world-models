"""Torch integrators: convergence order, long-horizon energy behaviour, differentiability (Req 5.2)."""

import math

import numpy as np
import pytest
import torch

from hwm.integrators.torch_integrators import hamiltonian_field, implicit_midpoint, leapfrog, rk4


def _sho_field(y):  # H = (q^2 + p^2) / 2
    q, p = y[..., :1], y[..., 1:]
    return torch.cat([p, -q], dim=-1)


def _exact(y0, t):
    q0, p0 = y0[..., 0], y0[..., 1]
    return torch.stack([q0 * math.cos(t) + p0 * math.sin(t), -q0 * math.sin(t) + p0 * math.cos(t)], -1)


def _integrate(method, y0, dt, T):
    y = y0.clone()
    for _ in range(round(T / dt)):
        if method == "rk4":
            y = rk4(_sho_field, y, dt)
        elif method == "leapfrog":
            q, p = leapfrog(lambda q: q, lambda p: p, y[..., :1], y[..., 1:], dt)
            y = torch.cat([q, p], -1)
        else:
            y = implicit_midpoint(_sho_field, y, dt, iters=12)
    return y


@pytest.mark.parametrize("method,order", [("rk4", 4), ("leapfrog", 2), ("implicit_midpoint", 2)])
def test_convergence_order(method, order):
    y0 = torch.tensor([[1.0, 0.3]], dtype=torch.float64)
    dts = [0.1, 0.05, 0.025]
    errs = [(_integrate(method, y0, dt, 2.0) - _exact(y0, 2.0)).abs().max().item() for dt in dts]
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    assert abs(slope - order) < 0.25, (method, errs, slope)


@pytest.mark.parametrize("method", ["leapfrog", "implicit_midpoint"])
def test_symplectic_energy_bounded_10k(method):
    y = torch.tensor([[1.0, 0.0], [0.2, 1.5], [-0.7, 0.4]], dtype=torch.float64)
    H = lambda y: 0.5 * (y**2).sum(-1)
    H0, dt = H(y), 0.1
    worst = 0.0
    for i in range(10_000):
        if method == "leapfrog":
            q, p = leapfrog(lambda q: q, lambda p: p, y[..., :1], y[..., 1:], dt)
            y = torch.cat([q, p], -1)
        else:
            y = implicit_midpoint(_sho_field, y, dt)
        if i % 100 == 99:
            worst = max(worst, ((H(y) - H0).abs() / H0).max().item())
    # leapfrog: O(dt^2) bounded oscillation; implicit midpoint: exact for quadratic H up to the
    # 6-iteration fixed-point residual (~1.6e-6 after 10k steps)
    assert worst < (5e-3 if method == "leapfrog" else 1e-5), worst


def test_rk4_drifts_where_symplectic_does_not():
    """Sanity contrast: RK4 is not symplectic, its energy decays secularly at large dt."""
    y = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    for _ in range(10_000):
        y = rk4(_sho_field, y, 0.5)
    assert 0.5 * (y**2).sum().item() < 0.5 * 0.5


def test_gradients_flow_through_implicit_midpoint():
    k = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)

    def run(kval):
        y = torch.tensor([[0.8, -0.2]], dtype=torch.float64)
        f = hamiltonian_field(lambda q, p: Hk_val(q, p, kval), n=1)
        for _ in range(5):
            y = implicit_midpoint(f, y, 0.1)
        return y[0, 0]

    def Hk_val(q, p, kval):
        return 0.5 * kval * (q**2).sum(-1) + 0.5 * (p**2).sum(-1) + 0.1 * (q**4).sum(-1)

    out = run(k)
    (g,) = torch.autograd.grad(out, k)
    eps = 1e-6
    fd = (
        run(torch.tensor(1.7 + eps, dtype=torch.float64)) - run(torch.tensor(1.7 - eps, dtype=torch.float64))
    ).item() / (2 * eps)
    assert g.abs() > 1e-4
    assert abs(g.item() - fd) < 1e-5 * max(1.0, abs(fd))


def test_hamiltonian_field_matches_analytic():
    f = hamiltonian_field(lambda q, p: 0.5 * (p**2).sum(-1) - torch.cos(q).sum(-1), n=1)
    y = torch.tensor([[0.4, 1.2]], dtype=torch.float64)
    torch.testing.assert_close(f(y), torch.tensor([[1.2, -math.sin(0.4)]], dtype=torch.float64))
