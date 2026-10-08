"""Differentiable one-step integrators for learned dynamics (design §5, Req 5.2).

All functions take a single step of length ``dt`` on batched tensors and are built from ordinary
torch ops, so gradients flow through them (including the unrolled fixed-point iterations of the
implicit midpoint rule). The numpy GL4 integrator in ``hwm.integrators.gl4`` is the ground truth;
these are the model-side integrators.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Tensor = torch.Tensor
VectorField = Callable[[Tensor], Tensor]


def rk4(f: VectorField, y: Tensor, dt: float) -> Tensor:
    """Classical 4th-order Runge-Kutta step of ``dy/dt = f(y)``."""
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def leapfrog(
    dHdq: Callable[[Tensor], Tensor],
    dHdp: Callable[[Tensor], Tensor],
    q: Tensor,
    p: Tensor,
    dt: float,
) -> tuple[Tensor, Tensor]:
    """Kick-drift-kick (Stormer-Verlet) step for a separable H(q, p) = T(p) + V(q).

    ``dHdq`` depends on q only and ``dHdp`` on p only. Symplectic and 2nd order.
    """
    p_half = p - 0.5 * dt * dHdq(q)
    q_new = q + dt * dHdp(p_half)
    p_new = p_half - 0.5 * dt * dHdq(q_new)
    return q_new, p_new


def implicit_midpoint(f: VectorField, y: Tensor, dt: float, iters: int = 6) -> Tensor:
    """Implicit midpoint step ``y1 = y + dt f((y + y1) / 2)``, solved by unrolled fixed-point iteration.

    Initialised with one explicit Euler step, then ``iters`` fixed-point updates; every update is a
    plain torch op, so the result is differentiable. Symplectic (up to the fixed-point tolerance)
    for Hamiltonian ``f`` and exactly energy-preserving for quadratic H. 2nd order.
    """
    y1 = y + dt * f(y)
    for _ in range(iters):
        y1 = y + dt * f(0.5 * (y + y1))
    return y1


def hamiltonian_field(
    H: Callable[[Tensor, Tensor], Tensor], n: int, create_graph: bool = True
) -> VectorField:
    """Vector field ``(dH/dp, -dH/dq)`` of a scalar Hamiltonian, by autograd.

    ``H(q, p)`` maps (B, n), (B, n) -> (B,); the returned ``f`` maps y = [q, p] (B, 2n) -> (B, 2n).
    With ``create_graph`` the field stays differentiable w.r.t. the parameters of H (needed for
    training); pass False for evaluation-only rollouts to save memory.
    """

    def f(y: Tensor) -> Tensor:
        with torch.enable_grad():
            if not y.requires_grad:
                y = y.detach().requires_grad_(True)
            q, p = y[..., :n], y[..., n:]
            (g,) = torch.autograd.grad(H(q, p).sum(), y, create_graph=create_graph)
        dq, dp = g[..., :n], g[..., n:]
        return torch.cat([dp, -dq], dim=-1)

    return f
