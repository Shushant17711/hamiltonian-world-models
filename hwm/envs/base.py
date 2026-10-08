"""Environment contract shared by all ground-truth simulators.

Internal state is canonical (q, p) with shape (B, 2n), float64. Observations are (q, q_dot) (Req 3.1).
Actions u have shape (B, d_u) and are held constant over a step of length dt (zero-order hold).
Angle coordinates are never wrapped inside the integrator; use ``wrap_obs`` for display/reward.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod

import numpy as np

from hwm.integrators.gl4 import integrate


class Env(ABC):
    name: str
    n: int  # degrees of freedom
    d_u: int  # action dimension
    u_max: float
    E_ref: float  # energy scale used for bands and for normalising drift
    dt: float = 0.05
    substeps: int = 10
    angle_dims: tuple[int, ...] = ()  # indices into q that are angles
    damping: float = 0.0  # linear damping: p_dot -= damping * q_dot

    # ---- physics -------------------------------------------------------------------------------
    @abstractmethod
    def H(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Hamiltonian, shape (B,)."""

    @abstractmethod
    def dH(self, q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(dH/dq, dH/dp), each (B, n)."""

    @functools.cached_property
    def B(self) -> np.ndarray:
        """Constant input matrix (n, d_u): p_dot += B @ u."""
        return np.eye(self.n, self.d_u)

    def vector_field(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        q, p = y[:, : self.n], y[:, self.n :]
        dHdq, dHdp = self.dH(q, p)
        p_dot = -dHdq + u @ self.B.T
        if self.damping:
            p_dot = p_dot - self.damping * dHdp
        return np.concatenate([dHdp, p_dot], axis=1)

    def step(self, qp: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        qp = np.asarray(qp, dtype=np.float64)
        u = np.zeros((qp.shape[0], self.d_u)) if u is None else np.clip(np.asarray(u, np.float64), -self.u_max, self.u_max)
        return integrate(lambda y: self.vector_field(y, u), qp, self.dt, self.substeps)

    def energy(self, qp: np.ndarray) -> np.ndarray:
        return self.H(qp[..., : self.n].reshape(-1, self.n), qp[..., self.n :].reshape(-1, self.n)).reshape(
            qp.shape[:-1]
        )

    # ---- observations --------------------------------------------------------------------------
    @property
    def d_obs(self) -> int:
        return 2 * self.n

    @abstractmethod
    def qp_to_obs(self, qp: np.ndarray) -> np.ndarray:
        """(..., 2n) canonical -> (..., 2n) observation (q, q_dot)."""

    @abstractmethod
    def obs_to_qp(self, obs: np.ndarray) -> np.ndarray:
        """Inverse of qp_to_obs."""

    def wrap_obs(self, obs: np.ndarray) -> np.ndarray:
        out = np.array(obs, dtype=np.float64, copy=True)
        for i in self.angle_dims:
            out[..., i] = (out[..., i] + np.pi) % (2 * np.pi) - np.pi
        return out

    # ---- initial states ------------------------------------------------------------------------
    @abstractmethod
    def sample_band(self, rng: np.random.Generator, band: tuple[float, float], N: int) -> np.ndarray:
        """N canonical states whose energy E satisfies band[0] <= E / E_ref <= band[1] (orbit: a in band)."""


def _flatten(q: np.ndarray, p: np.ndarray):
    return np.atleast_2d(q), np.atleast_2d(p)


def _solve_small(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve M x = b for batches of 1x1 / 2x2 matrices in closed form (numpy solve is call-overhead bound)."""
    n = M.shape[-1]
    if n == 1:
        return b / M[:, 0, :]
    if n == 2:
        a, c, d, e = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
        det = a * e - c * d
        x0 = (e * b[:, 0] - c * b[:, 1]) / det
        x1 = (a * b[:, 1] - d * b[:, 0]) / det
        return np.stack([x0, x1], axis=1)
    return np.linalg.solve(M, b[..., None])[..., 0]


class MechanicalEnv(Env):
    """H(q, p) = 0.5 p^T M(q)^-1 p + V(q). Subclasses provide M, dM, V, dV and ``sample_config``.

    dH/dp = M^-1 p = q_dot;  dH/dq = -0.5 q_dot^T (dM/dq_k) q_dot + dV/dq_k.
    ``separable = True`` (constant M) skips the dM term.
    """

    separable: bool = False

    @abstractmethod
    def M(self, q: np.ndarray) -> np.ndarray:
        """Mass matrix (B, n, n)."""

    @abstractmethod
    def dM(self, q: np.ndarray) -> np.ndarray:
        """dM/dq, shape (B, n, n, n) where [:, k] = dM/dq_k."""

    @abstractmethod
    def V(self, q: np.ndarray) -> np.ndarray:
        """Potential, (B,)."""

    @abstractmethod
    def dV(self, q: np.ndarray) -> np.ndarray:
        """dV/dq, (B, n)."""

    @abstractmethod
    def sample_config(self, rng: np.random.Generator, N: int) -> np.ndarray:
        """Candidate configurations q (N, n) for band sampling."""

    def qdot(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        return _solve_small(self.M(q), p)

    def H(self, q, p):
        q, p = _flatten(q, p)
        return 0.5 * (p * self.qdot(q, p)).sum(-1) + self.V(q)

    def dH(self, q, p):
        v = self.qdot(q, p)
        if self.separable:
            return self.dV(q), v
        dM = self.dM(q)
        quad = (v[:, None, :, None] * dM * v[:, None, None, :]).sum(axis=(2, 3))
        return -0.5 * quad + self.dV(q), v

    def qp_to_obs(self, qp):
        shape = qp.shape
        flat = qp.reshape(-1, 2 * self.n)
        q, p = flat[:, : self.n], flat[:, self.n :]
        return np.concatenate([q, self.qdot(q, p)], axis=1).reshape(shape)

    def obs_to_qp(self, obs):
        shape = obs.shape
        flat = obs.reshape(-1, 2 * self.n)
        q, v = flat[:, : self.n], flat[:, self.n :]
        p = (self.M(q) * v[:, None, :]).sum(-1)
        return np.concatenate([q, p], axis=1).reshape(shape)

    def sample_band(self, rng, band, N):
        lo, hi = band
        out_q, out_p, have = [], [], 0
        for _ in range(10_000):
            m = max(4 * (N - have), 64)
            E = rng.uniform(lo, hi, m) * self.E_ref
            q = self.sample_config(rng, m)
            Vq = self.V(q)
            ok = Vq < E - 1e-9 * self.E_ref
            ok &= self._accept(q, E)
            if not ok.any():
                continue
            q, E, Vq = q[ok], E[ok], Vq[ok]
            d = rng.standard_normal(q.shape)
            d = self._shape_momentum(q, d)
            T1 = 0.5 * (d * self.qdot(q, d)).sum(-1)
            p = d * np.sqrt((E - Vq) / T1)[:, None]
            out_q.append(q)
            out_p.append(p)
            have += len(q)
            if have >= N:
                break
        else:  # pragma: no cover
            raise RuntimeError(f"{self.name}: could not sample band {band}")
        q = np.concatenate(out_q)[:N]
        p = np.concatenate(out_p)[:N]
        return np.concatenate([q, p], axis=1)

    # hooks for env-specific rejection / momentum shaping (cart-pole uses both)
    def _accept(self, q: np.ndarray, E: np.ndarray) -> np.ndarray:
        return np.ones(len(q), dtype=bool)

    def _shape_momentum(self, q: np.ndarray, d: np.ndarray) -> np.ndarray:
        return d
