import numpy as np

from hwm.envs.base import MechanicalEnv
from hwm.envs.xp import mat2, xp_of


class Orbit(MechanicalEnv):
    """Planar two-body problem in relative coordinates (reduced mass 1, mu = k = 1); 2-D thrust.

    H = |p|^2 / 2 - 1/|q|. Bands are over the semi-major axis a (E = -1 / (2a)), eccentricity <= e_max.
    """

    name = "orbit"
    n = 2
    d_u = 2
    u_max = 0.05
    separable = True
    substeps = 4
    E_ref = 0.5  # |E| of the a = 1 orbit

    def __init__(self, mu: float = 1.0, e_max: float = 0.3, damping: float = 0.0):
        self.mu, self.e_max, self.damping = mu, e_max, damping

    def M(self, q):
        one, zero = xp_of(q).ones_like(q[:, 0]), xp_of(q).zeros_like(q[:, 0])
        return mat2(one, zero, zero, one)

    def dM(self, q):
        return xp_of(q).zeros_like(q)[:, :, None, None] * xp_of(q).zeros_like(q)[:, None, :, None]

    def V(self, q):
        return -self.mu / ((q**2).sum(-1) ** 0.5)

    def dV(self, q):
        r = ((q**2).sum(-1) ** 0.5)[:, None]
        return self.mu * q / r**3

    def qdot(self, q, p):
        return p

    def sample_config(self, rng, N):  # pragma: no cover - orbit samples from Kepler elements instead
        raise NotImplementedError

    def sample_band(self, rng, band, N, e_max: float | None = None):
        """Initial states from Kepler elements: a ~ U(band), e ~ U(0, e_max), random phase/orientation/sense."""
        e_max = self.e_max if e_max is None else e_max
        a = rng.uniform(band[0], band[1], N)
        e = rng.uniform(0.0, e_max, N)
        nu = rng.uniform(-np.pi, np.pi, N)  # true anomaly
        omega = rng.uniform(-np.pi, np.pi, N)  # argument of periapsis
        sense = rng.choice([-1.0, 1.0], N)  # prograde / retrograde
        return self.from_elements(a, e, nu, omega, sense)

    def from_elements(self, a, e, nu, omega=0.0, sense=1.0):
        a, e, nu, omega, sense = np.broadcast_arrays(
            *(np.asarray(x, float) for x in (a, e, nu, omega, sense))
        )
        slr = a * (1.0 - e**2)  # semi-latus rectum
        r = slr / (1.0 + e * np.cos(nu))
        pos = np.stack([r * np.cos(nu), r * np.sin(nu)], -1)
        vel = np.sqrt(self.mu / slr)[..., None] * np.stack([-np.sin(nu), e + np.cos(nu)], -1)
        pos[..., 1] *= sense
        vel[..., 1] *= sense
        c, s = np.cos(omega)[..., None], np.sin(omega)[..., None]
        rot = lambda v: np.stack(
            [c[..., 0] * v[..., 0] - s[..., 0] * v[..., 1], s[..., 0] * v[..., 0] + c[..., 0] * v[..., 1]], -1
        )
        return np.concatenate([rot(pos), rot(vel)], -1)

    def elements(self, qp):
        """(a, e, angular momentum) of each state, for tests and task checks."""
        q, p = qp[..., :2], qp[..., 2:]
        r = np.linalg.norm(q, axis=-1)
        v2 = (p**2).sum(-1)
        E = 0.5 * v2 - self.mu / r
        a = -self.mu / (2 * E)
        rv = (q * p).sum(-1)
        e_vec = ((v2 - self.mu / r)[..., None] * q - rv[..., None] * p) / self.mu
        L = q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0]
        return a, np.linalg.norm(e_vec, axis=-1), L

    # ---- task: transfer from the circular r = 1 orbit to r = 1.5 ---------------------------------
    episode_len = 400
    hold = 50
    r_target = 1.5

    def task_start(self, N: int = 1):
        return np.repeat(self.from_elements(1.0, 0.0, 0.0)[None], N, 0)

    @staticmethod
    def radial(obs):
        """(r, r_dot) from obs (..., 4) = (x, y, vx, vy); numpy or torch."""
        r = ((obs[..., :2] ** 2).sum(-1)) ** 0.5
        return r, (obs[..., :2] * obs[..., 2:]).sum(-1) / r

    def reward(self, obs, u):
        r, r_dot = self.radial(obs)
        return -(abs(r - self.r_target) + abs(r_dot)) - self._effort(u, self.u_max)

    def success(self, obs_seq):
        r, r_dot = self.radial(np.asarray(obs_seq)[..., -self.hold :, :])
        return ((np.abs(r - self.r_target) < 0.05) & (np.abs(r_dot) < 0.05)).all(-1)
