import numpy as np

from hwm.envs.base import MechanicalEnv


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
        return np.broadcast_to(np.eye(2), (len(q), 2, 2))

    def dM(self, q):
        return np.zeros((len(q), 2, 2, 2))

    def V(self, q):
        return -self.mu / np.linalg.norm(q, axis=-1)

    def dV(self, q):
        r = np.linalg.norm(q, axis=-1, keepdims=True)
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
        a, e, nu, omega, sense = np.broadcast_arrays(*(np.asarray(x, float) for x in (a, e, nu, omega, sense)))
        slr = a * (1.0 - e**2)  # semi-latus rectum
        r = slr / (1.0 + e * np.cos(nu))
        pos = np.stack([r * np.cos(nu), r * np.sin(nu)], -1)
        vel = np.sqrt(self.mu / slr)[..., None] * np.stack([-np.sin(nu), e + np.cos(nu)], -1)
        pos[..., 1] *= sense
        vel[..., 1] *= sense
        c, s = np.cos(omega)[..., None], np.sin(omega)[..., None]
        rot = lambda v: np.stack([c[..., 0] * v[..., 0] - s[..., 0] * v[..., 1], s[..., 0] * v[..., 0] + c[..., 0] * v[..., 1]], -1)  # noqa: E731
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
