import numpy as np

from hwm.envs.base import MechanicalEnv
from hwm.envs.xp import mat2, xp_of


class CartPole(MechanicalEnv):
    """Cart (x) with a point-mass pole (theta from the downward vertical); horizontal force on the cart.

    Bob position (x + l sin th, -l cos th). Non-separable: M depends on theta.
    """

    name = "cartpole"
    n = 2
    d_u = 1
    u_max = 10.0
    angle_dims = (1,)
    substeps = 4
    x_limit = 3.0  # trajectories that leave |x| <= x_limit are rejected (camera covers 3.2)
    max_cart_share = 0.2  # cap on the cart's share of kinetic energy at band sampling

    def __init__(self, m_c=1.0, m_p=0.1, l=0.5, g=9.81, damping: float = 0.0):
        self.m_c, self.m_p, self.l, self.g, self.damping = m_c, m_p, l, g, damping
        self.E_ref = 2.0 * m_p * g * l

    def M(self, q):
        xp = xp_of(q)
        c = self.m_p * self.l * xp.cos(q[:, 1])
        return mat2(xp.ones_like(c) * (self.m_c + self.m_p), c, c, xp.ones_like(c) * self.m_p * self.l**2)

    def dM(self, q):
        xp = xp_of(q)
        s = -self.m_p * self.l * xp.sin(q[:, 1])
        z = xp.zeros_like(s)
        return xp.stack([mat2(z, z, z, z), mat2(z, s, s, z)], 1)

    def V(self, q):
        return self.m_p * self.g * self.l * (1.0 - xp_of(q).cos(q[:, 1]))

    def dV(self, q):
        xp = xp_of(q)
        return xp.stack([xp.zeros_like(q[:, 0]), self.m_p * self.g * self.l * xp.sin(q[:, 1])], -1)

    def sample_config(self, rng, N):
        return np.stack([rng.uniform(-1.5, 1.5, N), rng.uniform(-np.pi, np.pi, N)], -1)

    def _sample_momentum_direction(self, rng, q):
        v = rng.standard_normal(q.shape)
        v[:, 0] *= 0.3
        T = 0.5 * (v * (self.M(q) @ v[..., None])[..., 0]).sum(-1)
        T_cart = 0.5 * (self.m_c + self.m_p) * v[:, 0] ** 2
        p = (self.M(q) @ v[..., None])[..., 0]
        return p, T_cart <= self.max_cart_share * T

    def valid_state(self, qp):
        return np.abs(qp[..., 0]) <= self.x_limit

    # ---- task: swing up and balance near the centre ---------------------------------------------
    episode_len = 250
    hold = 50

    def reward(self, obs, u):
        return -xp_of(obs).cos(obs[..., 1]) - 0.05 * obs[..., 0] ** 2 - self._effort(u, self.u_max)

    def success(self, obs_seq):
        tail = np.asarray(obs_seq)[..., -self.hold :, :]
        return ((np.cos(tail[..., 1]) < -0.95) & (np.abs(tail[..., 0]) < 2.0)).all(-1)
