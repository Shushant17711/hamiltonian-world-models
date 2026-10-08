import numpy as np

from hwm.envs.base import MechanicalEnv
from hwm.envs.xp import mat2, xp_of


class Acrobot(MechanicalEnv):
    """Double pendulum with point masses, torque at the elbow only (underactuated, chaotic).

    theta1: shoulder angle from the downward vertical; theta2: elbow angle relative to link 1.
    """

    name = "acrobot"
    n = 2
    d_u = 1
    u_max = 4.0
    angle_dims = (0, 1)
    substeps = 8  # chaotic: needs h = dt/8 to hold the 1e-5 drift target

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81, damping: float = 0.0):
        self.m1, self.m2, self.l1, self.l2, self.g, self.damping = m1, m2, l1, l2, g, damping
        self.E_ref = 2 * (m1 + m2) * g * l1 + 2 * m2 * g * l2  # V of the fully upright configuration

    @property
    def B(self):
        return np.array([[0.0], [1.0]])

    def M(self, q):
        xp = xp_of(q)
        m1, m2, l1, l2 = self.m1, self.m2, self.l1, self.l2
        c2 = xp.cos(q[:, 1])
        m11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2
        m12 = m2 * l2**2 + m2 * l1 * l2 * c2
        return mat2(m11, m12, m12, xp.ones_like(c2) * m2 * l2**2)

    def dM(self, q):
        xp = xp_of(q)
        s = -self.m2 * self.l1 * self.l2 * xp.sin(q[:, 1])
        z = xp.zeros_like(s)
        return xp.stack([mat2(z, z, z, z), mat2(2 * s, s, s, z)], 1)

    def V(self, q):
        xp = xp_of(q)
        return (self.m1 + self.m2) * self.g * self.l1 * (1 - xp.cos(q[:, 0])) + self.m2 * self.g * self.l2 * (
            1 - xp.cos(q[:, 0] + q[:, 1])
        )

    def dV(self, q):
        xp = xp_of(q)
        s12 = self.m2 * self.g * self.l2 * xp.sin(q[:, 0] + q[:, 1])
        return xp.stack([(self.m1 + self.m2) * self.g * self.l1 * xp.sin(q[:, 0]) + s12, s12], -1)

    def sample_config(self, rng, N):
        return rng.uniform(-np.pi, np.pi, (N, 2))

    def tip_height(self, q):
        """Height of the tip above the pivot (numpy or torch)."""
        xp = xp_of(q)
        return -self.l1 * xp.cos(q[..., 0]) - self.l2 * xp.cos(q[..., 0] + q[..., 1])

    # ---- task: raise the tip ---------------------------------------------------------------------
    episode_len = 300
    tip_target = 1.5

    def reward(self, obs, u):
        return self.tip_height(obs) / (self.l1 + self.l2) - self._effort(u, self.u_max)

    def success(self, obs_seq):
        return (self.tip_height(np.asarray(obs_seq)) >= self.tip_target).any(-1)
