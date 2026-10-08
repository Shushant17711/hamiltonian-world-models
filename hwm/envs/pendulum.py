import numpy as np

from hwm.envs.base import MechanicalEnv
from hwm.envs.xp import xp_of


class Pendulum(MechanicalEnv):
    """Point-mass pendulum, theta measured from the downward vertical; torque input."""

    name = "pendulum"
    n = 1
    d_u = 1
    u_max = 2.0
    angle_dims = (0,)
    separable = True
    substeps = 4

    def __init__(self, m: float = 1.0, l: float = 1.0, g: float = 9.81, damping: float = 0.0):
        self.m, self.l, self.g, self.damping = m, l, g, damping
        self.E_ref = 2.0 * m * g * l  # energy of the upright rest state

    def M(self, q):
        return xp_of(q).ones_like(q)[:, :, None] * (self.m * self.l**2)

    def dM(self, q):
        return xp_of(q).zeros_like(q)[:, :, None, None]

    def V(self, q):
        return self.m * self.g * self.l * (1.0 - xp_of(q).cos(q[:, 0]))

    def dV(self, q):
        return self.m * self.g * self.l * xp_of(q).sin(q)

    def sample_config(self, rng, N):
        return rng.uniform(-np.pi, np.pi, (N, 1))

    # ---- task: swing up and balance ------------------------------------------------------------
    episode_len = 200
    hold = 50

    def reward(self, obs, u):
        return -xp_of(obs).cos(obs[..., 0]) - self._effort(u, self.u_max)

    def success(self, obs_seq):
        tail = np.asarray(obs_seq)[..., -self.hold :, 0]
        return (np.cos(tail) < -0.95).all(-1)
