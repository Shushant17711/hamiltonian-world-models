import numpy as np

from hwm.envs.base import MechanicalEnv


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
        return np.full((len(q), 1, 1), self.m * self.l**2)

    def dM(self, q):
        return np.zeros((len(q), 1, 1, 1))

    def V(self, q):
        return self.m * self.g * self.l * (1.0 - np.cos(q[:, 0]))

    def dV(self, q):
        return self.m * self.g * self.l * np.sin(q[:, :1])

    def sample_config(self, rng, N):
        return rng.uniform(-np.pi, np.pi, (N, 1))
