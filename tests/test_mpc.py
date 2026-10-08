"""MPC over world models (Req 8.1, 8.2)."""

import numpy as np
import torch
from torch import nn

from hwm.data.dataset import Normaliser
from hwm.envs import make
from hwm.models.base import WorldModel
from hwm.models.ensemble import Ensemble
from hwm.planning.cem import CEMConfig
from hwm.planning.mpc import MPC


class TorchPendulum(WorldModel):
    """Cheap torch pendulum (4 symplectic-Euler substeps): an oracle-quality model for planning tests."""

    def __init__(self, env):
        super().__init__(env.d_obs, env.d_u, d_z=2)
        self.env = env
        self.dummy = nn.Parameter(torch.zeros(1))

    def encode(self, ctx):
        return ctx[:, -1]

    def step(self, z, u):
        th, w = z[:, 0], z[:, 1]
        h = self.env.dt / 4
        for _ in range(4):
            w = w + h * (-9.81 * torch.sin(th) + u[:, 0] * self.env.u_max)
            th = th + h * w
        return torch.stack([th, w], -1)

    def decode(self, z):
        return z


def test_mpc_swings_the_pendulum_up_on_the_true_simulator():
    env = make("pendulum")
    norm = Normaliser(np.zeros(2), np.ones(2))
    mpc = MPC(
        TorchPendulum(env),
        env,
        norm,
        CEMConfig(horizon=30, population=100, elites=10),
        beta=0.0,
        generator=torch.Generator().manual_seed(0),
    )
    qp = env.task_start(1)
    obs = [env.qp_to_obs(qp)[0]]
    for _ in range(env.episode_len):
        a, _ = mpc.act(torch.as_tensor(obs[-1], dtype=torch.float32)[None, None])
        qp = env.step(qp, a.double().numpy()[None] * env.u_max)
        obs.append(env.qp_to_obs(qp)[0])
    assert env.success(np.array(obs))


class _Toy(WorldModel):
    """1-D toy: z' = z + u. Member ``k`` decodes z * (1 + k * relu(z)) so members disagree for z > 0."""

    def __init__(self, cfg, env, obs_mode="state"):
        super().__init__(1, 1, d_z=1)
        self.k = float(cfg.get("k", 0.0))
        self.dummy = nn.Parameter(torch.zeros(1))

    def encode(self, ctx):
        return ctx[:, -1]

    def step(self, z, u):
        return z + u

    def decode(self, z):
        return z * (1 + self.k * torch.relu(z))


class _ToyEnv:
    d_u, u_max = 1, 1.0

    def reward(self, obs, u):  # small preference for moving right
        return 0.1 * obs[..., 0]


def test_disagreement_penalty_steers_plans_away_from_disagreement():
    from hwm.config import Config

    ens = Ensemble(Config({"members": 3, "member": {"name": "_toy"}}), _ToyEnv())
    for i, mem in enumerate(ens.members):
        mem.k = float(i)
    norm = Normaliser(np.zeros(1), np.ones(1))
    ctx = torch.zeros(1, 1, 1)
    chosen = {}
    for beta in (0.0, 5.0):
        mpc = MPC(
            ens,
            _ToyEnv(),
            norm,
            CEMConfig(horizon=5, population=200, elites=20),
            beta=beta,
            generator=torch.Generator().manual_seed(0),
        )
        seq, _ = mpc.cem.plan(lambda a, m=mpc: m.score(ens.encode(ctx), a)[0])
        _, _, D = mpc.score(ens.encode(ctx), seq[None])
        chosen[beta] = (seq.sum().item(), D.item())
    assert chosen[0.0][0] > 2.0  # without the penalty the plan moves right, where members disagree
    assert chosen[5.0][1] < 0.1 * chosen[0.0][1]  # beta > 0 picks a plan with far less disagreement


from hwm.models.registry import register

register("_toy")(_Toy)
