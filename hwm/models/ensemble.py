"""Ensemble of world models (design §5 E-ens, Req 5.4).

M members, each built after reseeding torch with its own seed (drawn from the run's RNG, so the
ensemble is reproducible from the run seed) and each trained on its own bootstrap resample of the
train trajectories. The trainer draws one batch per member from ``member_trajs`` and concatenates
them; ``loss`` splits the batch again and sums the member losses.

As a ``WorldModel`` the ensemble's latent is the concatenation of the member latents and it decodes
to the member mean, so evaluation and the trainer treat it like any other model. ``rollout`` also
returns the member predictions and their disagreement (per-step std over members of the decoded
observations, averaged over observation dims), which the planner penalises (design §8).

Config::

    model: {name: ensemble, members: 5, member: {name: hamiltonian, ...}}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from hwm.models.base import Rollout, WorldModel
from hwm.models.registry import get, register


@dataclass
class EnsembleRollout(Rollout):
    members: Tensor | None = None  # (M, B, H+1, *obs) decoded member predictions
    disagreement: Tensor | None = None  # (B, H+1)


@register("ensemble")
class Ensemble(WorldModel):
    def __init__(self, cfg, env, obs_mode: str = "state"):
        M = int(cfg.get("members", 5))
        if M < 1:
            raise ValueError("an ensemble needs at least one member")
        seeds = torch.randint(0, 2**31 - 1, (M,)).tolist()
        cls = get(cfg.member.name)
        state = torch.get_rng_state()
        members = []
        for s in seeds:
            torch.manual_seed(s)
            members.append(cls(cfg.member, env, obs_mode))
        torch.set_rng_state(state)
        m0 = members[0]
        super().__init__(m0.d_obs, m0.d_u, d_z=M * m0.d_z, obs_mode=obs_mode, context=m0.context)
        self.members = nn.ModuleList(members)
        self.member_seeds = seeds
        self.M, self.d_zm = M, m0.d_z
        self.member_trajs: list[Tensor] | None = None  # set by ``prepare``
        self.register_buffer("_trajs", torch.zeros(0, dtype=torch.long), persistent=True)

    # --- training data ---------------------------------------------------------------------------
    def prepare(self, normaliser, train_obs: Tensor) -> None:
        N = train_obs.shape[0]
        idx = [np.random.default_rng(s).integers(0, N, size=N) for s in self.member_seeds]
        self._trajs = torch.as_tensor(np.stack(idx), device=self._trajs.device)
        self.member_trajs = list(self._trajs)
        for m in self.members:
            m.prepare(normaliser, train_obs)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "_trajs"
        if key in state_dict:  # resize so a resumed / reloaded ensemble keeps its resamples
            self._trajs = torch.empty_like(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if self._trajs.numel():
            self.member_trajs = list(self._trajs)

    def loss(self, batch, horizon: int) -> tuple[Tensor, dict[str, float]]:
        total, logs = 0.0, {}
        for m, b in zip(self.members, batch.split(self.M), strict=True):
            loss, lg = m.loss(b, horizon)
            total = total + loss
            for k, v in lg.items():
                logs[k] = logs.get(k, 0.0) + v / self.M  # member mean; "loss" is the member mean too
        return total, logs

    def backward(self, batch, horizon: int) -> dict[str, float]:
        """One member at a time, so only one member's activations are alive (members share no parameters,
        so the gradients equal those of the summed loss); a 5-member pixel ensemble fits in 8 GB this way."""
        logs: dict[str, float] = {}
        for m, b in zip(self.members, batch.split(self.M), strict=True):
            loss, lg = m.loss(b, horizon)
            loss.backward()
            for k, v in lg.items():
                logs[k] = logs.get(k, 0.0) + v / self.M
        return logs

    # --- WorldModel contract -----------------------------------------------------------------------
    def _parts(self, z: Tensor) -> list[Tensor]:
        return list(z.split(self.d_zm, dim=-1))

    def encode(self, ctx: Tensor) -> Tensor:
        return torch.cat([m.encode(ctx) for m in self.members], dim=-1)

    def step(self, z: Tensor, u: Tensor) -> Tensor:
        return torch.cat([m.step(zm, u) for m, zm in zip(self.members, self._parts(z), strict=True)], -1)

    def decode(self, z: Tensor) -> Tensor:
        return torch.stack([m.decode(zm) for m, zm in zip(self.members, self._parts(z), strict=True)]).mean(0)

    def reward(self, z: Tensor, u: Tensor) -> Tensor | None:
        rs = [m.reward(zm, u) for m, zm in zip(self.members, self._parts(z), strict=True)]
        return None if rs[0] is None else torch.stack(rs).mean(0)

    def energy(self, z: Tensor) -> Tensor | None:
        es = [m.energy(zm) for m, zm in zip(self.members, self._parts(z), strict=True)]
        return None if es[0] is None else torch.stack(es).mean(0)

    def rollout(self, ctx: Tensor, actions: Tensor) -> EnsembleRollout:
        ros = [m.rollout(ctx, actions) for m in self.members]
        obs = torch.stack([r.obs for r in ros])  # (M, B, H+1, *obs)
        dis = obs.std(0, unbiased=False).flatten(2).mean(-1)
        rew = None if ros[0].reward is None else torch.stack([r.reward for r in ros]).mean(0)
        return EnsembleRollout(
            z=torch.cat([r.z for r in ros], dim=-1),
            obs=obs.mean(0),
            reward=rew,
            members=obs,
            disagreement=dis,
        )
