"""Model A: action-conditioned residual next-state MLP (design §5, Req 4.1).

The latent is the normalised observation itself: ``encode`` takes the last context frame,
``step`` is ``z + f([z, u])`` and ``decode`` is the identity. State mode only (in pixel mode A has
no latent and is not run).
"""

from __future__ import annotations

import torch

from hwm.models.base import WorldModel
from hwm.models.nets import mlp
from hwm.models.registry import register


@register("mlp")
class MLPModel(WorldModel):
    def __init__(self, cfg, env, obs_mode: str = "state"):
        if obs_mode != "state":
            raise ValueError("model A (mlp) is state-only")
        super().__init__(env.d_obs, env.d_u, d_z=env.d_obs, obs_mode="state")
        self.f = mlp(
            env.d_obs + env.d_u,
            env.d_obs,
            hidden=cfg.get("hidden", 256),
            layers=cfg.get("layers", 3),
            layernorm=cfg.get("layernorm", False),
        )

    def encode(self, ctx: torch.Tensor) -> torch.Tensor:
        return ctx[:, -1]

    def step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return z + self.f(torch.cat([z, u], dim=-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z
