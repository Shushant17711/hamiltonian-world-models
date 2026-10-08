"""Model C: latent Neural ODE + energy penalty (design §5, Req 4.3).

A re-implementation of the friend's model (Optimus2007/physics-informed-world-models), extended with
actions. An MLP encodes the observation into a latent z (d_z = 2n by default), a vector field
``z_dot = f(z, u)`` is integrated with one RK4 step per dt, and an MLP decodes back. An energy head
E(z) is trained two ways:

* **supervision**: (E(enc(x)) - H_true(x) / E_ref)^2 on every true observation of the window;
* **conservation**: the variance over time of E(z_t) along the model's own rollout, on *passive*
  windows only (u = 0, where the true energy is constant).

Energy enters only through the loss: nothing in the architecture makes the rollout conserve it.
That is the contrast with model E.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hwm.integrators.torch_integrators import rk4
from hwm.models.base import WorldModel
from hwm.models.nets import StateDecoder, StateEncoder, mlp
from hwm.models.registry import register


@register("latent_ode")
class LatentODEModel(WorldModel):
    def __init__(self, cfg, env, obs_mode: str = "state"):
        if obs_mode != "state":
            raise ValueError("model C (latent_ode) is state-only")
        d_z = int(cfg.get("d_z") or 2 * env.n)
        super().__init__(env.d_obs, env.d_u, d_z=d_z, obs_mode="state")
        hidden, layers = cfg.get("hidden", 256), cfg.get("layers", 3)
        self.env, self.dt = env, env.dt
        self.enc = StateEncoder(env.d_obs, d_z, 1, hidden, layers)
        self.dec = StateDecoder(d_z, env.d_obs, hidden, layers)
        self.f = mlp(d_z + env.d_u, d_z, hidden, layers)
        self.E = mlp(d_z, 1, hidden, layers)
        self.lambda_var = float(cfg.get("lambda_var", 1.0))
        self.lambda_sup = float(cfg.get("lambda_sup", 1.0))
        self._register_normaliser()

    def prepare(self, normaliser, train_obs: Tensor) -> None:
        self._load_normaliser(normaliser)

    def encode(self, ctx: Tensor) -> Tensor:
        return self.enc(ctx[:, -1:])

    def step(self, z: Tensor, u: Tensor) -> Tensor:
        return rk4(lambda y: self.f(torch.cat([y, u], dim=-1)), z, self.dt)

    def decode(self, z: Tensor) -> Tensor:
        return self.dec(z)

    def energy(self, z: Tensor) -> Tensor:
        return self.E(z).squeeze(-1)

    def true_energy(self, x: Tensor) -> Tensor:
        """H_true / E_ref of normalised state observations (..., d_obs)."""
        return self.env.energy(self.env.obs_to_qp(self.physical(x))) / self.env.E_ref

    def energy_terms(self, batch, ro) -> tuple[Tensor, Tensor]:
        """(conservation variance on passive windows, energy supervision)."""
        E_roll = self.energy(ro.z)  # (B, H+1)
        passive = batch.passive
        if passive.any():
            var = E_roll[passive].var(dim=1, unbiased=False).mean()
        else:
            var = E_roll.sum() * 0.0
        x = torch.cat([batch.ctx[:, -1:], batch.target[:, : ro.z.shape[1] - 1]], dim=1)
        E_enc = self.energy(self.enc(x.reshape(-1, 1, x.shape[-1]))).reshape(x.shape[:2])
        sup = (E_enc - self.true_energy(x)).pow(2).mean()
        return var, sup

    def extra_loss(self, batch, ro, horizon: int) -> dict[str, Tensor]:
        var, sup = self.energy_terms(batch, ro)
        return {"e_var": self.lambda_var * var, "e_sup": self.lambda_sup * sup}
