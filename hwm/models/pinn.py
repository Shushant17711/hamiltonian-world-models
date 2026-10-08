"""Model B: PINN = model A + a physics residual (design §5, Req 4.2).

The residual compares each predicted transition with the env's analytic acceleration ``a_true``
integrated over the step by Simpson's rule:

    r = ((v_{t+1} - v_t) - dt/6 * (a_0 + 4 a_m + a_1)) / std_v

with a_0, a_1 at the endpoints and a_m at the cubic-Hermite midpoint
(q_m = (q_0+q_1)/2 + dt/8 (v_0-v_1), v_m = (v_0+v_1)/2 + dt/8 (a_0-a_1)); u is held over the step.
The local error is O(dt^5), so true simulator transitions leave a residual far below any model
error (a plain midpoint rule left 1.3e-4 on the acrobot at dt = 0.05 and would bias B there).
The residual is the velocity-change error in normalised velocity units, so its scale matches the
prediction loss. It is applied to the model's own open-loop predictions and to 256 collocation states
per batch, drawn uniformly from the box spanned by the train split. ``a_true`` is the simulator's
dynamics: this is **privileged knowledge** no other model gets, and every table labels B as such.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hwm.models.mlp import MLPModel
from hwm.models.registry import register


@register("pinn")
class PINNModel(MLPModel):
    def __init__(self, cfg, env, obs_mode: str = "state"):
        super().__init__(cfg, env, obs_mode)
        self.env = env
        self.lambda_phys = float(cfg.get("lambda_phys", 1.0))
        self.n_colloc = int(cfg.get("collocation", 256))
        d = env.d_obs
        # filled by prepare(); buffers so a checkpoint carries them
        self._register_normaliser()
        self.register_buffer("box_lo", -torch.ones(d))
        self.register_buffer("box_hi", torch.ones(d))

    def prepare(self, normaliser, train_obs: Tensor) -> None:
        self._load_normaliser(normaliser)
        dev = self.obs_mean.device
        z = normaliser.norm(train_obs.reshape(-1, train_obs.shape[-1]).float().to(dev))
        self.box_lo.copy_(z.min(0).values)
        self.box_hi.copy_(z.max(0).values)

    def physics_residual(self, z0: Tensor, z1: Tensor, a: Tensor) -> Tensor:
        """Per-transition residual (..., n) for normalised states z0 -> z1 under scaled actions a."""
        n = self.env.n
        x0, x1 = self.physical(z0), self.physical(z1)
        lead, dt = x0.shape[:-1], self.env.dt
        q0, v0 = x0[..., :n].reshape(-1, n), x0[..., n:].reshape(-1, n)
        q1, v1 = x1[..., :n].reshape(-1, n), x1[..., n:].reshape(-1, n)
        u = (a * self.env.u_max).reshape(-1, self.env.d_u)
        acc = self.env.accel
        a0, a1 = acc(q0, v0, u), acc(q1, v1, u)
        qm = 0.5 * (q0 + q1) + dt / 8 * (v0 - v1)
        vm = 0.5 * (v0 + v1) + dt / 8 * (a0 - a1)
        am = acc(qm, vm, u)
        r = (v1 - v0) - dt / 6 * (a0 + 4 * am + a1)
        return r.reshape(*lead, n) / self.obs_std[n:]

    def extra_loss(self, batch, ro, horizon: int) -> dict[str, Tensor]:
        obs, act = ro.obs, batch.actions[:, :horizon]
        r_pred = self.physics_residual(obs[:, :-1], obs[:, 1:], act).pow(2).mean()
        g = torch.rand(self.n_colloc, obs.shape[-1], device=obs.device, dtype=obs.dtype)
        z = self.box_lo + (self.box_hi - self.box_lo) * g
        u = torch.rand(self.n_colloc, self.env.d_u, device=obs.device, dtype=obs.dtype) * 2 - 1
        r_col = self.physics_residual(z, self.step(z, u), u).pow(2).mean()
        return {"phys": self.lambda_phys * (r_pred + r_col)}
