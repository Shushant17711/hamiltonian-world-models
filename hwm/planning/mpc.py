"""MPC over a world model with CEM (design §8, Req 8.1, 8.2).

At every env step: encode the context once, roll all CEM candidates out in one batch from that
latent, score them, execute the first action of the best plan, shift the plan.

    cost(a) = - sum_t r_hat_t + beta * sum_t disagreement_t

Reward: state mode uses the env's analytic reward of the decoded state (for an ensemble, of the member
mean); pixel mode uses the learned reward head (ensemble: member mean).

Disagreement (ensembles only; single models have none, i.e. beta is irrelevant):
* state mode: per-step std over members of the decoded normalised state, averaged over dims (design §4);
* pixel mode: per-step std over members of the predicted reward. Decoding 64x64 frames for every
  candidate (5 members x 400 x 30 per CEM iteration) is far too costly, and the reward is the quantity
  the planner actually exploits; recorded in design §8 and LIMITATIONS.md.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hwm.planning.cem import CEM, CEMConfig


class MPC:
    def __init__(
        self,
        model,
        env,
        normaliser=None,
        cem: CEMConfig | None = None,
        beta: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        self.model, self.env, self.normaliser, self.beta = model, env, normaliser, beta
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.cem = CEM(env.d_u, cem, device=self.device, generator=generator)
        self.is_ens = hasattr(model, "members")
        if model.obs_mode == "state" and normaliser is None:
            raise ValueError("state-mode MPC needs the normaliser to evaluate the env reward")

    def reset(self) -> None:
        self.cem.reset()

    # --- scoring ------------------------------------------------------------------------------------
    def _state_reward(self, obs_norm: Tensor, a: Tensor) -> Tensor:
        x = self.normaliser.denorm(obs_norm)
        return self.env.reward(x, a * self.env.u_max)

    def score(self, z0: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(cost, total reward, total disagreement) of candidates ``actions`` (P, T, d_u) from z0 (1, d_z)."""
        m = self.model
        P, T = actions.shape[:2]
        z = z0.expand(P, -1)
        R = torch.zeros(P, device=self.device, dtype=self.dtype)
        D = torch.zeros_like(R)
        for t in range(T):
            u = actions[:, t]
            z = m.step(z, u)
            if self.is_ens:
                parts = m._parts(z)
                if m.obs_mode == "state":
                    obs = torch.stack([mm.decode(zp) for mm, zp in zip(m.members, parts, strict=True)])
                    r = self._state_reward(obs.mean(0), u)
                    d = obs.std(0, unbiased=False).mean(-1)
                else:
                    rs = torch.stack([mm.reward(zp, u) for mm, zp in zip(m.members, parts, strict=True)])
                    r, d = rs.mean(0), rs.std(0, unbiased=False)
                D = D + d
            elif m.obs_mode == "state":
                r = self._state_reward(m.decode(z), u)
            else:
                r = m.reward(z, u)
            R = R + r
        return -R + self.beta * D, R, D

    @torch.no_grad()
    def act(self, ctx: Tensor) -> tuple[Tensor, dict[str, float]]:
        """First action (d_u,) in [-1, 1] for the context ``ctx`` (1, k, *obs) in the model's input space."""
        was_training = self.model.training
        self.model.eval()
        z0 = self.model.encode(ctx.to(self.device, self.dtype))
        last = {}

        def cost_fn(a: Tensor) -> Tensor:
            c, R, D = self.score(z0, a.to(self.dtype))
            last["R"], last["D"] = R, D
            return c

        a, stats = self.cem.act(cost_fn)
        self.model.train(was_training)
        return a, stats
