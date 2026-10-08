"""Model D: RSSM, Dreamer-style (design §5, Req 4.4).

Latent z = [h, s]: a deterministic GRU state h (200) and a diagonal-Gaussian stochastic state s (30).

* ``encode``: h_0 = 0, s_0 from the posterior q(s | h_0, e(ctx)).
* ``step``:   h' = GRU([s, u], h), s' from the prior p(s | h'). Open-loop rollouts use the prior
  (sampled in training, its mean in eval mode, so evaluation is deterministic).
* ``decode``: MLP([h, s]) -> observation.

Loss = the shared open-loop rollout loss (Req 6.1; Dreamer's "overshooting") + the ELBO over the
window: the posterior is filtered through the true targets, its decodings are reconstructed, and the
KL(posterior || prior) uses KL balancing alpha = 0.8 and a free-nats floor of 1.0.

The observation encoder/decoder are built by ``_make_obs_nets``: MLPs in state mode; in pixel mode a
CNN on k = 3 frames, a transposed-CNN decoder from [h, s] and a reward head r(z) for planning, with
bf16 autocast around the CNNs only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hwm.models.base import WorldModel
from hwm.models.nets import CNNDecoder, CNNEncoder, StateDecoder, StateEncoder, cnn_autocast, mlp
from hwm.models.registry import register

MIN_STD = 0.1


def _gauss(params: Tensor) -> tuple[Tensor, Tensor]:
    mean, raw = params.chunk(2, dim=-1)
    return mean, F.softplus(raw) + MIN_STD


def kl_gauss(m1: Tensor, s1: Tensor, m2: Tensor, s2: Tensor) -> Tensor:
    """KL(N(m1, s1) || N(m2, s2)) summed over the last dim."""
    return (torch.log(s2 / s1) + (s1.pow(2) + (m1 - m2).pow(2)) / (2 * s2.pow(2)) - 0.5).sum(-1)


@register("rssm")
class RSSMModel(WorldModel):
    def __init__(self, cfg, env, obs_mode: str = "state"):
        self.h_dim, self.s_dim = int(cfg.get("deter", 200)), int(cfg.get("stoch", 30))
        super().__init__(env.d_obs, env.d_u, d_z=self.h_dim + self.s_dim, obs_mode=obs_mode)
        hidden, layers = cfg.get("hidden", 256), cfg.get("layers", 3)
        self.embed_dim = int(cfg.get("embed", 256))
        self._make_obs_nets(hidden, layers)
        self.gru_in = nn.Sequential(nn.Linear(self.s_dim + env.d_u, self.h_dim), nn.SiLU())
        self.gru = nn.GRUCell(self.h_dim, self.h_dim)
        self.prior_net = mlp(self.h_dim, 2 * self.s_dim, hidden, 1)
        self.post_net = mlp(self.h_dim + self.embed_dim, 2 * self.s_dim, hidden, 1)
        self.kl_balance = float(cfg.get("kl_balance", 0.8))
        self.free_nats = float(cfg.get("free_nats", 1.0))
        self.beta_kl = float(cfg.get("beta_kl", 1.0))

    def _make_obs_nets(self, hidden: int, layers: int) -> None:
        if self.obs_mode == "pixels":
            self.obs_enc = CNNEncoder(self.embed_dim, context=self.context)
            self.obs_dec = CNNDecoder(self.d_z)
            self.reward_head = mlp(self.d_z, 1, hidden, 2)  # r(z_{t+1}) as in Dreamer
        else:
            self.obs_enc = StateEncoder(self.d_obs, self.embed_dim, self.context, hidden, layers)
            self.obs_dec = StateDecoder(self.d_z, self.d_obs, hidden, layers)

    def _enc(self, x: Tensor) -> Tensor:
        if self.obs_mode == "pixels":
            with cnn_autocast(x):
                return self.obs_enc(x).float()
        return self.obs_enc(x)

    # --- latent pieces -----------------------------------------------------------------------------
    def _split(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return z[..., : self.h_dim], z[..., self.h_dim :]

    def _draw(self, mean: Tensor, std: Tensor) -> Tensor:
        return mean + std * torch.randn_like(std) if self.training else mean

    def prior(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return _gauss(self.prior_net(h))

    def posterior(self, h: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        return _gauss(self.post_net(torch.cat([h, e], dim=-1)))

    def _advance(self, z: Tensor, u: Tensor) -> Tensor:
        h, s = self._split(z)
        return self.gru(self.gru_in(torch.cat([s, u], dim=-1)), h)

    # --- WorldModel contract -----------------------------------------------------------------------
    def encode(self, ctx: Tensor) -> Tensor:
        h = ctx.new_zeros(ctx.shape[0], self.h_dim)
        s = self._draw(*self.posterior(h, self._enc(ctx)))
        return torch.cat([h, s], dim=-1)

    def step(self, z: Tensor, u: Tensor) -> Tensor:
        h = self._advance(z, u)
        return torch.cat([h, self._draw(*self.prior(h))], dim=-1)

    def decode(self, z: Tensor) -> Tensor:
        if self.obs_mode == "pixels":
            with cnn_autocast(z):
                return self.obs_dec(z).float()
        return self.obs_dec(z)

    def reward(self, z: Tensor, u: Tensor) -> Tensor | None:
        return self.reward_head(z).squeeze(-1) if self.obs_mode == "pixels" else None

    # --- ELBO --------------------------------------------------------------------------------------
    def filter(self, batch, horizon: int) -> dict[str, Tensor]:
        """Posterior filtering over the window: returns posterior latents and KL statistics."""
        ctx, act, target = batch.ctx, batch.actions[:, :horizon], batch.target[:, :horizon]
        B, H = act.shape[:2]
        stacks = self.target_stacks(ctx, target)
        emb = self._enc(stacks.reshape(B * H, *stacks.shape[2:])).reshape(B, H, -1)
        z = self.encode(ctx)
        zs, kls_post, kls_prior = [], [], []
        for t in range(H):
            h = self._advance(z, act[:, t])
            pm, ps = self.prior(h)
            qm, qs = self.posterior(h, emb[:, t])
            z = torch.cat([h, self._draw(qm, qs)], dim=-1)
            zs.append(z)
            # balanced KL: train the prior towards the posterior faster than the reverse
            kls_prior.append(kl_gauss(qm.detach(), qs.detach(), pm, ps))
            kls_post.append(kl_gauss(qm, qs, pm.detach(), ps.detach()))
        kl_prior, kl_post = torch.stack(kls_prior, 1), torch.stack(kls_post, 1)
        kl = self.kl_balance * kl_prior.mean() + (1 - self.kl_balance) * kl_post.mean()
        return {"z": torch.stack(zs, 1), "kl": kl, "kl_raw": kl_prior.mean().detach()}

    def kl_loss(self, kl: Tensor) -> Tensor:
        """Free nats: no gradient pressure below the floor (the value never drops under it)."""
        return torch.clamp(kl, min=self.free_nats)

    def extra_loss(self, batch, ro, horizon: int) -> dict[str, Tensor]:
        f = self.filter(batch, horizon)
        z = f["z"]
        tgt = batch.target[:, :horizon]
        post_recon = self.rollout_loss(self.decode(z.reshape(-1, self.d_z)).reshape(tgt.shape), tgt)
        out = {"elbo_recon": post_recon, "kl": self.beta_kl * self.kl_loss(f["kl"])}
        if self.obs_mode == "pixels":  # reward from posterior states too (the prior rollout's is in loss())
            r = self.reward(z.reshape(-1, self.d_z), None).reshape(z.shape[:2])
            out["post_reward"] = (r - batch.rewards[:, :horizon]).pow(2).mean()
        return out
