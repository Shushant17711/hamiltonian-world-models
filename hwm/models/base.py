"""The world-model contract shared by every model (design §4).

A model maps a context of observations to a latent ``z``, steps ``z`` forward one ``dt`` at a time
under actions, and decodes back to observations. Everything downstream (trainer, evaluation,
planner) talks to models only through this interface.

Conventions: state-mode observations are the *normalised* states the dataset yields; pixel-mode
observations are 64x64 frames in [0, 1]. Actions are scaled to [-1, 1] (``u / u_max``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from hwm.data.dataset import Batch

__all__ = ["Batch", "Rollout", "WorldModel"]

Tensor = torch.Tensor


@dataclass
class Rollout:
    z: Tensor  # (B, H+1, d_z)
    obs: Tensor  # (B, H+1, *obs) decoded observations; obs[:, 0] reconstructs the last context frame
    reward: Tensor | None = None  # (B, H) predicted rewards, if the model has a reward head


class WorldModel(nn.Module):
    """Base class. Subclasses implement ``encode``, ``step`` and ``decode``; the rest has defaults."""

    obs_mode: Literal["state", "pixels"] = "state"
    context: int = 1  # 1 for state, 3 for pixels

    def __init__(self, d_obs: int, d_u: int, d_z: int, obs_mode: str = "state", context: int | None = None):
        super().__init__()
        if obs_mode not in ("state", "pixels"):
            raise ValueError(f"obs_mode must be 'state' or 'pixels', got {obs_mode!r}")
        self.d_obs, self.d_u, self.d_z = d_obs, d_u, d_z
        self.obs_mode = obs_mode
        self.context = context if context is not None else (3 if obs_mode == "pixels" else 1)

    # --- to implement -----------------------------------------------------------------------
    def encode(self, ctx: Tensor) -> Tensor:
        """(B, k, *obs) -> z (B, d_z)."""
        raise NotImplementedError

    def step(self, z: Tensor, u: Tensor) -> Tensor:
        """Advance the latent by one dt under action u (B, d_u)."""
        raise NotImplementedError

    def decode(self, z: Tensor) -> Tensor:
        """(B, d_z) -> (B, *obs)."""
        raise NotImplementedError

    # --- optional heads -----------------------------------------------------------------------
    def reward(self, z: Tensor, u: Tensor) -> Tensor | None:
        """Learned reward r(z_{t+1}, u_t) (pixel mode); None when the env reward is used directly."""
        return None

    def energy(self, z: Tensor) -> Tensor | None:
        """Learned energy of a latent state, if the model has one (C, E)."""
        return None

    # --- defaults -----------------------------------------------------------------------------
    @property
    def obs_shape(self) -> tuple[int, ...]:
        return (64, 64) if self.obs_mode == "pixels" else (self.d_obs,)

    def rollout(self, ctx: Tensor, actions: Tensor) -> Rollout:
        """Open-loop rollout: encode the context, then step/decode under ``actions`` (B, H, d_u)."""
        z = self.encode(ctx)
        zs, rewards = [z], []
        for t in range(actions.shape[1]):
            z = self.step(z, actions[:, t])
            zs.append(z)
            r = self.reward(z, actions[:, t])
            if r is not None:
                rewards.append(r)
        Z = torch.stack(zs, dim=1)
        B, T1 = Z.shape[:2]
        obs = self.decode(Z.reshape(B * T1, -1)).reshape(B, T1, *self.obs_shape)
        return Rollout(z=Z, obs=obs, reward=torch.stack(rewards, dim=1) if rewards else None)

    def loss(self, batch: Batch, horizon: int) -> tuple[Tensor, dict[str, float]]:
        """Default multi-step loss: MSE of the open-loop rollout against the targets (+ reward MSE)."""
        H = min(horizon, batch.actions.shape[1])
        ro = self.rollout(batch.ctx, batch.actions[:, :H])
        pred = self.rollout_loss(ro.obs[:, 1:], batch.target[:, :H])
        recon = self.rollout_loss(ro.obs[:, 0], batch.ctx[:, -1])
        total = pred + recon
        logs = {"pred": pred.item(), "recon": recon.item()}
        if ro.reward is not None:
            rew = (ro.reward - batch.rewards[:, :H]).pow(2).mean()
            total = total + rew
            logs["reward"] = rew.item()
        for name, term in self.extra_loss(batch, ro, H).items():
            total = total + term
            logs[name] = term.item()
        logs["loss"] = total.item()
        return total, logs

    def backward(self, batch: Batch, horizon: int) -> dict[str, float]:
        """Accumulate gradients of ``loss`` (the trainer zeroes them first and steps afterwards)."""
        loss, logs = self.loss(batch, horizon)
        loss.backward()
        return logs

    def extra_loss(self, batch: Batch, ro: Rollout, horizon: int) -> dict[str, Tensor]:
        """Model-specific weighted loss terms added to the default loss (none by default)."""
        return {}

    def prepare(self, normaliser, train_obs: Tensor) -> None:
        """Called once by the trainer before training with the train-split normaliser and raw
        train observations (N, T+1, d_obs). Anything stored must be a buffer so checkpoints keep it."""

    # --- helpers for models whose losses need physical (un-normalised) states ----------------------
    def _register_normaliser(self) -> None:
        """Create ``obs_mean`` / ``obs_std`` buffers (identity until ``_load_normaliser``)."""
        self.register_buffer("obs_mean", torch.zeros(self.d_obs))
        self.register_buffer("obs_std", torch.ones(self.d_obs))

    def _load_normaliser(self, normaliser) -> None:
        self.obs_mean.copy_(normaliser.mean.to(self.obs_mean.device))
        self.obs_std.copy_(normaliser.std.to(self.obs_std.device))

    def physical(self, x: Tensor) -> Tensor:
        """Normalised state observation -> physical (q, q_dot)."""
        return x * self.obs_std + self.obs_mean

    def target_stacks(self, ctx: Tensor, target: Tensor) -> Tensor:
        """(B, H, k, *obs): for every target frame, it and the k-1 frames before it (encoder inputs)."""
        k, H = self.context, target.shape[1]
        seq = torch.cat([ctx, target], dim=1)
        return torch.stack([seq[:, t + 1 : t + 1 + k] for t in range(H)], dim=1)

    @staticmethod
    def rollout_loss(pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).pow(2).mean()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
