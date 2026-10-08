"""Training windows over generated trajectories (Req 2.1, 3.1, 3.3; design §3).

A window starting at step s of a trajectory is::

    ctx     obs[s-k+1 .. s]      (k context frames; k = 1 for states, 3 for pixels)
    actions act[s .. s+H-1]
    target  obs[s+1 .. s+H]
    rewards reward(obs[t+1], act[t]) for t in s .. s+H-1

State-mode ``ctx``/``target`` are normalised with statistics fitted on the train split only;
pixel mode renders 64x64 frames on the fly (on the dataset's device). ``target_obs`` is always the
raw state observation, for evaluation and reward heads. Actions are scaled by ``u_max``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import torch

from hwm.data.generate import load_split
from hwm.envs import make
from hwm.envs.render import render


@dataclass
class Batch:
    ctx: torch.Tensor  # (B, k, d_obs) normalised, or (B, k, 64, 64)
    actions: torch.Tensor  # (B, H, d_u) in [-1, 1]
    target: torch.Tensor  # (B, H, d_obs) normalised, or (B, H, 64, 64)
    rewards: torch.Tensor  # (B, H)
    passive: torch.Tensor  # (B,) bool
    target_obs: torch.Tensor  # (B, H, d_obs) raw state observations

    @staticmethod
    def collate(items: list[Batch]) -> Batch:
        """``collate_fn`` for ``torch.utils.data.DataLoader`` over single windows."""
        return Batch(**{f.name: torch.stack([getattr(b, f.name) for b in items]) for f in fields(Batch)})

    def to(self, device) -> Batch:
        return Batch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


class Normaliser:
    """Per-dimension affine observation normaliser; fit on train only and saved with each model."""

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @classmethod
    def fit(cls, obs, min_std: float = 1e-6) -> Normaliser:
        x = torch.as_tensor(obs, dtype=torch.float64).reshape(-1, obs.shape[-1])
        return cls(x.mean(0), x.std(0).clamp_min(min_std))

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> Normaliser:
        return cls.fit(load_split(data_dir, "train")["obs"])

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x)) / self.std.to(x)

    def denorm(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z) + self.mean.to(z)

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, d) -> Normaliser:
        return cls(d["mean"], d["std"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict()))

    @classmethod
    def load(cls, path: str | Path) -> Normaliser:
        return cls.from_state_dict(json.loads(Path(path).read_text()))


class WindowDataset(torch.utils.data.Dataset):
    """Every (trajectory, start) window of one split; also supports fast random batches via ``sample``."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        horizon: int,
        context: int = 1,
        obs_mode: Literal["state", "pixels"] = "state",
        normaliser: Normaliser | None = None,
        device: str | torch.device = "cpu",
        env_name: str | None = None,
    ):
        data_dir = Path(data_dir)
        self.env_name = env_name or data_dir.parent.name
        self.env = make(self.env_name)
        d = load_split(data_dir, split)
        self.device = torch.device(device)
        self.obs = torch.as_tensor(d["obs"], dtype=torch.float32, device=self.device)
        self.act = torch.as_tensor(d["act"] / self.env.u_max, dtype=torch.float32, device=self.device)
        self.passive = torch.as_tensor(d["passive"], device=self.device)
        self.normaliser = normaliser or Normaliser.from_dir(data_dir)
        self.obs_mode = obs_mode
        self.horizon, self.context = horizon, context
        N, T1, _ = self.obs.shape
        self.n_starts = T1 - horizon - context + 1  # s in [k-1, T-H]
        if self.n_starts < 1:
            raise ValueError(f"trajectories of {T1 - 1} steps are too short for H={horizon}, k={context}")
        self.n_traj = N
        # rewards for every transition, computed once: r_t = reward(obs_{t+1}, u_t)
        self.rew = self.env.reward(self.obs[:, 1:].double(), self.act.double() * self.env.u_max).float()

    def set_horizon(self, horizon: int) -> None:
        """Horizon curriculum: change H in place (the trainer grows it over training)."""
        n = self.obs.shape[1] - horizon - self.context + 1
        if n < 1:
            raise ValueError(f"horizon {horizon} too long")
        self.horizon, self.n_starts = horizon, n

    def __len__(self) -> int:
        return self.n_traj * self.n_starts

    def __getitem__(self, i: int) -> Batch:
        b = self._gather(torch.tensor([i // self.n_starts]), torch.tensor([i % self.n_starts]))
        return Batch(**{f.name: getattr(b, f.name)[0] for f in fields(b)})

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        dev = generator.device if generator is not None else "cpu"  # a CUDA generator draws on CUDA
        idx = torch.randint(len(self), (batch_size,), generator=generator, device=dev)
        return self._gather(idx // self.n_starts, idx % self.n_starts)

    def _gather(self, traj: torch.Tensor, start: torch.Tensor) -> Batch:
        traj, start = traj.to(self.device), start.to(self.device)
        k, H = self.context, self.horizon
        s = start + (k - 1)  # index of the last context frame
        ctx_t = s[:, None] + torch.arange(-k + 1, 1, device=self.device)
        tgt_t = s[:, None] + torch.arange(1, H + 1, device=self.device)
        act_t = s[:, None] + torch.arange(H, device=self.device)
        tr = traj[:, None]
        ctx_obs, target_obs = self.obs[tr, ctx_t], self.obs[tr, tgt_t]
        if self.obs_mode == "pixels":
            ctx, target = render(self.env_name, ctx_obs), render(self.env_name, target_obs)
        else:
            ctx, target = self.normaliser.norm(ctx_obs), self.normaliser.norm(target_obs)
        return Batch(
            ctx=ctx,
            actions=self.act[tr, act_t],
            target=target,
            rewards=self.rew[tr, act_t],
            passive=self.passive[traj],
            target_obs=target_obs,
        )
