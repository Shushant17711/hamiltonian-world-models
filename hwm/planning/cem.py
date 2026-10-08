"""Batched cross-entropy method for MPC (design §8, Req 8.1).

Plans an action sequence (horizon T, d_u) in [-1, 1] that minimises ``cost_fn``. Every iteration
samples ``population`` sequences from a diagonal Gaussian, evaluates them in one call
(``cost_fn((P, T, d_u)) -> (P,)``), refits mean/std to the ``elites`` best, with momentum
(new = momentum * old + (1 - momentum) * fit). The mean of the previous plan, shifted by one step,
warm-starts the next call; its last step repeats the previous last step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

CostFn = Callable[[Tensor], Tensor]


@dataclass
class CEMConfig:
    horizon: int = 30
    population: int = 400
    elites: int = 40
    iterations: int = 5
    momentum: float = 0.1
    init_std: float = 0.5
    min_std: float = 0.05


class CEM:
    def __init__(
        self, d_u: int, cfg: CEMConfig | None = None, device="cpu", generator: torch.Generator | None = None
    ):
        self.cfg = cfg or CEMConfig()
        if self.cfg.elites > self.cfg.population:
            raise ValueError("elites must not exceed the population")
        self.d_u, self.device, self.gen = d_u, torch.device(device), generator
        self.reset()

    def reset(self) -> None:
        """Forget the warm start (call at the start of every episode)."""
        self.mean = torch.zeros(self.cfg.horizon, self.d_u, device=self.device)

    def shift(self) -> None:
        """Warm start for the next time step: drop the executed action, repeat the last one."""
        self.mean = torch.cat([self.mean[1:], self.mean[-1:]], dim=0)

    @torch.no_grad()
    def plan(self, cost_fn: CostFn) -> tuple[Tensor, dict[str, float]]:
        """Optimise and return (best mean sequence (T, d_u), stats). Does not shift."""
        c = self.cfg
        mean = self.mean.clone()
        std = torch.full_like(mean, c.init_std)
        best_cost = float("inf")
        for _ in range(c.iterations):
            noise = torch.randn(c.population, *mean.shape, device=self.device, generator=self.gen)
            cand = (mean + std * noise).clamp(-1.0, 1.0)
            cand[0] = mean.clamp(-1.0, 1.0)  # always evaluate the current mean
            cost = cost_fn(cand)
            cost = torch.where(torch.isfinite(cost), cost, torch.full_like(cost, float("inf")))
            idx = torch.topk(cost, c.elites, largest=False).indices
            elite = cand[idx]
            best_cost = min(best_cost, cost[idx[0]].item())
            mean = c.momentum * mean + (1 - c.momentum) * elite.mean(0)
            std = (c.momentum * std + (1 - c.momentum) * elite.std(0, unbiased=False)).clamp_min(c.min_std)
        self.mean = mean
        return mean.clamp(-1.0, 1.0), {"best_cost": best_cost, "final_std": std.mean().item()}

    def act(self, cost_fn: CostFn) -> tuple[Tensor, dict[str, float]]:
        """MPC: plan, return the first action (d_u,), and shift the warm start."""
        seq, stats = self.plan(cost_fn)
        a = seq[0].clone()
        self.shift()
        return a, stats
