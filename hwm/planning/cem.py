"""Batched cross-entropy method for MPC (design §8, Req 8.1).

Plans action sequences (horizon T, d_u) in [-1, 1] that minimise ``cost_fn``. Every iteration
samples ``population`` sequences per problem from a diagonal Gaussian, evaluates all of them in one
call, refits mean/std to the ``elites`` best with momentum
(new = momentum * old + (1 - momentum) * fit). The previous plan, shifted by one step, warm-starts
the next call; its last step repeats the previous last step.

``n`` independent problems (e.g. 10 evaluation episodes run in lock-step) are solved together:
``cost_fn`` then receives (n * P, T, d_u), problem-major, and returns (n * P,). With ``n = None``
(default) the API is unbatched: ``plan`` returns (T, d_u) and ``act`` returns (d_u,).
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
        self,
        d_u: int,
        cfg: CEMConfig | None = None,
        device="cpu",
        generator: torch.Generator | None = None,
        n: int | None = None,
    ):
        self.cfg = cfg or CEMConfig()
        if self.cfg.elites > self.cfg.population:
            raise ValueError("elites must not exceed the population")
        self.d_u, self.device, self.gen = d_u, torch.device(device), generator
        self.n, self.batched = (n or 1), n is not None
        self.reset()

    def reset(self) -> None:
        """Forget the warm start (call at the start of every episode)."""
        self._mean = torch.zeros(self.n, self.cfg.horizon, self.d_u, device=self.device)

    @property
    def mean(self) -> Tensor:
        return self._mean if self.batched else self._mean[0]

    @mean.setter
    def mean(self, value: Tensor) -> None:
        self._mean = value.to(self.device) if self.batched else value.to(self.device)[None]

    def shift(self) -> None:
        """Warm start for the next time step: drop the executed action, repeat the last one."""
        self._mean = torch.cat([self._mean[:, 1:], self._mean[:, -1:]], dim=1)

    @torch.no_grad()
    def plan(self, cost_fn: CostFn) -> tuple[Tensor, dict[str, float]]:
        """Optimise and return (mean sequences, stats). Does not shift."""
        c, n = self.cfg, self.n
        mean = self._mean.clone()  # (n, T, d_u)
        std = torch.full_like(mean, c.init_std)
        best = torch.full((n,), float("inf"), device=self.device)
        for _ in range(c.iterations):
            noise = torch.randn(n, c.population, *mean.shape[1:], device=self.device, generator=self.gen)
            cand = (mean[:, None] + std[:, None] * noise).clamp(-1.0, 1.0)
            cand[:, 0] = mean.clamp(-1.0, 1.0)  # always evaluate the current mean
            cost = cost_fn(cand.reshape(n * c.population, *mean.shape[1:])).reshape(n, c.population)
            cost = torch.where(torch.isfinite(cost), cost, torch.full_like(cost, float("inf")))
            val, idx = torch.topk(cost, c.elites, dim=1, largest=False)
            elite = torch.gather(cand, 1, idx[:, :, None, None].expand(-1, -1, *mean.shape[1:]))
            best = torch.minimum(best, val[:, 0])
            mean = c.momentum * mean + (1 - c.momentum) * elite.mean(1)
            std = (c.momentum * std + (1 - c.momentum) * elite.std(1, unbiased=False)).clamp_min(c.min_std)
        self._mean = mean
        out = mean.clamp(-1.0, 1.0)
        stats = {"best_cost": best.mean().item(), "final_std": std.mean().item()}
        return (out if self.batched else out[0]), stats

    def act(self, cost_fn: CostFn) -> tuple[Tensor, dict[str, float]]:
        """MPC: plan, return the first action(s), and shift the warm start."""
        seq, stats = self.plan(cost_fn)
        a = (seq[:, 0] if self.batched else seq[0]).clone()
        self.shift()
        return a, stats
