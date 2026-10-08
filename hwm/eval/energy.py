"""Energy drift over long passive rollouts (design §7, Req 7.2; PREREGISTRATION.md §4).

From each passive initial state, roll the model out ``steps`` steps with u = 0, decode every step,
map to (q, p) with the env's ``obs_to_qp`` and evaluate the *true* Hamiltonian:

    drift_i = max_t |H(x_t) - H(x_0)| / E_ref,      drift = median_i drift_i

x_0 is the model's own decoding of the initial state. A rollout that produces a non-finite value has
drift = inf for that initial state. Models with a learned energy (C, E) also get the same statistic
for E(z_t), relative to |E(z_0)|; it is logged, never used in a verdict.

Only running maxima are kept, so memory does not grow with ``steps``; batches are chunked.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from hwm.data.dataset import Normaliser
from hwm.envs.base import Env
from hwm.eval.rollout import model_device, model_dtype
from hwm.eval.thresholds import DRIFT_STEPS
from hwm.models.base import WorldModel


def _update(worst: Tensor, dev: Tensor) -> Tensor:
    dev = torch.where(torch.isfinite(dev), dev, torch.full_like(dev, math.inf))
    return torch.maximum(worst, dev)


def _summary(per_state: Tensor) -> dict:
    v = per_state.cpu().double().numpy()
    return {
        "median": float(np.median(v)),  # absorbs inf unless more than half diverge
        "diverged": int((~np.isfinite(v)).sum()),
        "n": len(v),
        "per_state": [float(x) for x in v],
    }


@torch.no_grad()
def energy_drift(
    model: WorldModel,
    normaliser: Normaliser,
    env: Env,
    x0: np.ndarray | Tensor,
    steps: int = DRIFT_STEPS,
    chunk: int = 512,
) -> dict[str, dict | None]:
    """{"true": summary, "learned": summary | None} for raw initial observations ``x0`` (N, d_obs)."""
    if model.obs_mode != "state":
        raise NotImplementedError("pixel-mode energy drift arrives with task 10")
    dev, dt = model_device(model), model_dtype(model)
    x0 = torch.as_tensor(x0, dtype=torch.float64)
    true_parts, learned_parts = [], []
    for i in range(0, x0.shape[0], chunk):
        ctx = normaliser.norm(x0[i : i + chunk].to(dev, dt))[:, None]
        z = model.encode(ctx)
        u = torch.zeros(z.shape[0], env.d_u, device=dev, dtype=dt)

        def true_H(z: Tensor) -> Tensor:
            x = normaliser.denorm(model.decode(z).double())
            return env.energy(env.obs_to_qp(x))

        H0, E0 = true_H(z), model.energy(z)
        worst = torch.zeros_like(H0)
        worst_E = None if E0 is None else torch.zeros_like(E0, dtype=torch.float64)
        for _ in range(steps):
            z = model.step(z, u)
            worst = _update(worst, (true_H(z) - H0).abs())
            if worst_E is not None:
                worst_E = _update(worst_E, ((model.energy(z) - E0).abs() / E0.abs()).double())
            if not torch.isfinite(worst).any():  # everything diverged; nothing left to measure
                break
        true_parts.append(worst / env.E_ref)
        if worst_E is not None:
            learned_parts.append(worst_E)
    return {
        "true": _summary(torch.cat(true_parts)),
        "learned": _summary(torch.cat(learned_parts)) if learned_parts else None,
    }
