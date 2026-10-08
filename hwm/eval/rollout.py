"""Open-loop rollout error (design §7, Req 7.1; PREREGISTRATION.md §4).

nMSE(h) is the error *at* step h (not averaged over 1..h):

    nMSE(h) = mean over trajectories and features of (phi(o_hat_h) - phi(o_h))^2 / Var_train(phi(o))

phi replaces every angle coordinate by (cos, sin), so wrapping never inflates the error. Each
trajectory's error is capped at ``NMSE_CAP``; a non-finite error counts as the cap, and the number of
such diverged trajectories is reported next to every value.

State mode only for now; pixel-mode evaluation arrives with task 10.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from hwm.data.dataset import Normaliser
from hwm.envs.base import Env
from hwm.eval.thresholds import NMSE_CAP
from hwm.models.base import WorldModel


def feature_map(env: Env, obs: Tensor) -> Tensor:
    """(..., d_obs) raw observation -> (..., d_phi): angle coordinates become (cos, sin)."""
    cols = []
    for i in range(obs.shape[-1]):
        if i in env.angle_dims:
            cols += [torch.cos(obs[..., i]), torch.sin(obs[..., i])]
        else:
            cols.append(obs[..., i])
    return torch.stack(cols, dim=-1)


def feature_var(env: Env, train_obs: np.ndarray | Tensor) -> Tensor:
    """Per-feature variance of phi(o) over the train split, float64."""
    x = torch.as_tensor(train_obs, dtype=torch.float64)
    return feature_map(env, x.reshape(-1, x.shape[-1])).var(0)


def model_dtype(model: WorldModel) -> torch.dtype:
    p = next(model.parameters(), None)
    return p.dtype if p is not None else torch.float64


def model_device(model: WorldModel) -> torch.device:
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


@torch.no_grad()
def rollout_nmse(
    model: WorldModel,
    normaliser: Normaliser,
    env: Env,
    data: dict[str, np.ndarray],
    horizons: list[int] | tuple[int, ...],
    var: Tensor,
    cap: float = NMSE_CAP,
    chunk: int = 256,
) -> dict[int, dict[str, float]]:
    """{h: {"nmse", "diverged", "n"}} on a split (``obs`` (N, T+1, d), ``act`` (N, T, d_u)).

    The context is the first ``k`` frames; prediction h targets ``obs[:, k-1+h]``.
    """
    if model.obs_mode != "state":
        raise NotImplementedError("pixel-mode rollout error arrives with task 10")
    k, T = model.context, data["act"].shape[1]
    hs = [h for h in horizons if k - 1 + h <= T]
    if not hs:
        return {}
    H = max(hs)
    dev, dt = model_device(model), model_dtype(model)
    var = var.to(dev)
    obs = torch.as_tensor(data["obs"], dtype=torch.float64)
    act = torch.as_tensor(data["act"], dtype=torch.float64) / env.u_max
    errs = []
    for i in range(0, obs.shape[0], chunk):
        o = obs[i : i + chunk].to(dev)
        ctx = normaliser.norm(o[:, :k].to(dt))
        ro = model.rollout(ctx, act[i : i + chunk, k - 1 : k - 1 + H].to(dev, dt))
        pred = normaliser.denorm(ro.obs[:, hs].double())  # (B, len(hs), d)
        true = o[:, [k - 1 + h for h in hs]]
        e = ((feature_map(env, pred) - feature_map(env, true)) ** 2 / var).mean(-1)  # (B, len(hs))
        errs.append(e.cpu())
    e = torch.cat(errs)
    diverged = ~torch.isfinite(e) | (e >= cap)
    e = torch.where(torch.isfinite(e), e, torch.full_like(e, cap)).clamp(max=cap)
    return {
        h: {"nmse": e[:, j].mean().item(), "diverged": int(diverged[:, j].sum()), "n": e.shape[0]}
        for j, h in enumerate(hs)
    }
