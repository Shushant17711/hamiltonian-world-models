"""Vectorised torch SDF rasteriser: observations -> 64x64 grayscale frames in [0, 1] (Req 3.2).

Each env is drawn as a few primitives (segments with thickness, discs, boxes) in a fixed world
window centred on the origin, y up. A pixel's intensity is a smoothstep over a 1-pixel band
around the zero level set, so images vary smoothly with the state (anti-aliased). The union of
primitives is the max of their intensities. Runs on whatever device ``obs`` lives on; no display.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from hwm.envs.registry import make

# half-width of the square world window for each env (world units)
EXTENT = {"pendulum": 1.3, "cartpole": 3.2, "acrobot": 2.2, "orbit": 2.5}


@functools.cache
def _env(name: str):
    return make(name)


@functools.lru_cache(maxsize=16)
def _grid(size: int, extent: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Pixel-centre world coordinates, (size, size, 2); row 0 is the top of the image."""
    c = (torch.arange(size, dtype=dtype, device=device) + 0.5) / size * 2 - 1
    ys, xs = torch.meshgrid(-c * extent, c * extent, indexing="ij")
    return torch.stack([xs, ys], -1)


def _ink(d: torch.Tensor, px: float) -> torch.Tensor:
    """Smoothstep from 1 (inside, d <= -px/2) to 0 (outside, d >= px/2)."""
    t = (0.5 - d / px).clamp(0, 1)
    return t * t * (3 - 2 * t)


def sd_circle(P: torch.Tensor, c: torch.Tensor, r: float) -> torch.Tensor:
    """P (..., H, W, 2); c (..., 2) -> (..., H, W)."""
    return (P - c[..., None, None, :]).norm(dim=-1) - r


def sd_segment(P: torch.Tensor, a: torch.Tensor, b: torch.Tensor, r: float) -> torch.Tensor:
    a, b = a[..., None, None, :], b[..., None, None, :]
    pa, ba = P - a, b - a
    h = ((pa * ba).sum(-1) / (ba * ba).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (pa - h[..., None] * ba).norm(dim=-1) - r


def sd_box(P: torch.Tensor, c: torch.Tensor, half: tuple[float, float]) -> torch.Tensor:
    q = (P - c[..., None, None, :]).abs() - P.new_tensor(half)
    return q.clamp_min(0).norm(dim=-1) + q.max(-1).values.clamp_max(0)


def _polar(theta: torch.Tensor, length: float) -> torch.Tensor:
    """Offset of a link at angle theta from the downward vertical."""
    return torch.stack([length * torch.sin(theta), -length * torch.cos(theta)], -1)


def _shapes(name: str, obs: torch.Tensor, P: torch.Tensor) -> list[torch.Tensor]:
    env = _env(name)
    zero = torch.zeros_like(obs[..., :2])
    if name == "pendulum":
        bob = _polar(obs[..., 0], env.l)
        return [sd_segment(P, zero, bob, 0.05), sd_circle(P, bob, 0.15)]
    if name == "cartpole":
        cart = torch.stack([obs[..., 0], torch.zeros_like(obs[..., 0])], -1)
        bob = cart + _polar(obs[..., 1], env.l)
        return [sd_box(P, cart, (0.3, 0.15)), sd_segment(P, cart, bob, 0.06), sd_circle(P, bob, 0.15)]
    if name == "acrobot":
        elbow = _polar(obs[..., 0], env.l1)
        tip = elbow + _polar(obs[..., 0] + obs[..., 1], env.l2)
        return [
            sd_segment(P, zero, elbow, 0.08),
            sd_segment(P, elbow, tip, 0.08),
            sd_circle(P, tip, 0.12),
        ]
    if name == "orbit":
        return [sd_circle(P, zero, 0.15), sd_circle(P, obs[..., :2], 0.1)]
    raise KeyError(f"no renderer for env {name!r}")


def render(env_name: str, obs, size: int = 64) -> torch.Tensor:
    """Render observations (..., d_obs) (numpy or torch) to frames (..., size, size) in [0, 1]."""
    obs = torch.as_tensor(obs)
    if not obs.is_floating_point():
        obs = obs.float()
    lead = obs.shape[:-1]
    flat = obs.reshape(-1, obs.shape[-1])
    extent = EXTENT[env_name]
    P = _grid(size, extent, str(flat.device), flat.dtype)
    px = 2 * extent / size
    img = torch.stack([_ink(d, px) for d in _shapes(env_name, flat, P)]).amax(0)
    return img.reshape(*lead, size, size)


def to_numpy(img: torch.Tensor) -> np.ndarray:
    return img.detach().cpu().numpy()
