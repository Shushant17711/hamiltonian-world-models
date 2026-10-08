"""Shared network building blocks (design §5): MLPs, state encoder/decoder, 64x64 CNN encoder/decoder."""

from __future__ import annotations

import torch
from torch import nn


def mlp(
    d_in: int,
    d_out: int,
    hidden: int = 256,
    layers: int = 3,
    layernorm: bool = False,
    act: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    """``layers`` hidden layers of width ``hidden`` (SiLU, optional LayerNorm), then a linear head."""
    mods: list[nn.Module] = []
    d = d_in
    for _ in range(layers):
        mods.append(nn.Linear(d, hidden))
        if layernorm:
            mods.append(nn.LayerNorm(hidden))
        mods.append(act())
        d = hidden
    mods.append(nn.Linear(d, d_out))
    return nn.Sequential(*mods)


class StateEncoder(nn.Module):
    """(B, k, d_obs) context of normalised states -> (B, d_z)."""

    def __init__(
        self, d_obs: int, d_z: int, context: int = 1, hidden: int = 256, layers: int = 3, layernorm=False
    ):
        super().__init__()
        self.net = mlp(d_obs * context, d_z, hidden, layers, layernorm)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx.flatten(1))


class StateDecoder(nn.Module):
    """(..., d_z) -> (..., d_obs) normalised state."""

    def __init__(self, d_z: int, d_obs: int, hidden: int = 256, layers: int = 3, layernorm=False):
        super().__init__()
        self.net = mlp(d_z, d_obs, hidden, layers, layernorm)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


CNN_CHANNELS = (32, 64, 128, 256)


class CNNEncoder(nn.Module):
    """(B, k, 64, 64) stacked grayscale frames -> (B, d_z): 4 stride-2 convs (32-64-128-256), then linear."""

    def __init__(self, d_z: int, context: int = 3, channels=CNN_CHANNELS):
        super().__init__()
        mods: list[nn.Module] = []
        c = context
        for c_out in channels:
            mods += [nn.Conv2d(c, c_out, 4, stride=2, padding=1), nn.SiLU()]
            c = c_out
        self.conv = nn.Sequential(*mods)
        self.head = nn.Linear(c * 4 * 4, d_z)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(frames).flatten(1))


class CNNDecoder(nn.Module):
    """(..., d_z) -> (..., 64, 64) frame in [0, 1]: linear to 256x4x4, then 4 transposed convs."""

    def __init__(self, d_z: int, channels=CNN_CHANNELS):
        super().__init__()
        rev = tuple(reversed(channels))
        self.c0 = rev[0]
        self.head = nn.Linear(d_z, self.c0 * 4 * 4)
        mods: list[nn.Module] = []
        for c_in, c_out in zip(rev, rev[1:] + (1,), strict=True):
            mods.append(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1))
            if c_out != 1:
                mods.append(nn.SiLU())
        self.deconv = nn.Sequential(*mods)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        lead = z.shape[:-1]
        x = self.head(z.reshape(-1, z.shape[-1])).view(-1, self.c0, 4, 4)
        return torch.sigmoid(self.deconv(x)).view(*lead, 64, 64)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
