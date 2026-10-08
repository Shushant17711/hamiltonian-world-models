"""World-model contract, nets and registry (design §4)."""

import pytest
import torch
from torch import nn

from hwm.config import Config
from hwm.models import Batch, WorldModel, build, get, register
from hwm.models.nets import CNNDecoder, CNNEncoder, StateDecoder, StateEncoder, mlp, n_params


class _Linear(WorldModel):
    """Minimal model used to exercise the default rollout/loss."""

    def __init__(self, cfg, env, obs_mode="state"):
        super().__init__(env.d_obs, env.d_u, d_z=4, obs_mode=obs_mode)
        self.enc = StateEncoder(env.d_obs, 4, hidden=16, layers=1)
        self.dyn = nn.Linear(4 + env.d_u, 4)
        self.dec = StateDecoder(4, env.d_obs, hidden=16, layers=1)

    def encode(self, ctx):
        return self.enc(ctx)

    def step(self, z, u):
        return z + self.dyn(torch.cat([z, u], -1))

    def decode(self, z):
        return self.dec(z)


register("_linear_test")(_Linear)


def _batch(B=5, H=7, d_obs=2, d_u=1):
    return Batch(
        ctx=torch.randn(B, 1, d_obs),
        actions=torch.rand(B, H, d_u) * 2 - 1,
        target=torch.randn(B, H, d_obs),
        rewards=torch.randn(B, H),
        passive=torch.zeros(B, dtype=torch.bool),
        target_obs=torch.randn(B, H, d_obs),
    )


def test_default_rollout_and_loss_contract():
    m = build(Config({"env": "pendulum", "model": {"name": "_linear_test"}}))
    b = _batch()
    ro = m.rollout(b.ctx, b.actions)
    assert ro.z.shape == (5, 8, 4) and ro.obs.shape == (5, 8, 2) and ro.reward is None
    loss, logs = m.loss(b, horizon=3)
    assert loss.ndim == 0 and {"loss", "pred", "recon"} <= set(logs)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    assert m.context == 1 and m.obs_shape == (2,)


def test_registry_errors_and_lazy_names():
    with pytest.raises(KeyError, match="unknown model"):
        get("nope")
    with pytest.raises(ValueError):
        _Linear.__mro__[1](2, 1, 4, obs_mode="video")


def test_mlp_layers():
    net = mlp(3, 2, hidden=8, layers=2, layernorm=True)
    assert net(torch.randn(4, 3)).shape == (4, 2)
    assert sum(isinstance(x, nn.LayerNorm) for x in net) == 2


def test_cnn_shapes_and_range():
    enc, dec = CNNEncoder(d_z=6, context=3), CNNDecoder(d_z=6)
    x = torch.rand(2, 3, 64, 64)
    z = enc(x)
    assert z.shape == (2, 6)
    y = dec(z)
    assert y.shape == (2, 64, 64) and 0 <= y.min() and y.max() <= 1
    assert dec(torch.randn(2, 5, 6)).shape == (2, 5, 64, 64)
    assert n_params(enc) + n_params(dec) < 5_000_000
