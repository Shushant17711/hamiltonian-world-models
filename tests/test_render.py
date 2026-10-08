"""Torch SDF renderer (Req 3.2)."""

import time

import numpy as np
import pytest
import torch

from hwm.envs import make, names
from hwm.envs.render import render


def _obs(name, N, seed=0):
    env = make(name)
    band = (0.8, 1.2) if name == "orbit" else (0.1, 0.6)
    return env.qp_to_obs(env.sample_band(np.random.default_rng(seed), band, N))


@pytest.mark.parametrize("name", names())
def test_shape_range_and_leading_dims(name):
    obs = _obs(name, 6).reshape(2, 3, -1)
    img = render(name, obs)
    assert img.shape == (2, 3, 64, 64)
    assert img.min() >= 0 and img.max() <= 1
    assert img.max() > 0.99  # something is drawn
    assert img.mean() < 0.2  # ... on a mostly empty background


@pytest.mark.parametrize("name", names())
def test_different_states_give_different_images(name):
    img = render(name, _obs(name, 8))
    diff = (img[:, None] - img[None]).abs().flatten(2).sum(-1)
    off_diag = diff[~torch.eye(8, dtype=torch.bool)]
    assert off_diag.min() > 1.0


def _centroid(img):
    H, W = img.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing="ij")
    m = img.sum((-1, -2))
    return (img * xs).sum((-1, -2)) / m, (img * ys).sum((-1, -2)) / m


def test_pendulum_centroid_follows_theta():
    th = torch.tensor([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=torch.float64)
    obs = torch.stack([th, torch.zeros_like(th)], -1)
    cx, cy = _centroid(render("pendulum", obs))
    c = 31.5
    assert cy[0] > c + 5 and abs(cx[0] - c) < 1  # hanging: below centre (image y down)
    assert cx[1] > c + 5 and abs(cy[1] - c) < 1  # theta = pi/2: to the right
    assert cy[2] < c - 5 and abs(cx[2] - c) < 1  # upright: above
    assert cx[3] < c - 5


def test_cartpole_cart_moves_with_x():
    obs = torch.zeros(2, 4, dtype=torch.float64)
    obs[1, 0] = 2.0
    cx, _ = _centroid(render("cartpole", obs))
    assert cx[1] - cx[0] == pytest.approx(2.0 / 6.4 * 64, abs=0.5)


def test_smooth_in_state():
    """Anti-aliasing: a sub-pixel move changes the image a little, not by whole pixels."""
    obs = torch.tensor([[0.3, 0.0], [0.3 + 1e-3, 0.0]], dtype=torch.float64)
    img = render("pendulum", obs)
    d = (img[0] - img[1]).abs()
    assert 0 < d.max() < 0.2


def test_float32_and_numpy_input():
    obs = _obs("acrobot", 4)
    a = render("acrobot", obs)
    b = render("acrobot", torch.as_tensor(obs, dtype=torch.float32))
    assert b.dtype == torch.float32
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-3)


def test_thousand_frames_under_a_second_on_cpu():
    obs = torch.as_tensor(_obs("acrobot", 1000), dtype=torch.float32)
    render("acrobot", obs[:10])  # warm up
    t0 = time.perf_counter()
    render("acrobot", obs)
    assert time.perf_counter() - t0 < 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu():
    obs = torch.as_tensor(_obs("cartpole", 16), dtype=torch.float32)
    np.testing.assert_allclose(
        render("cartpole", obs.cuda()).cpu().numpy(), render("cartpole", obs).numpy(), atol=1e-5
    )
