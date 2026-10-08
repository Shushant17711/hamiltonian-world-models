"""Model B (PINN): physics residual, contract, tiny train and exact resume (Req 4.2)."""

import numpy as np
import pytest
import torch

from hwm.config import Config
from hwm.data.dataset import Normaliser
from hwm.data.generate import generate, load_split
from hwm.envs import make
from hwm.models import build
from hwm.train.trainer import Trainer
from scripts.train import build_run_config
from tests.test_data_generate import CONFIGS, TINY
from tests.test_trainer import FAST, ROOT, _records


def _pinn(env_name):
    m = build(Config({"env": env_name, "model": {"name": "pinn", "hidden": 16, "layers": 1}}))
    return m.double()


def _true_pairs(env, N=32, seed=0):
    rng = np.random.default_rng(seed)
    qp = env.sample_band(rng, (0.2, 0.5) if env.name != "orbit" else (0.8, 1.2), N)
    u = rng.uniform(-env.u_max, env.u_max, (N, env.d_u))
    x0, x1 = env.qp_to_obs(qp), env.qp_to_obs(env.step(qp, u))
    return x0, x1, u / env.u_max


@pytest.mark.parametrize("name", ["pendulum", "cartpole", "acrobot", "orbit"])
def test_residual_small_on_truth_and_large_when_perturbed(name):
    env = make(name)
    x0, x1, a = _true_pairs(env)
    m = _pinn(name)
    norm = Normaliser.fit(np.concatenate([x0, x1]))
    m.prepare(norm, torch.as_tensor(np.stack([x0, x1], 1)))
    z0, z1 = (norm.norm(torch.as_tensor(x)).double() for x in (x0, x1))
    a = torch.as_tensor(a)
    r_true = m.physics_residual(z0, z1, a).pow(2).mean().item()
    z1_bad = z1 + 0.05 * torch.randn(z1.shape, generator=torch.Generator().manual_seed(1), dtype=z1.dtype)
    r_bad = m.physics_residual(z0, z1_bad, a).pow(2).mean().item()
    # Simpson + Hermite midpoint: true transitions leave only an O(dt^5) quadrature residual
    # (acrobot, the stiffest, sits at ~6e-6: rms 2.5e-3 std per step, far below any model's error)
    assert r_true < 2e-5 and r_true < 1e-2 * r_bad, (r_true, r_bad)
    assert r_bad > 1e-4


def test_contract_and_extra_term_has_gradients():
    m = build(Config({"env": "pendulum", "model": {"name": "pinn", "hidden": 16, "layers": 1}}))
    with pytest.raises(ValueError):
        build(Config({"env": "pendulum", "obs_mode": "pixels", "model": {"name": "pinn"}}))
    from tests.test_models_base import _batch

    b = _batch(d_obs=2, d_u=1)
    loss, logs = m.loss(b, horizon=4)
    assert {"loss", "pred", "recon", "phys"} <= set(logs) and logs["phys"] > 0
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    from hwm.config import load_config

    cfg = load_config(CONFIGS / "pendulum.yaml", TINY + ["sizes.train=[16,30]", "sizes.val=[8,30]"])
    return generate(cfg, tmp_path_factory.mktemp("data"))


def _cfg():
    return build_run_config(str(ROOT / "configs/model/pinn.yaml"), FAST + ["model.collocation=32"])


def test_tiny_train_lowers_loss_and_resumes_exactly(tmp_path, data_dir):
    full = Trainer(_cfg(), tmp_path / "a", data_dir)
    v0 = full.validate()
    assert full.fit()["best_val_nmse"] < 0.5 * v0
    assert all("train_phys" in r for r in _records(tmp_path / "a"))
    # box buffers come from the train split and are checkpointed
    raw = torch.as_tensor(load_split(data_dir, "train")["obs"], dtype=torch.float32)
    z = full.normaliser.norm(raw.reshape(-1, raw.shape[-1]))
    torch.testing.assert_close(full.model.box_lo.cpu(), z.min(0).values)
    part = Trainer(_cfg(), tmp_path / "b", data_dir)
    part.fit(stop_after=25)
    resumed = Trainer(_cfg(), tmp_path / "b", data_dir)
    resumed.fit()
    for (k, a), b in zip(full.model.state_dict().items(), resumed.model.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b, msg=k)
