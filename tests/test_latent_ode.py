"""Model C (latent Neural ODE + energy penalty): energy terms, contract, tiny train (Req 4.3)."""

import numpy as np
import pytest
import torch

from hwm.config import Config
from hwm.data.dataset import Batch, Normaliser
from hwm.envs import make
from hwm.models import build
from hwm.train.trainer import Trainer
from scripts.train import build_run_config
from tests.test_pinn import data_dir  # noqa: F401  (shared tiny pendulum dataset fixture)
from tests.test_trainer import FAST, ROOT, _records


def _model(**kw):
    return build(
        Config({"env": "pendulum", "model": {"name": "latent_ode", "hidden": 16, "layers": 1, **kw}})
    )


def _batch(passive, B=6, H=5):
    g = torch.Generator().manual_seed(0)
    return Batch(
        ctx=torch.randn(B, 1, 2, generator=g),
        actions=torch.rand(B, H, 1, generator=g) * 2 - 1,
        target=torch.randn(B, H, 2, generator=g),
        rewards=torch.randn(B, H, generator=g),
        passive=torch.as_tensor(passive),
        target_obs=torch.randn(B, H, 2, generator=g),
    )


def test_contract_and_latent_size():
    m = _model()
    assert m.d_z == 2 and _model(d_z=4).d_z == 4
    with pytest.raises(ValueError):
        build(Config({"env": "pendulum", "obs_mode": "pixels", "model": {"name": "latent_ode"}}))
    b = _batch([True, False] * 3)
    ro = m.rollout(b.ctx, b.actions)
    assert ro.z.shape == (6, 6, 2) and ro.obs.shape == (6, 6, 2)
    assert m.energy(ro.z).shape == (6, 6)
    loss, logs = m.loss(b, horizon=5)
    assert {"e_var", "e_sup", "pred", "recon"} <= set(logs)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_variance_penalty_is_zero_without_passive_windows():
    m = _model()
    b = _batch([False] * 6)
    var, sup = m.energy_terms(b, m.rollout(b.ctx, b.actions))
    assert var.item() == 0.0 and sup.item() > 0
    var.backward()  # still part of the graph, so a batch without passive windows does not break backward
    b = _batch([True] * 6)
    var, _ = m.energy_terms(b, m.rollout(b.ctx, b.actions))
    assert var.item() > 0


def test_variance_penalty_only_counts_passive_rows():
    m = _model()
    b = _batch([True, False, False, False, False, False])
    ro = m.rollout(b.ctx, b.actions)
    var, _ = m.energy_terms(b, ro)
    torch.testing.assert_close(var, m.energy(ro.z[:1]).var(dim=1, unbiased=False).mean())


def test_true_energy_uses_physical_states():
    env = make("pendulum")
    qp = env.sample_band(np.random.default_rng(0), (0.2, 0.5), 64)
    x = env.qp_to_obs(qp)
    norm = Normaliser.fit(x)
    m = _model()
    m.prepare(norm, torch.as_tensor(x[:, None]))
    got = m.true_energy(norm.norm(torch.as_tensor(x))).double().numpy()
    np.testing.assert_allclose(got, env.energy(qp) / env.E_ref, rtol=1e-4, atol=1e-5)


def test_tiny_train_lowers_loss(tmp_path, data_dir):  # noqa: F811
    cfg = build_run_config(str(ROOT / "configs/model/latent_ode.yaml"), FAST)
    t = Trainer(cfg, tmp_path / "c", data_dir)
    v0 = t.validate()
    # encoder + ODE + decoder learns slower than A in 50 tiny steps; a clear drop is enough here
    assert t.fit()["best_val_nmse"] < 0.7 * v0
    rec = _records(tmp_path / "c")[-1]
    assert "train_e_var" in rec and "train_e_sup" in rec
