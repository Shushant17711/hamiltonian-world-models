"""Model D (RSSM): KL floor, prior rollout shapes, determinism in eval, tiny train (Req 4.4)."""

import torch

from hwm.config import Config
from hwm.models import build
from hwm.models.rssm import kl_gauss
from hwm.train.trainer import Trainer
from scripts.train import build_run_config
from tests.test_models_base import _batch
from tests.test_pinn import data_dir  # noqa: F401
from tests.test_trainer import FAST, ROOT, _records


def _model(**kw):
    return build(Config({"env": "pendulum", "model": {"name": "rssm", "hidden": 32, "embed": 16, **kw}}))


def test_kl_gauss_matches_torch():
    m1, s1, m2, s2 = torch.randn(4, 3), torch.rand(4, 3) + 0.1, torch.randn(4, 3), torch.rand(4, 3) + 0.1
    ref = torch.distributions.kl_divergence(
        torch.distributions.Normal(m1, s1), torch.distributions.Normal(m2, s2)
    ).sum(-1)
    torch.testing.assert_close(kl_gauss(m1, s1, m2, s2), ref)


def test_prior_rollout_shapes_and_eval_determinism():
    m = _model()
    assert m.d_z == 230
    b = _batch(B=5, H=7)
    ro = m.rollout(b.ctx, b.actions)
    assert ro.z.shape == (5, 8, 230) and ro.obs.shape == (5, 8, 2) and ro.reward is None
    m.eval()
    a, c = m.rollout(b.ctx, b.actions).obs, m.rollout(b.ctx, b.actions).obs
    torch.testing.assert_close(a, c)  # eval uses distribution means
    m.train()
    assert not torch.equal(m.rollout(b.ctx, b.actions).obs, a)  # training samples


def test_kl_term_respects_free_nats_floor_and_backprops():
    m = _model()
    b = _batch(B=5, H=7)
    loss, logs = m.loss(b, horizon=7)
    assert logs["kl"] >= m.free_nats - 1e-6
    assert {"pred", "recon", "elbo_recon", "kl"} <= set(logs)
    loss.backward()
    assert all(p.grad is not None for n, p in m.named_parameters()), [
        n for n, p in m.named_parameters() if p.grad is None
    ]


def test_tiny_train_lowers_loss(tmp_path, data_dir):  # noqa: F811
    cfg = build_run_config(str(ROOT / "configs/model/rssm.yaml"), FAST)
    t = Trainer(cfg, tmp_path / "d", data_dir)
    v0 = t.validate()
    assert t.fit()["best_val_nmse"] < 0.7 * v0
    assert all(r["train_kl"] >= 1.0 - 1e-6 for r in _records(tmp_path / "d"))
