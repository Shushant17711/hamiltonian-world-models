"""Pixel mode for model E and the RSSM (Req 3.3, 4.4, 5.1)."""

import pytest
import torch

from hwm.config import Config
from hwm.data.dataset import Batch
from hwm.models import build
from hwm.train.trainer import ParamBudgetError, Trainer
from scripts.train import build_run_config
from tests.test_pinn import data_dir  # noqa: F401
from tests.test_trainer import FAST, ROOT, _records

PIX_FAST = [o for o in FAST if not o.startswith("model.")] + [
    "train.steps=30",
    "train.batch=8",
    "train.val_windows=16",
    "train.val_horizon=4",
    "train.curriculum.horizons=[2,4]",
    "train.curriculum.at=[0.0,0.5]",
    "train.lr=1.0e-3",
]


def _pix(name, env="pendulum", **kw):
    torch.manual_seed(0)
    return build(Config({"env": env, "obs_mode": "pixels", "model": {"name": name, **kw}}))


def _pix_batch(B=2, H=3, d_u=1):
    g = torch.Generator().manual_seed(0)
    return Batch(
        ctx=torch.rand(B, 3, 64, 64, generator=g),
        actions=torch.rand(B, H, d_u, generator=g) * 2 - 1,
        target=torch.rand(B, H, 64, 64, generator=g),
        rewards=torch.randn(B, H, generator=g),
        passive=torch.zeros(B, dtype=torch.bool),
        target_obs=torch.randn(B, H, 2, generator=g),
    )


def test_hamiltonian_pixel_decoder_ignores_p_and_has_reward_head():
    m = _pix("hamiltonian", "acrobot", h_hidden=16)
    assert m.context == 3 and m.obs_shape == (64, 64)
    z = torch.randn(4, 4)
    z2 = z.clone()
    z2[:, 2:] += 5.0  # change p only
    torch.testing.assert_close(m.decode(z), m.decode(z2))
    z3 = z.clone()
    z3[:, :2] += 2 * torch.pi  # q enters through (cos, sin)
    torch.testing.assert_close(m.decode(z), m.decode(z3), atol=1e-5, rtol=1e-5)
    assert m.reward(z, torch.zeros(4, 1)).shape == (4,)
    assert _pix("hamiltonian", h_hidden=16).n_params() < 5_000_000


@pytest.mark.parametrize("name", ["hamiltonian", "rssm"])
def test_pixel_contract_and_loss_terms(name):
    m = _pix(name, **({"h_hidden": 16} if name == "hamiltonian" else {}))
    b = _pix_batch()
    ro = m.rollout(b.ctx, b.actions)
    assert ro.obs.shape == (2, 4, 64, 64) and ro.reward.shape == (2, 3)
    loss, logs = m.loss(b, horizon=3)
    assert "reward" in logs and ("ae" in logs if name == "hamiltonian" else "post_reward" in logs)
    loss.backward()
    assert loss.isfinite()


@pytest.mark.parametrize("cfg", ["hamiltonian_pixels", "rssm_pixels"])
def test_tiny_pixel_train_lowers_loss(tmp_path, data_dir, cfg):  # noqa: F811
    over = PIX_FAST + (["model.h_hidden=16"] if cfg.startswith("hamiltonian") else ["model.embed=32"])
    t = Trainer(build_run_config(str(ROOT / f"configs/model/{cfg}.yaml"), over), tmp_path / "p", data_dir)
    assert t.model.obs_mode == "pixels" and t.val_batch.ctx.shape[-2:] == (64, 64)
    v0 = t.validate()
    out = t.fit()
    assert out["best_val_nmse"] < 0.8 * v0
    assert "train_reward" in _records(tmp_path / "p")[-1]


def test_ensemble_budget_is_per_member(tmp_path, data_dir):  # noqa: F811
    over = PIX_FAST + ["obs_mode=pixels", "model.members=5", "model.member.h_hidden=16"]
    cfg = build_run_config(str(ROOT / "configs/model/ensemble.yaml"), over)
    t = Trainer(cfg, tmp_path / "ens", data_dir)  # 5 x 1.5M > 5M in total, each member < 5M
    assert t.n_params > 5_000_000
    with pytest.raises(ParamBudgetError):
        Trainer(
            build_run_config(str(ROOT / "configs/model/ensemble.yaml"), over + ["model.member.hidden=4096"]),
            tmp_path / "big",
            data_dir,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_autocast_only_around_cnns_and_rssm_memory():
    m = _pix("hamiltonian").cuda()
    b = _pix_batch(B=4).to("cuda")
    z = m.encode(b.ctx)
    assert z.dtype == torch.float32 and m.decode(z).dtype == torch.float32
    with torch.no_grad():
        assert m.step(z, b.actions[:, 0]).dtype == torch.float32  # integrator stays fp32
    d = _pix("rssm").cuda()
    torch.cuda.reset_peak_memory_stats()
    full = _pix_batch(B=128, H=32).to("cuda")
    loss, _ = d.loss(full, 32)
    loss.backward()
    assert torch.cuda.max_memory_allocated() < 7 * 2**30
