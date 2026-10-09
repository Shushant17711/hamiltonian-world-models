"""Ensemble wrapper (Req 5.4): distinct members, zero disagreement for identical copies, training."""

import copy

import torch

from hwm.config import Config
from hwm.models import build
from hwm.train.trainer import Trainer, load_trained, run_id
from scripts.train import build_run_config
from tests.test_models_base import _batch
from tests.test_pinn import data_dir  # noqa: F401
from tests.test_trainer import FAST, ROOT, _records

MEMBER = {"name": "mlp", "hidden": 16, "layers": 1}


def _ens(M=3):
    torch.manual_seed(0)
    return build(Config({"env": "pendulum", "model": {"name": "ensemble", "members": M, "member": MEMBER}}))


def test_members_differ_after_init_and_are_reproducible():
    e = _ens()
    w = [m.f[0].weight for m in e.members]
    assert not torch.equal(w[0], w[1]) and not torch.equal(w[1], w[2])
    torch.testing.assert_close(_ens().members[2].f[0].weight, w[2])
    assert len(set(e.member_seeds)) == 3


def test_rollout_mean_members_and_disagreement():
    e = _ens()
    b = _batch(B=4, H=5)
    ro = e.rollout(b.ctx, b.actions)
    assert ro.members.shape == (3, 4, 6, 2) and ro.obs.shape == (4, 6, 2)
    assert ro.disagreement.shape == (4, 6) and (ro.disagreement[:, 1:] > 0).all()
    torch.testing.assert_close(ro.obs, ro.members.mean(0))
    # the default WorldModel path (encode/step/decode) agrees with the batched rollout
    z = e.encode(b.ctx)
    for t in range(5):
        z = e.step(z, b.actions[:, t])
    torch.testing.assert_close(e.decode(z), ro.obs[:, -1])


def test_disagreement_zero_for_identical_members():
    e = _ens()
    for i in (1, 2):
        e.members[i] = copy.deepcopy(e.members[0])
    b = _batch(B=4, H=5)
    assert e.rollout(b.ctx, b.actions).disagreement.abs().max() == 0


def test_bootstrap_resamples_and_training(tmp_path, data_dir):  # noqa: F811
    cfg = build_run_config(
        str(ROOT / "configs/model/ensemble.yaml"),
        FAST
        + ["model.members=3", "model.member.name=mlp", "model.member.hidden=32", "model.member.layers=2"],
    )
    assert run_id(cfg) == "pendulum-mlp_ens-state-s0"
    t = Trainer(cfg, tmp_path / "ens", data_dir)
    trajs = t.model.member_trajs
    assert len(trajs) == 3 and all(len(x) == 16 for x in trajs)
    assert not torch.equal(trajs[0], trajs[1])  # each member has its own resample
    b = t._sample_batch()
    assert b.ctx.shape[0] == 3 * t.tc.batch
    v0 = t.validate()
    assert t.fit()["best_val_nmse"] < 0.7 * v0
    assert "train_pred" in _records(tmp_path / "ens")[-1]
    model, _ = load_trained(tmp_path / "ens", device="cpu")
    torch.testing.assert_close(model._trajs, t.model._trajs.cpu())


def test_member_wise_backward_equals_summed_loss_gradient():
    e = _ens()
    b = _batch(B=6, H=4)
    torch.manual_seed(1)
    loss, _ = e.loss(b, 4)
    loss.backward()
    ref = [p.grad.clone() for p in e.parameters()]
    e.zero_grad()
    e.backward(b, 4)
    for g, r in zip((p.grad for p in e.parameters()), ref, strict=True):
        torch.testing.assert_close(g, r)


def test_a_non_finite_member_skips_alone():
    e = _ens()
    b = _batch(B=6, H=4)
    orig = e.members[1].loss
    e.members[1].loss = lambda batch, h: (orig(batch, h)[0] * float("nan"), {"loss": float("nan")})
    logs = e.backward(b, 4)
    assert logs["skipped_members"] == 1.0 and logs["loss"] == logs["loss"]  # finite average of the others
    assert all(p.grad is None for p in e.members[1].parameters())
    assert all(p.grad is not None for p in e.members[0].parameters())
