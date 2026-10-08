"""Model E: symplecticity, positive-definite A, bounded learned energy, contract, tiny train (Req 5.1-5.3)."""

import pytest
import torch

from hwm.config import Config
from hwm.models import build
from hwm.train.trainer import Trainer
from scripts.train import build_run_config
from tests.test_models_base import _batch
from tests.test_pinn import data_dir  # noqa: F401
from tests.test_trainer import FAST, ROOT, _records


def _model(env="acrobot", **kw):
    torch.manual_seed(0)
    m = build(Config({"env": env, "model": {"name": "hamiltonian", "hidden": 32, "h_hidden": 32, **kw}}))
    return m.double()


def _omega(n):
    eye, zero = torch.eye(n, dtype=torch.float64), torch.zeros(n, n, dtype=torch.float64)
    return torch.cat([torch.cat([zero, eye], 1), torch.cat([-eye, zero], 1)], 0)


@pytest.mark.parametrize("env,separable", [("acrobot", False), ("pendulum", False), ("pendulum", True)])
def test_conservative_step_is_symplectic(env, separable):
    m = _model(env, separable=separable, midpoint_iters=12)
    z = 0.5 * torch.randn(1, m.d_z, dtype=torch.float64)
    J = torch.autograd.functional.jacobian(lambda y: m.conservative_step(y[None], m.dt)[0], z[0])
    O = _omega(m.n)
    assert (J.T @ O @ J - O).abs().max() < 1e-4


def test_A_is_positive_definite_and_angle_features():
    m = _model("acrobot")
    assert m.angle_dims == (0, 1)
    q = 3 * torch.randn(64, 2, dtype=torch.float64)
    assert torch.linalg.eigvalsh(m.A(q)).min() > 0
    torch.testing.assert_close(m.H(q, q), m.H(q + 2 * torch.pi, q))  # angles enter via (cos, sin)
    with pytest.raises(ValueError):
        _model("pendulum", n_lat=1, angle_dims=[3])


def test_dissipation_starts_at_zero_and_is_psd():
    m = _model("acrobot")
    q = torch.randn(8, 2, dtype=torch.float64)
    assert m.R(q).abs().max() == 0
    torch.nn.init.normal_(m.K_net[-1].weight)
    assert torch.linalg.eigvalsh(m.R(q)).min() > -1e-12


def _energy_dev(m, steps, z):
    H0, worst, out = m.energy(z), 0.0, {}
    u = torch.zeros(z.shape[0], m.d_u, dtype=z.dtype)
    with torch.no_grad():
        for t in range(1, steps + 1):
            z = m.step(z, u)
            worst = max(worst, (m.energy(z) - H0).abs().max().item())
            out[t] = worst
    return out


def _bounded(m, short, long_):
    z = 0.3 * torch.randn(8, m.d_z, dtype=torch.float64)
    dev = _energy_dev(m, long_, z)
    assert dev[long_] < 10 * max(dev[short], 1e-12), (dev[short], dev[long_])


def test_learned_energy_bounded_fast():
    _bounded(_model("acrobot", dissipation=False), 100, 1000)


@pytest.mark.slow
@pytest.mark.parametrize("env", ["pendulum", "acrobot"])
def test_learned_energy_bounded_10k(env):
    _bounded(_model(env, dissipation=False), 1000, 10_000)  # the Req 5.3 criterion


def test_contract_and_losses():
    m = _model("pendulum").float()
    b = _batch(B=5, H=6)
    ro = m.rollout(b.ctx, b.actions)
    assert ro.z.shape == (5, 7, 2) and ro.obs.shape == (5, 7, 2)
    assert m.energy(ro.z[:, 0]).shape == (5,)
    loss, logs = m.loss(b, horizon=6)
    assert {"pred", "recon", "ae", "lat"} <= set(logs)
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert missing == ["V.4.bias"], missing  # a constant offset in H does not change the dynamics
    with pytest.raises(NotImplementedError):
        build(Config({"env": "pendulum", "obs_mode": "pixels", "model": {"name": "hamiltonian"}}))


def test_eval_rollout_builds_no_graph():
    m = _model("acrobot").float().eval()
    b = _batch(B=3, H=4, d_obs=4, d_u=1)
    with torch.no_grad():
        ro = m.rollout(b.ctx, b.actions)
    assert not ro.obs.requires_grad


def test_tiny_train_lowers_loss(tmp_path, data_dir):  # noqa: F811
    cfg = build_run_config(str(ROOT / "configs/model/hamiltonian.yaml"), FAST + ["model.h_hidden=32"])
    t = Trainer(cfg, tmp_path / "e", data_dir)
    v0 = t.validate()
    assert t.fit()["best_val_nmse"] < 0.7 * v0
    assert "train_lat" in _records(tmp_path / "e")[-1]


@pytest.mark.parametrize("env,separable", [("acrobot", False), ("cartpole", False), ("orbit", True)])
def test_closed_form_gradient_matches_autograd(env, separable):
    m = _model(env, separable=separable)
    torch.nn.init.normal_(m.V[-1].weight)
    q = torch.randn(16, m.n, dtype=torch.float64, requires_grad=True)
    p = torch.randn(16, m.n, dtype=torch.float64, requires_grad=True)
    gq, gp = torch.autograd.grad(m.H(q, p).sum(), (q, p))
    dq, dp = m.dH(q, p)
    torch.testing.assert_close(dq, gq)
    torch.testing.assert_close(dp, gp)
