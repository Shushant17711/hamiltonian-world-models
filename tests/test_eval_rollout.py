"""Rollout error and energy drift (Req 7.1-7.3): oracle model, divergence handling, CLI."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from hwm.data.dataset import Normaliser
from hwm.data.generate import load_split
from hwm.envs import make
from hwm.eval.energy import energy_drift
from hwm.eval.rollout import feature_map, feature_var, rollout_nmse
from hwm.models.base import WorldModel
from tests.test_pinn import data_dir  # noqa: F401
from tests.test_trainer import FAST, ROOT


class Oracle(WorldModel):
    """The ground-truth simulator behind the WorldModel interface (float64 throughout).

    ``bias`` adds a constant energy injection per step so drift can be made non-zero on purpose.
    """

    def __init__(self, env, normaliser, bias: float = 0.0):
        super().__init__(env.d_obs, env.d_u, d_z=env.d_obs)
        self.env, self.nrm, self.bias = env, normaliser, bias

    def encode(self, ctx):
        return self.nrm.denorm(ctx[:, -1].double())

    def step(self, z, u):
        qp = self.env.obs_to_qp(z.cpu().numpy())
        nxt = self.env.step(qp, u.cpu().numpy() * self.env.u_max)
        nxt[:, self.env.n :] *= 1.0 + self.bias
        return torch.as_tensor(self.env.qp_to_obs(nxt), dtype=z.dtype, device=z.device)

    def decode(self, z):
        return self.nrm.norm(z)


def _setup(name="pendulum", rng_seed=0, N=64):
    env = make(name)
    qp = env.sample_band(np.random.default_rng(rng_seed), (0.2, 0.5), N)
    x0 = env.qp_to_obs(qp)
    return env, Normaliser.fit(x0), x0


def test_feature_map_wraps_angles():
    env = make("acrobot")  # angle_dims (0, 1), obs = (q1, q2, q1dot, q2dot)
    x = torch.tensor([[0.3, -2.0, 1.0, 2.0]], dtype=torch.float64)
    y = x.clone()
    y[0, :2] += 2 * math.pi
    torch.testing.assert_close(feature_map(env, x), feature_map(env, y))
    assert feature_map(env, x).shape == (1, 6)


def test_oracle_has_zero_error_and_simulator_level_drift(data_dir):  # noqa: F811
    env = make("pendulum")
    norm = Normaliser.from_dir(data_dir)
    var = feature_var(env, load_split(data_dir, "train")["obs"])
    r = rollout_nmse(Oracle(env, norm), norm, env, load_split(data_dir, "test_in"), (1, 5, 10, 100), var)
    assert set(r) == {1, 5, 10}  # tiny split has T = 10: h = 100 does not fit and is skipped
    assert all(v["nmse"] < 1e-20 and v["diverged"] == 0 for v in r.values())
    _, norm, x0 = _setup()
    d = energy_drift(Oracle(env, norm), norm, env, x0, steps=1000)
    assert d["true"]["median"] < 1e-9 and d["true"]["n"] == 64 and d["learned"] is None


def test_drift_detects_energy_injection():
    env, norm, x0 = _setup()
    d = energy_drift(Oracle(env, norm, bias=1e-4), norm, env, x0, steps=200)
    assert d["true"]["median"] > 1e-4


def test_errors_are_capped_and_divergence_counted(data_dir):  # noqa: F811
    env = make("pendulum")
    norm = Normaliser.from_dir(data_dir)
    var = feature_var(env, load_split(data_dir, "train")["obs"])
    split = load_split(data_dir, "test_in")

    class Blowup(Oracle):
        def step(self, z, u):
            return torch.full_like(z, float("nan"))

    r = rollout_nmse(Blowup(env, norm), norm, env, split, (1,), var)
    assert r[1]["nmse"] == 100.0 and r[1]["diverged"] == len(split["obs"])
    _, norm2, x0 = _setup(N=5)
    d = energy_drift(Blowup(env, norm2), norm2, env, x0, steps=50)
    assert d["true"]["diverged"] == 5 and d["true"]["median"] == math.inf


def test_learned_energy_drift_reported_for_latent_ode():
    from hwm.config import Config
    from hwm.models import build

    env, norm, x0 = _setup(N=8)
    m = build(Config({"env": "pendulum", "model": {"name": "latent_ode", "hidden": 16, "layers": 1}}))
    m.prepare(norm, torch.as_tensor(x0[:, None]))
    d = energy_drift(m.double().eval(), norm, env, x0, steps=20)
    assert d["learned"] is not None and d["learned"]["n"] == 8
    assert np.isfinite(d["true"]["median"])


@pytest.mark.parametrize("model", ["mlp", "rssm", "ensemble"])
def test_evaluate_cli_writes_metrics(tmp_path, model):
    from tests.test_data_generate import TINY

    over = [o for o in FAST] + [f"data_root={tmp_path / 'data'}", f"results_root={tmp_path / 'res'}"]
    over += [
        f"data.{o}" for o in TINY + ["sizes.train=[16,30]", "sizes.val=[8,30]", "sizes.test_long=[4,120]"]
    ]
    over += ["train.steps=20", "train.device=cpu"]
    if model == "ensemble":  # 2 small E members (compile is CUDA-only, so this stays eager)
        over += ["model.members=2", "model.member.hidden=16", "model.member.h_hidden=16"]
    cfg = f"configs/model/{model}.yaml"
    subprocess.run(
        [sys.executable, "scripts/train.py", "--config", cfg, *over],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    name = "hamiltonian_ens" if model == "ensemble" else model
    run = tmp_path / "res" / f"pendulum-{name}-state-s0"
    r = subprocess.run(
        [sys.executable, "scripts/evaluate.py", str(run), "--drift-steps", "30", "--device", "cpu"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    m = json.loads((run / "eval" / "metrics.json").read_text())
    assert set(m["nmse"]) == {"test_in", "test_ood", "test_long", "test_long_ood"}
    assert set(m["nmse"]["test_long"]) == {"10", "100"} and set(m["nmse"]["test_in"]) == {"10"}
    assert m["drift"]["test_in"]["n"] == 4 and m["drift_steps"] == 30
    assert m["model"] == model
    if model == "ensemble":
        assert m["learned_drift"]["test_in"]["n"] == 4  # E members have a learned energy
        cal = m["calibration"]["test_in"]
        assert cal["horizons"] == [1, 5, 10] and len(cal["reliability"]["rmse"]) == 10
        assert -1 <= cal["spearman_pooled"] <= 1
    else:
        assert m["learned_drift"]["test_in"] is None and "calibration" not in m
