"""Maximal Lyapunov exponent (Req 9.2)."""

import json
import subprocess
import sys

import numpy as np

from hwm.envs import make
from hwm.eval.lyapunov import lyapunov_by_band, lyapunov_exponent
from tests.test_trainer import ROOT


def test_pendulum_is_regular():
    env = make("pendulum")
    lam = lyapunov_exponent(env, env.sample_band(np.random.default_rng(0), (0.1, 0.6), 8), steps=4000)
    assert abs(lam.mean()) < 0.05


def test_acrobot_is_chaotic_at_high_energy():
    env = make("acrobot")
    lam = lyapunov_exponent(env, env.sample_band(np.random.default_rng(1), (0.5, 0.8), 4), steps=1000)
    assert lam.mean() > 0.3


def test_by_band_and_cli(tmp_path):
    res = lyapunov_by_band(make("pendulum"), {"test_in": (0.1, 0.6)}, n_states=3, steps=200)
    assert (
        res["test_in"]["t_lambda"] == 1 / res["test_in"]["lambda"]
        and len(res["test_in"]["lambda_per_state"]) == 3
    )
    r = subprocess.run(
        [sys.executable, "scripts/lyapunov.py", "--env", "pendulum", "--steps", "50", "--out", str(tmp_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    d = json.loads((tmp_path / "lyapunov_pendulum.json").read_text())
    assert set(d["bands"]) == {"test_in", "test_ood"} and d["dt"] == 0.05
