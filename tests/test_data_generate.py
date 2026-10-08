"""Dataset generation (Req 2.1-2.3)."""

from pathlib import Path

import numpy as np
import pytest

from hwm.config import load_config
from hwm.data.generate import SPLITS, generate, load_split, ou_actions
from hwm.envs import make

CONFIGS = Path(__file__).parents[1] / "configs" / "data"
TINY = [
    "sizes.train=[8,10]",
    "sizes.val=[4,10]",
    "sizes.test_in=[4,10]",
    "sizes.test_ood=[4,10]",
    "sizes.test_long=[2,20]",
]


def _band_value(env, d):
    if env.name == "orbit":
        return env.elements(d["qp"][:, 0])[0]
    return d["energy"][:, 0] / env.E_ref


@pytest.mark.parametrize("name", ["pendulum", "cartpole", "acrobot", "orbit"])
def test_tiny_config_generates_every_split(tmp_path, name):
    cfg = load_config(CONFIGS / f"{name}.yaml", TINY)
    out = generate(cfg, tmp_path)
    assert out.parent.name == name and len(out.name) == 12
    env = make(name)
    for split in SPLITS:
        d = load_split(out, split)
        N, T = cfg.sizes[split]
        N = 2 * N if split == "test_long" else N
        assert d["obs"].shape == (N, T + 1, env.d_obs)
        assert d["qp"].shape == (N, T + 1, 2 * env.n)
        assert d["act"].shape == (N, T, env.d_u)
        assert d["energy"].shape == (N, T + 1)
        assert np.abs(d["act"]).max() <= env.u_max
        assert (d["act"][d["passive"]] == 0).all()
        assert env.valid_state(d["qp"]).all()
        np.testing.assert_allclose(d["obs"], env.qp_to_obs(d["qp"]))
    train, ood, long = (load_split(out, s) for s in ("train", "test_ood", "test_long"))
    lo, hi = cfg.bands.train
    v = _band_value(env, train)
    assert (v >= lo - 1e-9).all() and (v <= hi + 1e-9).all()
    v = _band_value(env, ood)
    assert (v > hi).all() and ood["ood"].all() and not train["ood"].any()
    assert long["passive"].all() and long["ood"].sum() == len(long["ood"]) // 2
    # passive trajectories conserve energy
    e = long["energy"]
    assert np.abs(e - e[:, :1]).max() / env.E_ref < 1e-5


def test_passive_fraction():
    cfg = load_config(CONFIGS / "pendulum.yaml", TINY + ["sizes.train=[100,5]"])
    from hwm.data.generate import build_split

    d = build_split(make("pendulum"), cfg, "train")
    assert d["passive"].sum() == 25


def test_same_hash_identical_data_and_skip(tmp_path):
    cfg = load_config(CONFIGS / "cartpole.yaml", TINY)
    a = generate(cfg, tmp_path / "a")
    b = generate(cfg, tmp_path / "b")
    assert a.name == b.name
    for s in SPLITS:
        da, db = load_split(a, s), load_split(b, s)
        for k in da:
            np.testing.assert_array_equal(da[k], db[k])
    mtime = (a / "train.npz").stat().st_mtime_ns
    assert generate(cfg, tmp_path / "a") == a
    assert (a / "train.npz").stat().st_mtime_ns == mtime  # skipped
    other = generate(cfg.with_overrides(["seed=1"]), tmp_path / "a")
    assert other != a


def test_ou_actions_statistics():
    rng = np.random.default_rng(0)
    u = ou_actions(rng, 2000, 200, 1, 1.0, 0.15, 0.3)
    assert abs(u.mean()) < 0.02
    stationary = 0.3 / np.sqrt(0.15 * 1.85)
    assert u.std() == pytest.approx(stationary, rel=0.1)  # clipping trims the tails a little
    lag1 = np.corrcoef(u[:, :-1].ravel(), u[:, 1:].ravel())[0, 1]
    assert lag1 == pytest.approx(0.85, abs=0.03)
