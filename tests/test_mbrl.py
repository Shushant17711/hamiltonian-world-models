"""Model-based RL loop (Req 8.3): a 2-iteration smoke run with tiny models."""

import json

import numpy as np
import pytest

from hwm.planning.mbrl import MBRL, mbrl_run_id
from scripts.run_mbrl import build_mbrl_config
from tests.test_trainer import ROOT

TINY = [
    "mbrl.random_episodes=2",
    "mbrl.train_steps_per_iter=5",
    "mbrl.batch=8",
    "mbrl.eval_at=[100,250]",
    "mbrl.eval_episodes=3",
    "mbrl.max_env_steps=250",
    "mbrl.stop_at_success=false",
    "mbrl.cem={horizon: 4, population: 16, elites: 4, iterations: 2, momentum: 0.1}",
    "train.device=cpu",
]


def _cfg(model, *extra):
    return build_mbrl_config(str(ROOT / f"configs/model/{model}.yaml"), TINY + list(extra))


@pytest.mark.parametrize(
    "model,extra",
    [
        ("mlp", ["model.hidden=16", "model.layers=1"]),
        (
            "ensemble",
            ["model.members=2", "model.member.name=mlp", "model.member.hidden=16", "model.member.layers=1"],
        ),
        ("rssm_pixels", ["model.embed=16", "model.hidden=16", "model.deter=16", "model.stoch=4"]),
    ],
)
def test_two_iteration_smoke_run(tmp_path, model, extra):
    cfg = _cfg(model, *extra)
    run = tmp_path / mbrl_run_id(cfg)
    out = MBRL(cfg, run).run(max_iterations=2)
    T = 200  # pendulum episode length
    # 2 random episodes (400 steps) -> eval at 100 and 250 -> all checkpoints done -> stop
    assert out["evaluated"] == [100, 250] and out["done"]
    recs = [json.loads(x) for x in (run / "mbrl.jsonl").read_text().splitlines()]
    assert [r["checkpoint"] for r in recs] == [100, 250]
    assert all(0 <= r["success_rate"] <= 1 and len(r["returns"]) == 3 for r in recs)
    with np.load(run / "buffer" / "train.npz") as z:
        assert z["obs"].shape == (2, T + 1, 2) and z["act"].shape == (2, T, 1)
        assert np.abs(z["act"]).max() <= 2.0


def test_collects_mpc_episodes_and_resumes(tmp_path):
    over = ["model.hidden=16", "model.layers=1", "mbrl.eval_at=[1000]", "mbrl.max_env_steps=1000"]
    cfg = _cfg("mlp", *over)
    run = tmp_path / "r"
    m = MBRL(cfg, run)
    m.run(max_iterations=1)  # 2 random + 1 MPC episode
    assert m.env_steps == 600 and m.grad_steps == 5
    with np.load(run / "buffer" / "train.npz") as z:
        first = z["obs"].copy()
    assert first.shape[0] == 3
    m2 = MBRL(cfg, run)
    m2.run(max_iterations=1)  # resumes: one more iteration, appends one episode
    assert m2.env_steps == 800 and m2.grad_steps == 10 and m2.iteration == 2
    with np.load(run / "buffer" / "train.npz") as z:
        np.testing.assert_array_equal(z["obs"][:3], first)  # earlier data untouched
        assert z["obs"].shape[0] == 4
    train = [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines()]
    assert [t["horizon"] for t in train] == [4, 8]  # horizon schedule advances per iteration


def test_config_merge_and_run_id():
    cfg = _cfg("ensemble", "obs_mode=pixels", "env=cartpole", "mbrl.cem.population=200")
    assert mbrl_run_id(cfg) == "mbrl-cartpole-hamiltonian_ens-pixels-s0"
    assert cfg.mbrl.cem.population == 200 and cfg.mbrl.cem.horizon == 4 and cfg.mbrl.beta == 1.0


def test_cartpole_random_warmup_stays_on_the_track(tmp_path):
    cfg = _cfg("mlp", "env=cartpole", "model.hidden=16", "model.layers=1")
    m = MBRL(cfg, tmp_path / "c")
    assert m.centering is not None
    obs, act, _ = m.run_episodes(5, "random")
    assert np.abs(obs[..., 0]).max() < 3.2 and np.abs(act).max() <= m.env.u_max


def test_episodes_end_when_the_state_leaves_the_valid_region(tmp_path):
    cfg = _cfg("mlp", "env=cartpole", "model.hidden=16", "model.layers=1")
    m = MBRL(cfg, tmp_path / "v")
    m.centering = None  # uncentred OU forces push some carts off the track
    m.mc["ou_sigma"] = 1.0
    obs, act, lengths = m.run_episodes(8, "random")
    assert (lengths < m.env.episode_len).any()
    for o, a, L in zip(obs, act, lengths, strict=True):
        assert np.abs(o[:L, 0]).max() <= m.env.x_limit  # every state before the last executed one is valid
        if L < m.env.episode_len:
            assert np.abs(o[L, 0]) > m.env.x_limit and (o[L:] == o[L]).all() and (a[L:] == 0).all()
    m.add(obs, act, lengths)
    assert m.env_steps == int(lengths.sum())
    from hwm.data.dataset import WindowDataset

    ds = WindowDataset(m.run_dir / "buffer", "train", 8, normaliser=m.normaliser, env_name="cartpole")
    assert len(ds) == int(np.clip(lengths - 8 + 1, 0, None).sum())
    b = ds.sample(256, __import__("torch").Generator().manual_seed(0))
    assert b.target_obs[..., 0].abs().max() <= m.env.x_limit + 0.5  # windows never run past an episode end
