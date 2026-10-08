"""Shared trainer with model A (Req 4.1, 6.1-6.3)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from hwm.config import Config
from hwm.data.generate import generate
from hwm.train.trainer import ParamBudgetError, Trainer, horizon_at, load_trained
from scripts.train import build_run_config
from tests.test_data_generate import CONFIGS, TINY

ROOT = Path(__file__).parents[1]
FAST = [
    "train.steps=50",
    "train.batch=16",
    "train.eval_every=10",
    "train.val_windows=32",
    "train.val_horizon=8",
    "train.curriculum.horizons=[2,4,8]",
    "train.curriculum.at=[0.0,0.3,0.6]",
    "train.device=cpu",
    "model.hidden=32",
    "model.layers=2",
    "train.lr=3.0e-3",
]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    from hwm.config import load_config

    cfg = load_config(CONFIGS / "pendulum.yaml", TINY + ["sizes.train=[16,30]", "sizes.val=[8,30]"])
    return generate(cfg, tmp_path_factory.mktemp("data"))


def _cfg(*extra):
    return build_run_config(str(ROOT / "configs/model/mlp.yaml"), FAST + list(extra))


def _records(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_training_lowers_loss_and_writes_artifacts(tmp_path, data_dir):
    run = tmp_path / "run"
    t = Trainer(_cfg(), run, data_dir)
    v0 = t.validate()
    out = t.fit()
    assert out["step"] == 50
    recs = _records(run)
    assert [r["step"] for r in recs] == [10, 20, 30, 40, 50]
    assert [r["horizon"] for r in recs] == [2, 4, 4, 8, 8]  # H in force at the last step before each eval
    assert out["best_val_nmse"] < 0.5 * v0
    for f in ("config.yaml", "ckpt.pt", "ckpt_best.pt", "normaliser.json"):
        assert (run / f).exists(), f
    _, cfg = load_trained(run, device="cpu")
    assert cfg.model.name == "mlp" and cfg.data.env == "pendulum"
    # a finished run is not retrained
    assert Trainer(_cfg(), run, data_dir).fit()["step"] == 50
    assert len(_records(run)) == 5


def test_resume_matches_uninterrupted(tmp_path, data_dir):
    full = Trainer(_cfg(), tmp_path / "a", data_dir)
    full.fit()
    part = Trainer(_cfg(), tmp_path / "b", data_dir)
    assert part.fit(stop_after=25)["interrupted"]
    resumed = Trainer(_cfg(), tmp_path / "b", data_dir)
    out = resumed.fit()
    assert out["step"] == 50
    assert [r["step"] for r in _records(tmp_path / "b")] == [10, 20, 30, 40, 50]
    for (k, a), b in zip(full.model.state_dict().items(), resumed.model.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b, msg=k)


def test_param_budget_refused(tmp_path, data_dir):
    with pytest.raises(ParamBudgetError):
        Trainer(_cfg("model.hidden=2048", "model.layers=3"), tmp_path / "big", data_dir)


def test_config_change_refused_on_resume(tmp_path, data_dir):
    Trainer(_cfg(), tmp_path / "r", data_dir).fit(stop_after=10)
    with pytest.raises(ValueError, match="different config"):
        Trainer(_cfg("train.lr=1.0e-3"), tmp_path / "r", data_dir).fit()


def test_horizon_curriculum_schedule():
    c = Config({"horizons": [4, 8, 16, 32], "at": [0.0, 0.2, 0.4, 0.6]})
    assert [horizon_at(s, 100, c) for s in (0, 19, 20, 39, 40, 60, 99)] == [4, 4, 8, 8, 16, 32, 32]


def test_early_stopping_only_at_final_horizon(tmp_path, data_dir):
    run = tmp_path / "es"
    out = Trainer(_cfg("train.lr=0.0", "train.patience=1"), run, data_dir).fit()
    recs = _records(run)
    # lr=0: val never improves after the first eval; stopping waits for the final horizon (step >= 30)
    assert out["step"] == 40 and recs[-1]["horizon"] == 8


def test_train_script_cli(tmp_path):
    over = [o for o in FAST] + [f"data_root={tmp_path / 'data'}", f"results_root={tmp_path / 'res'}"]
    over += [f"data.{o}" for o in TINY + ["sizes.train=[16,30]", "sizes.val=[8,30]"]]
    r = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", "configs/model/mlp.yaml", *over],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "res" / "pendulum-mlp-state-s0" / "ckpt_best.pt").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_resume_on_cuda(tmp_path, data_dir):
    cfg = _cfg("train.device=cuda")
    Trainer(cfg, tmp_path / "g", data_dir).fit(stop_after=20)
    out = Trainer(cfg, tmp_path / "g", data_dir).fit()  # generator state reloads onto a CUDA generator
    assert out["step"] == 50
