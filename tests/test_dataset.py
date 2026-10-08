"""Window dataset, normaliser and pixel contexts (Req 2.1, 3.1, 3.3)."""

import numpy as np
import pytest
import torch

from hwm.config import load_config
from hwm.data.dataset import Batch, Normaliser, WindowDataset
from hwm.data.generate import generate, load_split
from hwm.envs import make
from tests.test_data_generate import CONFIGS, TINY


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    cfg = load_config(CONFIGS / "pendulum.yaml", TINY + ["sizes.train=[6,30]"])
    return generate(cfg, tmp_path_factory.mktemp("data"))


def test_window_alignment(data_dir):
    raw = load_split(data_dir, "train")
    ds = WindowDataset(data_dir, "train", horizon=4, context=3)
    assert len(ds) == 6 * (31 - 4 - 3 + 1)
    i = 5 * ds.n_starts + 7  # trajectory 5, start 7 -> last ctx frame at step 9
    b = ds[i]
    obs = torch.as_tensor(raw["obs"][5], dtype=torch.float32)
    np.testing.assert_allclose(ds.normaliser.denorm(b.ctx), obs[7:10], atol=1e-5)
    np.testing.assert_allclose(b.target_obs, obs[10:14])
    np.testing.assert_allclose(b.actions * 2.0, raw["act"][5, 9:13], atol=1e-6)  # u_max = 2
    env = make("pendulum")
    np.testing.assert_allclose(b.rewards, env.reward(raw["obs"][5, 10:14], raw["act"][5, 9:13]), rtol=1e-5)
    assert b.passive.item() == raw["passive"][5]


def test_target_follows_context_under_simulator(data_dir):
    """target[0] is exactly one simulator step after ctx[-1] under actions[0]."""
    ds = WindowDataset(data_dir, "val", horizon=2)
    env = make("pendulum")
    b = ds.sample(8, torch.Generator().manual_seed(0))
    last = ds.normaliser.denorm(b.ctx[:, -1]).double().numpy()
    nxt = env.qp_to_obs(env.step(env.obs_to_qp(last), b.actions[:, 0].double().numpy() * env.u_max))
    np.testing.assert_allclose(nxt, b.target_obs[:, 0].numpy(), atol=1e-4)


def test_normaliser_fit_on_train_and_round_trip(tmp_path, data_dir):
    n = Normaliser.from_dir(data_dir)
    train = torch.as_tensor(load_split(data_dir, "train")["obs"], dtype=torch.float32)
    z = n.norm(train)
    assert z.reshape(-1, 2).mean(0).abs().max() < 1e-5
    assert (z.reshape(-1, 2).std(0) - 1).abs().max() < 1e-3
    np.testing.assert_allclose(n.denorm(z), train, atol=1e-5)
    n.save(tmp_path / "norm.json")
    m = Normaliser.load(tmp_path / "norm.json")
    torch.testing.assert_close(m.mean, n.mean)
    # every split uses the train statistics
    assert torch.equal(WindowDataset(data_dir, "test_ood", 4).normaliser.mean, n.mean)


def test_pixel_batch_shapes(data_dir):
    ds = WindowDataset(data_dir, "train", horizon=5, context=3, obs_mode="pixels")
    b = ds.sample(4)
    assert b.ctx.shape == (4, 3, 64, 64) and b.target.shape == (4, 5, 64, 64)
    assert b.actions.shape == (4, 5, 1) and b.rewards.shape == (4, 5) and b.target_obs.shape == (4, 5, 2)
    assert 0 <= b.ctx.min() and b.ctx.max() <= 1


def test_horizon_curriculum_and_validation(data_dir):
    ds = WindowDataset(data_dir, "train", horizon=4)
    ds.set_horizon(16)
    assert ds.sample(3).target.shape == (3, 16, 2)
    with pytest.raises(ValueError):
        ds.set_horizon(40)
    with pytest.raises(ValueError):
        WindowDataset(data_dir, "train", horizon=31)


def test_dataloader_compatible(data_dir):
    ds = WindowDataset(data_dir, "val", horizon=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=Batch.collate)
    b = next(iter(loader))
    assert b.ctx.shape == (8, 1, 2)
